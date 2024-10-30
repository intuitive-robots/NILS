import numpy as np
import torch
from deva.ext.SAM.automatic_mask_generator import SamAutomaticMaskGenerator
from matplotlib import pyplot as plt
from segment_anything import (
    SamPredictor,
    sam_model_registry,
    sam_model_registry_baseline,
)
from segment_anything.utils.transforms import ResizeLongestSide
from transformers import SamProcessor

from nils.specialist_models.detectors.utils import create_batches

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    num_chunks = (len(lst) + n - 1) // n
    chunked_lst = [lst[i * n:(i + 1) * n] for i in range(num_chunks)]

    return chunked_lst


class SAM():
    def __init__(self, model_type, sam_checkpoint, hf_path,
                 generator_settings, sam_hq_settings):

        self.processor = SamProcessor.from_pretrained(hf_path)


        if sam_hq_settings.sam_hq:
            print(f"Loading SAM-HQ model {sam_hq_settings.sam_hq_model_type} from {sam_hq_settings.sam_hq_checkpoint}")
            self.sam = sam_model_registry[sam_hq_settings.sam_hq_model_type](
                checkpoint=sam_hq_settings.sam_hq_checkpoint).to(torch.float16).to(device)
        else:
            print(f"Loading SAM model {model_type} from {sam_checkpoint}")
            self.sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint).to(torch.float16).to(device)
        # self.sam = self.sam.half()

        self.predictor = SamPredictor(self.sam)

        # self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=8,crop_n_layers=2,
        # crop_n_points_downscale_factor = 2, min_mask_region_area = 150)

        # Groot config
        # self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=128, crop_n_layers=0,
        #                                                 stability_score_thresh=0.95, box_nms_thresh=0.7,
        #                                                 crop_nms_thresh=0.7, crop_n_points_downscale_factor=1,
        #                                                 pred_iou_thresh=0.88,
        #                                                 crop_overlap_ratio=0.3413333333333333,
        #                                                 points_per_batch=64,
        #                                                 stability_score_offset=1.0)
        #

        self.mask_generator = SamAutomaticMaskGenerator(self.sam, **generator_settings)

        self.batch_size = 2

    def set_mask_generator(self, custom_point_grid):
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=None, point_grids=custom_point_grid)

    def set_mask_generator_settings(self, kwargs):
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, **kwargs)

    def segment_single(self, images, xyxy=None, point_labels=None, boxes = None,multimask_output=True):
        """Segment a batch of images with batch size 1

        Args:
            images: batch of images of shape (B, H, W, C)
            xyxy: batch of points of shape (B, N, 2)
            point_labels:
            boxes: batch of bounding boxes of shape (B, N, 4)
            multimask_output:

        Returns:

        """
        masks = []
        scores = []

        if point_labels is None and xyxy is not None:
            point_labels = [np.ones(len(point)) for point in xyxy]

        for i in (range(images.shape[0])):
            with torch.inference_mode():
                with torch.autocast(device_type=device, dtype=torch.float16):
                    
                    self.predictor.set_image(images[i])
                    mask, score, sized_masks = self.predictor.predict(
                        point_coords=xyxy[i] if xyxy is not None else None,
                        point_labels=point_labels[i] if point_labels is not None else None,
                        box =boxes[i] if boxes is not None else None,
                        multimask_output=multimask_output,
                        
                    )

            masks.append(mask)
            scores.append(score)

        return masks, scores

    def segment_no_label(self, images):
        masks = []
        for i in (range(images.shape[0])):
            with torch.inference_mode():
                with torch.autocast(device_type=device, dtype=torch.float16):
                    mask = self.mask_generator.generate(images[i])
            masks.append(mask)

        return masks

    def plot_segmentation_masks(self, image, masks):
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        if len(masks) == 0:
            return
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
        plt.axis('off')
        plt.show()

    def segment(self, images, xyxy=None, input_points = None, point_labels=None, multimask_output=True):
        """Args:
            images (np.ndarray):(B, H, W, C)
            xyxy (List[np.ndarray]): List with bounding boxes of len B. Each element is a np.ndarray of shape (N, 4)
            point_labels:
            multimask_output:

        Returns:

        """
        # images = torch.tensor(images)

        bsz = min(self.batch_size, images.shape[0])

        images_split = create_batches(bsz, images)

        xyxy_split = list(chunks(xyxy, bsz))
        if point_labels:
            point_labels_split = list(chunks(point_labels, bsz))

        resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        masks = []
        scores = []
        for i in (range(len(images_split))):
            image_batch = images_split[i]

            # Use boxes
            if xyxy_split[i][0].shape[-1] == 4:
                batched_input = [{"image": self.prepare_image(image, resize_transform, self.sam),
                                  "boxes": resize_transform.apply_boxes_torch(torch.tensor(box, device=self.sam.device),
                                                                              image.shape[:2]),
                                  "original_size": image.shape[:2]} for image, box in zip(image_batch, xyxy_split[i])]
            else:
                batched_input = [{"image": self.prepare_image(image, resize_transform, self.sam),
                                  "point_coords": resize_transform.apply_coords_torch(
                                      torch.tensor(point, device=self.sam.device).unsqueeze(1),
                                      # Weird SAM bug needs extra dim
                                      image.shape[:2]),
                                  "point_labels": torch.tensor(labels).unsqueeze(1),
                                  "original_size": image.shape[:2]} for image, point, labels in
                                 zip(image_batch, xyxy_split[i], point_labels_split[i])]

            # boxes = [[list(row) for row in array] for array in list(xyxy_split[i])]
            # inputs = self.processor(images=image_batch, input_boxes=boxes, return_tensors="pt")
            # inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.inference_mode():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    # outputs = self.detection_model(**inputs)
                    batched_output = self.sam(batched_input, multimask_output=multimask_output)

            # target_sizes = (torch.Tensor([image.squeeze().shape[0:2]])).repeat_interleave(self.config.detector.bsz,
            #                                                                               dim=0).to(device)

            # target_sizes = (torch.Tensor([image.squeeze().shape[1:3]])).repeat(bsz, 1).to(device)
            for output in batched_output:
                masks.append(output["masks"].detach().cpu().numpy())
                scores.append(output["iou_predictions"].detach().cpu().to(torch.float16).numpy())

        return masks, scores

    def prepare_image(self, image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device.device)
        return image.permute(2, 0, 1).contiguous()
