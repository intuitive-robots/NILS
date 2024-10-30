import numpy as np
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os

from nils.specialist_models.detectors.utils import create_batches
from nils.specialist_models.sam2.build_sam import build_sam2
from nils.specialist_models.sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2():
    def __init__(self):

        checkpoint = "checkpoints/sam2_hiera_large.pt"



        cur_dir = os.path.dirname(__file__)
        checkpoint_dir = os.path.join(cur_dir,checkpoint)
        self.model = SAM2ImagePredictor(build_sam2(None,checkpoint_dir))
        



    def segment(self, images, xyxy, box_labels=None, input_size=1024, better_quality=False, bsz=32, threshold=0.5):


        boxes = xyxy



        batched_images = (create_batches(bsz, np.array(images)))
        batched_boxes = (create_batches(bsz, np.array(boxes)))



        masks = []
        ious = []

        for i in tqdm(range(len(batched_images)), disable=True):
            cur_images = list(batched_images[i])
            boxes = batched_boxes[i]


            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

                self.model.set_image_batch(cur_images)
                predicted_masks,predicted_iou,_ = self.model.predict_batch(None,
                                                                      None,
                                                                      box_batch=boxes,
                                                                      multimask_output=True)




            torch.cuda.empty_cache()

            predicted_iou = np.stack(predicted_iou)
            predicted_masks = np.stack(predicted_masks)

            if predicted_masks.ndim == 4:
                predicted_masks = predicted_masks[:,None,...]
            if predicted_iou.ndim == 2:
                predicted_iou = predicted_iou[:,None,...]

            sorted_ids = np.argsort(predicted_iou, axis=-1)[:, :, [1], ...][::-1]
            predicted_iou_cpu = np.take_along_axis(predicted_iou, sorted_ids, axis=2)



            best_masks = np.take_along_axis(predicted_masks, sorted_ids[..., None, None], axis=2)
            best_masks = best_masks[:, :, 0]





            masks.append(best_masks)
            ious.append(predicted_iou_cpu)

        all_masks = np.concatenate(masks, axis=0).astype(bool)
        all_ious = np.concatenate(ious, axis=0)

        # all_masks = F.interpolate((all_masks).float(), size=(h, w), mode="nearest").numpy().astype(bool)

        return all_masks,all_ious

