import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from matplotlib import patches
from supervision import ColorPalette
from supervision.draw.color import Color, ColorPalette

from nils.specialist_models.sam2.utils.amg import remove_small_regions

green = (15, 157, 88)
red = (219, 68, 55)
gray = (125, 125, 200)

possible_colors = [green, red, gray]
def resize_image_and_detections(image,detections: sv.Detections, size):

    new_height = size[0]
    new_width = size[1]
    original_height, original_width = image.shape[:2]

    width_scale = new_width / original_width
    height_scale = new_height / original_height

    detections = copy.deepcopy(detections)
    detections.xyxy[:, 0] = detections.xyxy[:, 0] * width_scale
    detections.xyxy[:, 1] = detections.xyxy[:, 1] * height_scale
    detections.xyxy[:, 2] = detections.xyxy[:, 2] * width_scale
    detections.xyxy[:, 3] = detections.xyxy[:, 3] * height_scale

    image = cv2.resize(image, (new_width, new_height))

    return image, detections


def to_detections(image_source: np.ndarray, boxes: torch.Tensor) -> sv.Detections:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return sv.Detections(xyxy=xyxy)


def annotate(image_source: np.ndarray, detections: sv.Detections, labels) -> np.ndarray:
    box_annotator = BoxAnnotator(thickness=2, text_scale=0.8,displacement_x = 0, displacement_y = -0)
    #annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = image_source
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    #annotated_frame_np = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame



def plot_boxes_np_sv(image, boxes, labels=None, scores=None, return_image = False):
   
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.numpy()
    if isinstance(labels, str):
        labels = np.array([labels])
    if isinstance(scores, float):
        scores = np.array([scores])[None][None]
        
        
        
    
    xyxy = boxes

    invalid_indices = np.where(xyxy < 0)[0]
    
    
    target_size = 800
    max_side = max(image.shape)
    scale = target_size / max_side
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

    
    xyxy[xyxy < 0] = 0
    
    xyxy = xyxy * scale
    
    xyxy = xyxy.astype(int)
    
    if labels is not None:
        labels = labels[np.where(~np.isnan(scores))[0]]
    
    
    if scores is not None:
        scores = scores[~np.isnan(scores)]
        
        labels = [f"{label} {confidence:0.2f}" for label,confidence in zip(labels,scores)]
    
    if len(xyxy) == 0:
        return image
    detections_object = sv.Detections(xyxy=xyxy.astype(int))
    
    image= np.array(image)
    

    
    annotated_frame = annotate(image, detections_object, labels)
    
    return annotated_frame
    
class BoxAnnotator:
    """A class for drawing bounding boxes on an image using detections provided.

    Attributes:
        color (Union[Color, ColorPalette]): The color to draw the bounding box,
            can be a single color or a color palette
        thickness (int): The thickness of the bounding box lines, default is 2
        text_color (Color): The color of the text on the bounding box, default is white
        text_scale (float): The scale of the text on the bounding box, default is 0.5
        text_thickness (int): The thickness of the text on the bounding box,
            default is 1
        text_padding (int): The padding around the text on the bounding box,
            default is 5

    """

    def __init__(
        self,
        color= ColorPalette.default(),
        thickness: int = 2,
        text_color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        displacement_x = 10,
        displacement_y = -10,
    ):
        self.color= color
        self.thickness: int = thickness
        self.text_color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        
        self.displacement_x = displacement_x
        self.displacement_y = displacement_y
        

    def annotate(
        self,
        scene: np.ndarray,
        detections,
        labels ,
        skip_label: bool = False,
    ) -> np.ndarray:
        """Draws bounding boxes on the frame using the detections provided.

        Args:
            scene (np.ndarray): The image on which the bounding boxes will be drawn
            detections (Detections): The detections for which the
                bounding boxes will be drawn
            labels (Optional[List[str]]): An optional list of labels
                corresponding to each detection. If `labels` are not provided,
                corresponding `class_id` will be used as label.
            skip_label (bool): Is set to `True`, skips bounding box label annotation.

        Returns:
            np.ndarray: The image with the bounding boxes drawn on it

        Example:
            ```python
            >>> import supervision as sv

            >>> classes = ['person', ...]
            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> box_annotator = sv.BoxAnnotator()
            >>> labels = [
            ...     f"{classes[class_id]} {confidence:0.2f}"
            ...     for _, _, confidence, class_id, _
            ...     in detections
            ... ]
            >>> annotated_frame = box_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections,
            ...     labels=labels
            ... )
            ```
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
            idx = class_id if class_id is not None else i
            color = (
                self.color.by_idx(idx)
                if isinstance(self.color, ColorPalette)
                else self.color
            )
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            if skip_label:
                continue

            text = (
                f"{class_id}"
                if (labels is None or len(detections) != len(labels))
                else labels[i]
            )

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            text_x = x1 + self.displacement_x
            text_y = y1 - self.displacement_y

            text_background_x1 = x1
            text_background_y1 = y1 - 2 * self.displacement_y - text_height

            text_background_x2 = x1 + 2 * self.displacement_x + text_width
            text_background_y2 = y1

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene


def postprocess_masks(masks,max_size = 512*512):


    cleaned_masks = [[remove_small_regions(mask, area_thresh=max_size, mode="holes")[0] for mask in masks_frame] for masks_frame in masks]
    cleaned_masks = [[remove_small_regions(mask, area_thresh=max_size, mode="islands")[0] for mask in cleaned_masks_frame] for cleaned_masks_frame in cleaned_masks]

    return np.stack(cleaned_masks)

def plot_boxes_np(image, boxes, labels=None, scores=None, return_image = False):
    
    
    
    # if labels is not None:
    #     nan_filter = ~np.isnan(scores)
    #     boxes = boxes[nan_filter]
    #     labels = labels[nan_filter]
    #     scores = scores[nan_filter]
    #     
    #     class_ids = np.array([np.where(labels == label)[0][0] for label in labels])
    # 
    #     detections = sv.Detections(boxes, confidence=scores,class_id=class_ids)
    #     
    #     max_side = max(image.shape)
    #     target_size = 512
    #     scale = target_size / max_side
    #     
    #     image, detections = resize_image_and_detections(np.array(image),detections,(int(image.shape[0]*scale),int(image.shape[1]*scale)))
    #     box_annotator = sv.BoxAnnotator(thickness=2)
    # 
    #     #slabels = [f"{label} {confidence:0.2f}" for label,confidence in zip(labels,detections.confidence)]
    # 
    # 
    # 
    #     annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    #     image = annotated_frame
    # 
    
        # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Create a Rectangle patch for each box and add it to the axes
    for i, box in enumerate(boxes):
        if box.sum() == 0:
            continue
            
            
        x1, y1, x2, y2 = box
        center = (x1 + x2) / 2, (y1 + y2) / 2
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Add the score and label text next to the box if they are provided
        if scores is not None and labels is not None:
            plt.text(x1,y1, f'{labels[i]}: {round((scores[i]),2):.2f}', bbox=dict(facecolor='white', alpha=0.5))
    
    
    
    if not return_image:
        plt.show()
    else:
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.tight_layout()
        plt.axis('off')
        ax.axis('off')
        ax.set_axis_off()
        fig.canvas.draw()
        image  = np.array(fig.canvas.renderer.buffer_rgba())
        #image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return image
    




def plot_detection_boxes(image, classes, detections: sv.Detections, labels = None):

    image = image.copy()



    plt.figure(figsize=(10, 10))

    new_width = 512
    new_height = 512
    image, detections = resize_image_and_detections(image,detections,(512,512))
    box_annotator = sv.BoxAnnotator(thickness=4)
    if labels is None:

        labels = [
            f"{classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
    else:
        labels = [f"{label} {confidence:0.2f}" for label,confidence in zip(labels,detections.confidence)]

    if isinstance(detections.class_id[0], str):
        detections.class_id = None
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    plt.imshow(annotated_frame)
    plt.axis('off')
    plt.tight_layout()
    # Leave space at the top for the common title
    # plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.show()

    return annotated_frame

def resize_image_to_same_shape(source_img, reference_img=None, reference_size=None):
    # if source_img is larger than reference_img
    if reference_img is None and reference_size is None:
        raise ValueError("Either reference_img or reference_size must be specified.")
    if reference_img is not None:
        reference_size = (reference_img.shape[0], reference_img.shape[1])
    if source_img.shape[0] > reference_size[0] or source_img.shape[1] > reference_size[1]:
        result_img = cv2.resize(source_img, (reference_size[0], reference_size[1]), interpolation=cv2.INTER_NEAREST)
    else:
        result_img = cv2.resize(source_img, (reference_size[0], reference_size[1]), interpolation=cv2.INTER_NEAREST)
    return result_img


def plot_segmentation_masks(image, detections: sv.Detections):

    if isinstance(detections,sv.Detections):
        mask_annotator = sv.MaskAnnotator()

        annotated_frame = mask_annotator.annotate(scene=image.copy(), detections=detections)
        plt.imshow(annotated_frame)
        plt.show()
        return annotated_frame
    else:
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        if len(detections) == 0:
            return
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((detections[0].shape[0], detections[0].shape[1], 4))
        img[:, :, 3] = 0
        for m in detections:

            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
        plt.axis('off')
        plt.show()

    #plt.imshow(annotated_frame)
    #plt.show()

def add_segmentation_masks(image,masks):
    img = np.zeros((masks[0].shape[0], masks[0].shape[1], 3))
    for i,m in enumerate(masks):
        color_mask = possible_colors[i]
        img[m] = color_mask
    img = img.astype(np.uint8)
    res = cv2.addWeighted(image,0.5,img,0.5,0.0)

    return res





def plot_detections_and_masks(image, classes, detections: sv.Detections):
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_frame = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    plt.imshow(annotated_frame)
    plt.show()
    return annotated_frame


def plot_segmentation_sequence(robot_masks, object_masks, points_objects=None, points_robots=None, orig_images=None,
                               cmap='gray', title=None, figsize=(15, 5), only_object = False):

    only_object = robot_masks is None
    if robot_masks is not None:
        num_images = len(robot_masks)
    else:
        num_images = len(object_masks)
    nrows = 2 if orig_images is not None else 1

    figsize = (num_images * 4, 4 * nrows)
    # Check if titles are provided and their length matches the number of images
    if title is not None:
        plt.suptitle(title, fontsize=16)

    # Create a new figure with the specified size
    plt.figure(figsize=figsize)


    # If base images are provided plot them


    if orig_images is not None:
        for i in range(len(orig_images)):
            robot_patches = []
            highlight_patches = []
            if points_objects is not None:
                center_static =  points_objects[i].squeeze()
            if points_robots is not None:
                robot_tcp_static = points_robots[i]
            plt.subplot(nrows, num_images, i + 1)
            plt.imshow(orig_images[i], cmap=cmap)
            plt.axis('off')  # Turn off axis labels

            if points_objects is not None:
                highlight_patch = patches.Circle(
                    (center_static[0], center_static[1]),
                    4,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                highlight_patches.append(highlight_patch)
            if points_robots is not None:
                robot_patches += [patches.Circle(
                    (tcp_static[0], tcp_static[1]),
                    4,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                ) for tcp_static in robot_tcp_static]

            for patch_object in highlight_patches:
                plt.gca().add_patch(patch_object)
            for patch in robot_patches:
                plt.gca().add_patch(patch)



    # Plot each image in the row
    for i in range(num_images):
        plt.subplot(nrows, num_images, (i + 1) + (num_images if nrows == 2 else 0))
        plt.imshow(((robot_masks[i] if not only_object else 0) + object_masks[i]), cmap=cmap)
        plt.axis('off')  # Turn off axis labels

        # Display titles if provided

    # Adjust layout to prevent clipping of titles
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show the plot
    plt.show()




def plot_points_on_img(image,points,radius = 2):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.axis('off')
    patches_lst = [patches.Circle(
        (point[0], point[1]),
        radius,
        linewidth=4,
        edgecolor='r',
        facecolor='r'
    ) for point in points]


    for patch in patches_lst:
        plt.gca().add_patch(patch)

    plt.tight_layout()
    plt.show()

def plot_segmentations_only(mask):
    image = np.zeros(mask.shape[1:], dtype=np.bool_)

    for m in mask:
        image = image + m

    plt.imshow(image)
    plt.show()

    # if isinstance(mask,sv.Detections):

    #     image = np.ones(mask.mask.shape[1:],dtype = np.uint8)* 255
    #
    #     plot_segmentation_masks(image,mask)
    # else:
    #     #mask = mask.any(axis=-1)
    #     plt.imshow(mask)
    #     plt.show()


def create_colored_segmentation_mask(mask, color=(255, 0, 0)):
    assert len(color) == 3

    mask = mask.squeeze()
    height, width = mask.shape

    base_images = np.ones((height, width, 3), dtype=np.uint8)

    mask_colored = base_images.copy()
    mask_colored[mask.squeeze()] = color

    return mask_colored


def create_batches(bsz, images):
    if images.shape[0] % bsz != 0:
        num_chunks = (len(images) + bsz - 1) // bsz
        last_chunk_size = len(images) - (num_chunks - 1) * bsz
        images_split = np.array_split(images, [bsz * i for i in range(1, num_chunks)])
    else:
        images_split = np.array_split(images, images.shape[0] // bsz)
    return images_split


def compute_iou(reference_mask, segmentation_masks, weights=None):
    # Convert to binary arrays
    reference_mask = reference_mask.astype(bool)
    segmentation_masks = segmentation_masks.astype(bool)

    #Downscale masks for faster computations:
    #reference_mask = cv2.resize(reference_mask.astype(np.uint8), (reference_mask.shape[0]//2, reference_mask.shape[1]//2), interpolation=cv2.INTER_NEAREST).astype(bool)
    #segmentation_masks = cv2.resize(segmentation_masks.astype(np.uint8), (segmentation_masks.shape[0]//2, segmentation_masks.shape[1]//2), interpolation=cv2.INTER_NEAREST).astype(bool)

    if reference_mask.ndim == 2:
        reference_mask = reference_mask[None, ...]

    # if weights is not None:
    #   reference_mask = reference_mask*weights
    # Calculate intersection and union

    reference_mask = reference_mask[:, None, ...]

    intersection = np.logical_and(reference_mask, segmentation_masks)
    union = np.logical_or(reference_mask, segmentation_masks)

    iou = np.zeros((intersection.shape[:2]))


    # if union.sum() == 0 or intersection.sum() == 0:
    #    return np.zeros((intersection.shape[0]))

    if weights is not None:
        intersection = intersection * weights

        union = union * weights

    # Compute IoU
    union_sum = union.sum(axis=(-2, -1))
    intersection_sum = intersection.sum(axis=(-2, -1))

    non_zero_union_mask = union_sum != 0
    non_zero_intersection_mask = intersection_sum != 0
    valid_mask = np.logical_and(non_zero_union_mask, non_zero_intersection_mask)


    iou[valid_mask] = intersection_sum[valid_mask] / union_sum[valid_mask]

    # iou = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2))

    return iou
