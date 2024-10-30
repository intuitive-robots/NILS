import copy
import itertools
from collections import defaultdict

import cv2
import numpy as np
import torch
from supervision import Detections
from torchvision.ops import nms, box_iou

from nils.utils.utils import score_nl_object_proposals, compute_detection_cooccurrence, \
    get_best_detection_per_class
from nils.utils.plot import crop_images_with_boxes
from nils.utils.visualizer import Visualizer


def get_best_frame_objectness(images, bsz, n_boxes, owl, owl_text_classifier):
    n_boxes = 15
    region_proposals, objectness, grounding_scores, class_embeds = owl.get_region_proposals(images,
                                                                                            bsz,
                                                                                            n_boxes=n_boxes,
                                                                                            text_classifier=owl_text_classifier)

    region_proposals = torch.tensor(region_proposals).float()
    objectness = torch.tensor(objectness)

    indices_to_keep = [nms(region_proposals[i], objectness[i], 0.3) for i in range(len(region_proposals))]

    objectness_filtered = [[objectness[i][j].item() if j in indices_to_keep[i] else 0 for j in range(n_boxes)]
                           for i
                           in range(len(region_proposals))]

    objectness = np.array(objectness_filtered)
    region_proposals = region_proposals.numpy().astype(np.int32)
    # objectness = objectness.numpy()

    high_confidence_objects = []

    high_conf_scores = []
    for frame in range(len(region_proposals)):
        high_confidence_objects.append((objectness[frame] > 0.3).sum())
        high_conf_scores.append(np.sum(objectness[frame][objectness[frame] > 0.3]))

    high_confidence_objects_scaled = np.array(high_confidence_objects) / np.max(high_confidence_objects)
    high_conf_scores_scaled = np.array(high_conf_scores) / np.max(high_conf_scores)

    best_frame = np.argmax(np.sum(objectness, axis=1))

    return best_frame


def create_som_annotated_image(annotated_frame, bsz, cur_frame_idx, use_som_mask):
    max_side = max(annotated_frame.shape)
    reshape_factor = 700 / max_side
    annotated_frame = cv2.resize(annotated_frame, (
        int(annotated_frame.shape[1] * reshape_factor), int(annotated_frame.shape[0] * reshape_factor)))
    annotated_frame_orig = np.array(annotated_frame)[None, ...]
    som_visualizer = Visualizer(annotated_frame, metadata=None, scale=1.5)
    image_area = annotated_frame.shape[0] * annotated_frame.shape[1]
    objectness_thresh = max(0.25, np.sort(objectness[cur_frame_idx])[::-1][8])
    objectness_thresh = 0.35
    good_boxes = region_proposals[cur_frame_idx][objectness[cur_frame_idx] >= objectness_thresh]
    good_boxes_idx = np.where(objectness[cur_frame_idx] >= objectness_thresh)[0]
    box_areas = (good_boxes[:, 2] - good_boxes[:, 0]) * (good_boxes[:, 3] - good_boxes[:, 1])
    large_boxes = box_areas > 0.5 * image_area
    good_boxes = good_boxes[~large_boxes]
    good_boxes_idx = good_boxes_idx[~large_boxes]
    mask = \
        self.segmentation_model.segment(annotated_frame_orig, good_boxes[None, ...] * reshape_factor,
                                        bsz=bsz)[
            0]
    for idx, box in enumerate(good_boxes):
        # mask = np.zeros((annotated_frame.shape[0], annotated_frame.shape[1]))
        # box = box * reshape_factor
        # box = box.astype(int)

        # mask[box[1]:box[3], box[0]:box[2]] = 1

        # img_annotated = som_visualizer.draw_binary_mask_with_box(box, text=str(idx), label_mode=1, alpha=0.2,
        #                                                         anno_mode=["Mark"])

        anno_mode = ["Mark"]
        if use_som_mask:
            anno_mode += ["Mask"]

        img_annotated = som_visualizer.draw_binary_mask_with_number(mask[idx], text=str(idx + 1),
                                                                    label_mode=1,
                                                                    alpha=0.0,
                                                                    anno_mode=anno_mode)
    return img_annotated


def get_clip_class_scores(clip_model, detections, init_obj_det_frames, ov_prompts):
    cls = clip_model.compute_text_embeddings(ov_prompts)
    all_cropped_images = []
    n_cropped_images_per_frame = []
    for frame in range(len(init_obj_det_frames)):
        cur_image = init_obj_det_frames[frame]
        cropped_images = np.stack([crop_images_with_boxes(cur_image, cur_box, resize=True,
                                                          resize_dim=
                                                          clip_model.model.visual.image_size[
                                                              0],
                                                          padding=2) for cur_box in
                                   detections[frame].xyxy])
        all_cropped_images.append(cropped_images)
        n_cropped_images_per_frame.append(len(cropped_images))
    all_cropped_images = np.concatenate(all_cropped_images)
    class_scores = clip_model.predict_objects(all_cropped_images, bsz=64)
    split_class_scores = np.array_split(class_scores, np.cumsum(n_cropped_images_per_frame))
    if len(split_class_scores[-1]) == 0:
        split_class_scores = split_class_scores[:-1]
    all_clip_class_scores = split_class_scores
    return all_clip_class_scores


def retrieve_surface_object_and_possible_surface_objects(detection, image_area, vlm_obj_names):
    if 0 in detection.class_id:
        surface_obj_score = detection.confidence[detection.class_id == 0].max()

        surface_obj_box = detection.xyxy[detection.class_id == 0][0]
        surface_obj_area = (surface_obj_box[2] - surface_obj_box[0]) * (
                surface_obj_box[3] - surface_obj_box[1])
    else:
        surface_obj_score = 0
        surface_obj_area = 0
    image_area

    surface_obj = None

    if surface_obj_area > 0.2 * image_area:
        surface_obj = vlm_obj_names[0]
    else:
        surface_obj_score = 0.

    surface_obj_idx = detection.class_id == 0
    detection = detection[~surface_obj_idx]

    possible_surface_obj = None
    possible_surface_obj_score = 0

    object_areas = [(det[2] - det[0]) * (det[3] - det[1]) for det in detection.xyxy]

    possible_surface_objects_idx = np.array(object_areas) > 0.6 * image_area
    possible_surface_objects_det = detection[possible_surface_objects_idx]

    if len(possible_surface_objects_det) > 0:
        best_surface_obj_idx = possible_surface_objects_det.confidence.argmax()

        best_surface_obj_score = possible_surface_objects_det.confidence[best_surface_obj_idx]
        possible_surface_obj = vlm_obj_names[1:][best_surface_obj_idx]

    return surface_obj, surface_obj_score, possible_surface_obj, possible_surface_obj_score


def get_temporal_coocurence(init_obj_det_frames, obj_set_detections, vlm_object_names):
    all_class_scores = {}
    all_nl_classes_to_compare = list(itertools.chain.from_iterable(vlm_object_names))
    all_frame_detections_concat = []
    all_matching_classes = []
    all_matching_classes_scores = []
    all_matching_classes_nl = []
    n_class_appearances = defaultdict(int)
    for frame_idx in range(len(init_obj_det_frames)):
        cur_frame_object_set_dets = [obj_set_detection[frame_idx] for obj_set_detection in obj_set_detections]
        frame_detections = copy.deepcopy(cur_frame_object_set_dets)

        frame_detections_concat = []
        running_class_id_offset = 0
        for det_idx in range(len(frame_detections)):
            cur_detections = frame_detections[det_idx]

            if cur_detections is None:
                continue

            cur_detections.class_id = cur_detections.class_id + running_class_id_offset
            frame_detections_concat.append(cur_detections)
            running_class_id_offset += len(vlm_object_names[det_idx])

        frame_detections_concat = Detections.merge(frame_detections_concat)
        all_frame_detections_concat.append(frame_detections_concat)
        for unique_class_id in np.unique(frame_detections_concat.class_id):
            class_idx = unique_class_id - 1
            class_name = all_nl_classes_to_compare[class_idx]["name"]
            class_scores = frame_detections_concat.confidence[
                frame_detections_concat.class_id == unique_class_id]
            if unique_class_id not in all_class_scores:
                all_class_scores[unique_class_id] = [class_scores]
            else:
                all_class_scores[unique_class_id].append(class_scores)
            n_class_appearances[unique_class_id] += 1

        matching_classes, matching_classes_scores, matching_classes_nl = score_nl_object_proposals(vlm_object_names,
                                                                                                   cur_frame_object_set_dets)
        all_matching_classes.append(matching_classes)
        all_matching_classes_scores.append(matching_classes_scores)
        all_matching_classes_nl.append(matching_classes_nl)
    matching_classes_coocurrences = compute_detection_cooccurrence(all_matching_classes)
    return all_class_scores, all_frame_detections_concat, all_nl_classes_to_compare, matching_classes_coocurrences, n_class_appearances


def filter_object_names_by_confidence(detection_model, init_obj_det_frames, vlm_object_names):
    scores_per_class = {}
    ov_prompts = [f"{obj['color']} {obj['name']}" for obj in vlm_object_names]
    ov_prompts = [name.replace(".", "") for name in ov_prompts]
    # ov_detections = self.detection_model.detect_objects(np.array(image_subset[[best_frame]]),
    #                                                    ov_prompts,bsz= 1)
    ov_detections = detection_model.detect_objects(init_obj_det_frames,
                                                   ov_prompts, bsz=8)
    for ov_det_frame in ov_detections:
        for cls_idx in range(len(ov_prompts)):
            if cls_idx not in scores_per_class:
                scores_per_class[cls_idx] = []
            cur_cls_scores = ov_det_frame[ov_det_frame.class_id == cls_idx].confidence

            if len(cur_cls_scores) == 0:
                continue
            best_score = cur_cls_scores.max()
            scores_per_class[cls_idx].append(best_score)
    for key, val in scores_per_class.items():
        scores_per_class[key] = np.sum(val) / len(init_obj_det_frames)
    top_5_score = np.sort(np.array(list(scores_per_class.values())))[::-1][:5][-1]
    valid_classes = [k for k, v in scores_per_class.items() if v > min(top_5_score - 0.07, 0.35)]

    return ov_detections, valid_classes


def detect_high_overlap_objects(detections, reference_detections):
    high_overlap_class_id_counter = {}

    for idx, reference_detection in enumerate(reference_detections):

        best_boxes_per_class = get_best_detection_per_class(reference_detection)
        reference_boxes = np.array(best_boxes_per_class.xyxy)

        cur_obj_detections = detections[idx]
        # iou = np.array(box_iou(torch.tensor(robot_det.xyxy), torch.tensor(cur_obj_detections.xyxy)))
        iou = np.array(box_iou(torch.tensor(reference_boxes), torch.tensor(cur_obj_detections.xyxy)))
        high_overlap_boxes = np.any(iou > 0.65, axis=0)
        high_overlap_class_ids = cur_obj_detections.class_id[high_overlap_boxes]
        for class_id in high_overlap_class_ids:
            if class_id in high_overlap_class_id_counter:
                high_overlap_class_id_counter[class_id] += 1
            else:
                high_overlap_class_id_counter[class_id] = 1

    return high_overlap_class_id_counter
