import copy
import itertools
import logging
import os
from pathlib import Path
import pickle
import traceback
from collections import defaultdict
from functools import wraps

import cv2
import numpy as np
import pandas as pd
import shapely
import torch
from PIL.Image import Image
from matplotlib import pyplot as plt
from quadrilateral_fitter import QuadrilateralFitter
from scipy.ndimage import label
from shapely import Polygon
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from supervision import Detections
import torch.nn.functional as F
from torchvision.ops import box_iou

from nils.specialist_models.sam2.utils.amg import remove_small_regions


def retry_on_exception(exception, retries=3, logger=None):
    """
    Decorator to retry a function call on specified exception.

    Parameters:
    exception (Exception): The exception to catch and retry.
    retries (int): Number of times to retry the function call.
    logger (logging.Logger): Optional logger for logging retry attempts.
    """

    def decorator_retry(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = retries
            while attempts > 0:
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    attempts -= 1
                    if logger:
                        logger.info(f"Retrying due to {exception.__name__}: {e}")
                    else:
                        print(f"Retrying due to {exception.__name__}: {e}")
                    traceback.print_exc()
                    if attempts == 0:
                        # just return nothing
                        return None

        return wrapper

    return decorator_retry

def check_masks_contact(masks, best_mask_idx):
    masks_dilated = np.stack(
        [cv2.dilate(mask.astype(np.uint8), np.ones((3, 3)), iterations=2) for mask in masks])

    overlap = np.any(np.logical_and(masks_dilated[best_mask_idx], masks_dilated), axis=(1, 2))

    return overlap


def map_annotation_indices(annotation_indices, task_annotations, cur_batch_frame_indices, sampling_interval):
    annotation_indices_idx = (annotation_indices >= cur_batch_frame_indices[0]) & (
            annotation_indices <= cur_batch_frame_indices[-1])
    annotation_indices = annotation_indices[
        (annotation_indices >= cur_batch_frame_indices[0]) & (annotation_indices <= cur_batch_frame_indices[-1])]

    task_annotations = np.array(task_annotations)[annotation_indices_idx]

    annotation_indices = np.ceil(annotation_indices / sampling_interval).astype(
        int) * sampling_interval

    sorted_annotation_indices = np.argsort(annotation_indices)
    annotation_indices = annotation_indices[sorted_annotation_indices]
    task_annotations = task_annotations[sorted_annotation_indices]

    return annotation_indices, task_annotations

def remove_and_keep_highest(array, scores, keystate_objects, threshold):

    # Convert arrays to numpy arrays
    array = np.array(array)
    scores = np.array(scores)
    keystate_objects = np.array(keystate_objects)

    # Sort the array based on values
    sorted_indices = np.argsort(array)
    sorted_array = array[sorted_indices]
    sorted_scores = scores[sorted_indices]
    sorted_keystate_objects = keystate_objects[sorted_indices]

    # Initialize variables to keep track of the current group
    current_highest_index = sorted_indices[0]
    current_highest_score = sorted_scores[0]
    new_array = []
    new_scores = []
    new_keystate_objects = []
    new_indices = []  # New list to store the indices of the highest scoring elements

    # Iterate through the sorted array
    for i in range(1, len(sorted_array)):
        # If the current element is within the threshold of the current group
        if sorted_array[i] - sorted_array[current_highest_index] <= threshold:
            # Update the highest scoring integer and its index
            if sorted_scores[i] > current_highest_score:
                current_highest_score = sorted_scores[i]
                current_highest_index = sorted_indices[i]
        else:
            # Add the highest scoring integer of the current group and its index to the result
            new_array.append(sorted_array[current_highest_index])
            new_scores.append(current_highest_score)
            new_keystate_objects.append(sorted_keystate_objects[current_highest_index])
            new_indices.append(current_highest_index)
            # Update the highest score and its index for the new group
            current_highest_score = sorted_scores[i]
            current_highest_index = sorted_indices[i]

    # Add the highest scoring integer of the last group and its index to the result
    new_array.append(sorted_array[current_highest_index])
    new_scores.append(current_highest_score)
    new_keystate_objects.append(sorted_keystate_objects[current_highest_index])
    new_indices.append(current_highest_index)

    return new_indices


def fill_false_outliers_np(array, n):
    s = pd.Series(array)

    s1 = s.ne(False).cumsum()
    grp = s.eq(False).groupby(s1).transform('sum').lt(n)

    # Fill the groups of False values shorter than n with True
    s = s.mask(s.eq(False) & grp, True)

    return s.values


def filter_consecutive_values_with_outlier_fill(array, min_length=8, outlier_fill_tolerance=3):
    s = pd.Series(array)

    # Define the minimum length of consecutive values
    n = min_length

    # Define the maximum length of outliers
    k = outlier_fill_tolerance

    # Identify groups of consecutive identical values
    s_groups = s.ne(s.shift()).cumsum()

    # Compute group sizes
    group_sizes = s_groups.map(s_groups.value_counts())

    # Identify outliers that are surrounded by at least 2 identical values
    outliers = (s.shift(-k) == s.shift(k)) & (group_sizes <= k)

    # Replace outliers with the surrounding value
    s[outliers] = s.shift(-k)[outliers]

    # Identify groups of consecutive identical values again after replacing outliers
    s_groups = s.ne(s.shift()).cumsum()

    # Compute group sizes again after replacing outliers
    group_sizes = s_groups.map(s_groups.value_counts())

    # Replace groups shorter than n with NaN, excluding leading and trailing groups
    s = s.where((group_sizes >= n) | (s_groups == s_groups.iat[0]) | (s_groups == s_groups.iat[-1]), np.nan)

    # Forward fill to replace NaNs with the last valid consecutive occurrences
    s = s.ffill()

    # Backward fill to replace leading NaNs with the first valid consecutive occurrences
    s = s.bfill()

    return s.values


def add_paraphrases_to_predictions(out_dir, paraphrases):
    folders = os.listdir(out_dir)

    for folder in folders:
        with open(os.path.join(out_dir, folder, "robot_data.pickle"), "rb") as f:
            robot_data = pickle.load(f)

        if "None" in robot_data["language_description"]:
            print(folder)
            continue
        if isinstance(robot_data["language_description"], list) or robot_data[
            "language_description"] not in paraphrases:
            print(folder)
            continue
        robot_data["language_description"] = paraphrases[robot_data["language_description"]]

        with open(os.path.join(out_dir, folder, "robot_data.pickle"), "wb") as f:
            pickle.dump(robot_data, f)


def get_keystate_prediction_accuracy(predictions, gt_keystates, gt_lang=None, pred_lang=None, tolerances=[0]):
    metrics = {}
    wrong_pred_lang = []
    for eps in tolerances:
        TP = 0
        FP = 0
        FN = 0

        TP_LANG = 0
        FP_LANG = 0
        for i in range(len(predictions)):
            dists = np.absolute(gt_keystates - predictions[i])
            dists_fwd = gt_keystates - predictions[i]
            dists_fwd = dists_fwd[dists_fwd <= 0]
            dists_fwd = np.abs(dists_fwd)
            dists_bwd = gt_keystates - predictions[i]
            dists_bwd = dists_bwd[dists_bwd >= 0]
            dists_bwd = np.abs(dists_bwd)
            if len(dists_bwd) == 0:
                dists_bwd = 100000000
            if len(dists_fwd) == 0:
                dists_fwd = 100000000
            if dists.min() > 32:
                FP += 1
                continue
            best_idx = np.argmin(dists)
            if dists[best_idx] <= eps:
                # if np.min(dists_bwd) <= 4 or np.min(dists_fwd) < eps:
                TP += 1
                if gt_lang is not None and pred_lang is not None:
                    if gt_lang[best_idx] in pred_lang[i]:
                        TP_LANG += 1
                    else:
                        wrong_pred_lang.append((gt_lang[best_idx], pred_lang[i]))
                        FP_LANG += 1
            else:
                FP += 1
                # FP_LANG += 1
            FN = max(len(gt_keystates) - TP, 0)

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        metrics[eps] = {"accuracy": TP / len(predictions) if len(predictions) > 0 else 0, "precision": precision,
                        "recall": recall, "f1": f1,
                        "TP": TP, "FP": FP, "FN": FN, "TP_LANG": TP_LANG, "FP_LANG": FP_LANG,
                        "wrong_pred_lang": wrong_pred_lang}
    return metrics



def save_action_predictions_to_dir(out_path, save_idx, orig_frames_cam_1, orig_frames_cam_2, orig_signal_dict, lang_ann,
                                   predicted_keystates, task_nl_map):
    cur_idx_string = str(save_idx)
    start_idx = int(predicted_keystates[0])
    end_idx = int(predicted_keystates[1])
    subset_images_cam_1 = np.array(orig_frames_cam_1)[start_idx:end_idx]
    subset_images_cam_2 = np.array(orig_frames_cam_2)[start_idx:end_idx]
    # lang_ann = llm_outputs[cur_pred_idx]
    singal_dict_subset = {key: orig_signal_dict[key][start_idx:end_idx] for key in orig_signal_dict.keys()}

    os.makedirs(os.path.join(out_path, str(cur_idx_string)), exist_ok=True)
    os.makedirs(os.path.join(out_path, str(cur_idx_string), "cam_1"), exist_ok=True)
    os.makedirs(os.path.join(out_path, str(cur_idx_string), "cam_2"), exist_ok=True)

    try:
        singal_dict_subset["language_description"] = np.array(task_nl_map[lang_ann])
    except:
        singal_dict_subset["language_description"] = np.array(lang_ann)

    with open(os.path.join(out_path, cur_idx_string, "robot_data.pickle"), "wb") as f:
        pickle.dump(singal_dict_subset, f)
    with open(os.path.join(out_path, cur_idx_string, "lang_ann.txt"), "w") as f:
        f.writelines(lang_ann)

    for idx, img in enumerate(subset_images_cam_1):
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(out_path, cur_idx_string, "cam_1", f"{idx}.jpeg"), im_rgb)
    for idx, img in enumerate(subset_images_cam_2):
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(out_path, cur_idx_string, "cam_2", f"{idx}.jpeg"), im_rgb)


def filter_consecutive_trues_np(array, n):
    # Create an array of the same length initialized with False
    filtered_array = np.full(array.shape, False)

    # Find the indices where array is True
    true_indices = np.where(array)[0]

    # Label groups of consecutive True values
    labels, num_groups = label(array)

    # For each group, if its size is >= n, set those indices in filtered_array to True
    for i in range(1, num_groups + 1):
        group_indices = np.where(labels == i)[0]
        if len(group_indices) >= n:
            filtered_array[group_indices] = True

    return filtered_array


def list_of_dict_of_dict_to_dict_of_dict_of_list(dict_list):
    all_keys = np.unique(np.concatenate(
        [np.array(list(frame.keys())) for frame in dict_list if len(frame.keys()) > 0]))
    all_objects = np.array([obj for obj in all_keys])

    # Initialize the objects dictionary with empty lists for each key in the frames
    objects = {obj: {key: [] for key in dict_list[-2][obj].keys()} for obj in all_objects}

    for frame in dict_list:
        for obj in all_objects:
            if obj not in frame.keys():
                for key in objects[obj].keys():
                    objects[obj][key].append(None)
            else:
                for key in frame[obj].keys():
                    objects[obj][key].append(np.array(frame[obj][key]))

    # convert to numpy arrays:
    for obj in objects.keys():
        for key in objects[obj].keys():
            objects[obj][key] = np.array(objects[obj][key])

    return objects


def get_gripper_movement_nl_3d(robot_obs):
    movement = robot_obs[:, :3]

    movement = np.diff(movement, axis=0)

    movements = []

    movements.append("No movement")
    for mov in movement:
        s = 1


def compute_detection_cooccurrence(all_matching_classes):
    cooccurrence_counts = defaultdict(lambda: defaultdict(int))

    # Iterate over each frame's groups
    for frame_groups in all_matching_classes:
        # For each group in the frame...
        for group in frame_groups:
            # For each pair of descriptions in the group...
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # Increment the co-occurrence count for the pair
                    cooccurrence_counts[group[i]][group[j]] += 1
                    cooccurrence_counts[group[j]][group[i]] += 1

    return cooccurrence_counts


def compute_final_groups(cooccurrence_counts, n_class_appearances, threshold=0.2):
    # Initialize a dictionary to store the total co-occurrence counts for each description
    total_counts = defaultdict(int)

    # Compute the total co-occurrence counts for each description
    for description, related_descriptions in cooccurrence_counts.items():
        total_counts[description] = sum(related_descriptions.values())
        total_counts[description] = n_class_appearances[description]
        # total_counts[description] = len(cooccurrence_counts)

    # Initialize a dictionary to store the average co-occurrence counts for each pair of descriptions
    average_counts = defaultdict(lambda: defaultdict(float))

    # Compute the average co-occurrence counts for each pair of descriptions
    for description1, related_descriptions in cooccurrence_counts.items():
        for description2, count in related_descriptions.items():
            # print(total_counts[description1])
            average_counts[description1][description2] = count / total_counts[description1]

    final_groups = []

    # While there are still descriptions to be grouped...
    while average_counts:
        # Find the description with the highest total average co-occurrence count
        max_description = max(average_counts.keys(), key=lambda d: sum(average_counts[d].values()))
        max_score = list(average_counts[max_description].values())

        # Initialize a group with the description and its related descriptions
        group = [max_description] + [desc for desc, avg_count in average_counts[max_description].items() if
                                     avg_count > 0.6]

        # Add the group to the final groups
        final_groups.append(group)

        # Remove the descriptions in the group from the average counts
        # for description in group:
        #     del average_counts[description]
        for description in group:
            del average_counts[description]
            for remaining_description in average_counts:
                if description in average_counts[remaining_description]:
                    del average_counts[remaining_description][description]

    return final_groups


def get_object_movement_nl(object_positions):
    movement = object_positions[1] - object_positions[0]

    movement_str = ""
    thresh = 0.03
    if abs(movement[0]) > thresh:
        if movement[0] > thresh:
            movement_str += f" {abs(movement[0])} m to the right "
        elif movement[0] < -thresh:
            movement_str += f" {abs(movement[0])} m to the left"
    if abs(movement[1]) > thresh:
        if movement[1] > thresh:
            movement_str += f" {abs(movement[1])} m up"
        elif movement[1] < -thresh:
            movement_str += f" {abs(movement[1])} m down"
    if abs(movement[2]) > thresh:
        if movement[2] > thresh:
            movement_str += f" {abs(movement[2])} m forward"
        elif movement[2] < -thresh:
            movement_str += f" {abs(movement[2])} m backward"

    return movement_str


def check_for_undetected_objects(region_proposals, objectness, class_embeddings, object_detections, images):
    undetected_objects_mem = None
    undetected_objects_indices = []  # stores tuples of (frame_idx,object_idx)
    for idx, region_proposal in enumerate(region_proposals):

        boxes = object_detections[idx]
        cur_region_proposals = region_proposals[idx]
        cur_objectness = objectness[idx]
        cur_img = images[idx]
        cur_class_embeds = class_embeddings[idx]

        region_proposal_indices = np.arange(len(cur_region_proposals))
        region_proposal_areas = (cur_region_proposals[:, 2] - cur_region_proposals[:, 0]) * (
                cur_region_proposals[:, 3] - cur_region_proposals[:, 1])
        area_filter = region_proposal_areas > 0.5 * images.shape[1] * images.shape[2]

        filter_score = max(np.sort(cur_objectness)[::-1][5], 0.4)

        objectness_filter = cur_objectness <= filter_score

        combined_filter = area_filter | objectness_filter

        region_proposals_filtered = cur_region_proposals[~combined_filter]
        region_proposal_indices_filtered = region_proposal_indices[~combined_filter]

        iou = box_iou(torch.tensor(boxes), torch.tensor(region_proposals_filtered))

        max_ref_iou, max_ref_iou_idx = iou.max(axis=0)

        undetected_regions = region_proposals_filtered[np.array(max_ref_iou < 0.2)]
        undetected_regions_idx = region_proposal_indices_filtered[np.array(max_ref_iou < 0.2)]

        undetected_regions_embeds = torch.tensor(cur_class_embeds[undetected_regions_idx])

        if undetected_regions_embeds.ndim == 1:
            undetected_regions_embeds = undetected_regions_embeds[None, ...]
            undetected_regions = undetected_regions[None, ...]
            undetected_regions_idx = [undetected_regions_idx]

        if len(undetected_regions) == 0:
            continue

        # cosine sim with tracked objects

        if undetected_objects_mem is None:
            print(f"Found new object in frame {idx}")
            undetected_objects_mem = torch.tensor(np.array(undetected_regions_embeds))
            for undetected_region_idx in undetected_regions_idx:
                undetected_objects_indices.append((idx, undetected_region_idx))
        else:
            sim = torch.nn.functional.cosine_similarity(torch.tensor(np.array(undetected_objects_mem))[None, ...],
                                                        undetected_regions_embeds[:, None], dim=-1)

            for sim_idx in range(len(sim)):

                max_sim = sim[sim_idx].max()
                max_sim_idx = sim[sim_idx].argmax()
                if max_sim < 0.85:

                    print(f"Found new object in frame {idx}, sim: {max_sim}")
                    # plot_boxes_np(cur_img, [undetected_regions[sim_idx]])

                    undetected_objects_mem = torch.cat([undetected_objects_mem, undetected_regions_embeds[[sim_idx]]])
                    undetected_objects_indices.append((idx, undetected_regions_idx[sim_idx]))
                elif max_sim > 0.85:

                    undetected_objects_mem[max_sim_idx] += undetected_regions_embeds[sim_idx]
                    undetected_objects_mem[max_sim_idx] /= 2

        s = 1
        continue
    return undetected_objects_mem, undetected_objects_indices


def get_best_detection_per_class(detection):
    max_det_indices = []
    for cls_idx in np.unique(detection.class_id):
        max_det_idx = np.argmax(detection.confidence[detection.class_id == cls_idx])

        max_det_global_idx = np.where(detection.class_id == cls_idx)[0][max_det_idx]

        max_det_indices.append(max_det_global_idx)

    detections_filtered = detection[max_det_indices]

    return detections_filtered


def incorporate_clip_scores_into_detections(detections, clip_scores):
    detections = copy.deepcopy(detections)

    for det_idx in range(len(detections)):
        cur_class_id = detections.class_id[det_idx]
        clip_score = clip_scores[det_idx, cur_class_id]
        detections.confidence[det_idx] += clip_score
        detections.confidence[det_idx] /= 2

    return detections


def incorporate_objectness_into_class_score(detections, region_proposals, objectness, cur_clip_class_scores=None):
    detections = copy.deepcopy(detections)
    region_proposals = torch.tensor(region_proposals)
    objectness = torch.tensor(objectness)
    boxes = torch.tensor(detections.xyxy)

    ious = box_iou(boxes, region_proposals)

    max_ious, max_ious_idx = ious.max(axis=1)

    for det_idx in range(len(detections)):
        if max_ious[det_idx] < 0.7:
            continue
        matched_objectness = objectness[max_ious_idx[det_idx]]
        matched_objectness = matched_objectness.item()
        detections.confidence[det_idx] += matched_objectness * 0.2

        return detections


def score_nl_object_proposals(nl_proposals, frame_detections):
    frame_detections = copy.deepcopy(frame_detections)
    all_nl_classes_to_compare = list(itertools.chain.from_iterable(nl_proposals))

    frame_detections_concat = []
    running_class_id_offset = 0
    for det_idx in range(len(frame_detections)):
        cur_detections = frame_detections[det_idx]
        if cur_detections is None:
            continue
        cur_detections.class_id = cur_detections.class_id + running_class_id_offset
        # print(cur_detections.class_id)
        frame_detections_concat.append(cur_detections)
        running_class_id_offset += len(nl_proposals[det_idx])

    frame_detections_concat = Detections.merge(frame_detections_concat)
    s = 1

    boxes_to_compare = torch.tensor(frame_detections_concat.xyxy)
    classes_to_compare = frame_detections_concat.class_id
    conf_to_compare = frame_detections_concat.confidence
    nl_classes_to_compare = all_nl_classes_to_compare
    nl_classes_to_compare = np.array(nl_classes_to_compare)[frame_detections_concat.class_id - 1]
    ious = box_iou(boxes_to_compare, boxes_to_compare)
    similar_classes_triu = torch.triu(ious)
    similar_classes = ious > 0.5
    ind_to_skip = []
    matching_classes = []
    matching_classes_scores = []
    matching_classes_nl = []
    for cls in range(len(ious)):
        if cls in ind_to_skip:
            continue
        cur_sim_classes = similar_classes[cls]
        matching_classes.append(classes_to_compare[cur_sim_classes])
        matching_classes_nl.append(np.array(nl_classes_to_compare)[cur_sim_classes])
        matching_cls_scores = conf_to_compare[cur_sim_classes]
        matching_classes_scores.append(matching_cls_scores)
        ind_to_skip += [i for i in range(len(cur_sim_classes)) if cur_sim_classes[i]]

    return matching_classes, matching_classes_scores, matching_classes_nl


def move_line_to_intersect_point(line, point):
    """
    Given a line segment and a point, move the line segment so that it intersects the point.
    """

    x1, y1, x2, y2 = line
    px, py = point

    # Calculate the direction of the original line
    dx, dy = x2 - x1, y2 - y1

    # Determine the perpendicular direction
    perp_dx, perp_dy = -dy, dx

    # Calculate the midpoint of the original line segment
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2

    # Calculate the vector from the midpoint to the point
    vec_to_point = np.array([px - mx, py - my])

    # Project vec_to_point onto the perpendicular direction to find the distance to move
    proj_length = np.dot(vec_to_point, np.array([perp_dx, perp_dy])) / np.sqrt(perp_dx ** 2 + perp_dy ** 2)

    # Normalize the perpendicular direction
    norm = np.sqrt(perp_dx ** 2 + perp_dy ** 2)
    perp_dx, perp_dy = (perp_dx / norm), (perp_dy / norm)

    # Move the original line segment along the perpendicular direction
    new_x1, new_y1 = x1 + perp_dx * proj_length, y1 + perp_dy * proj_length
    new_x2, new_y2 = x2 + perp_dx * proj_length, y2 + perp_dy * proj_length

    return new_x1, new_y1, new_x2, new_y2

def iou(polygon1: Polygon, polygon2: Polygon):
    """
    Calculate the Intersection over Union (IoU) between two polygons.

    :param polygon1: Polygon. The first polygon.
    :param polygon2: Polygon. The second polygon.
    :param precomputed_polygon_1_area: float|None. The area of the first polygon. If None, it will be computed.
    :return: float. The IoU value.
    """

    precomputed_polygon_1_area = polygon1.area
    # Calculate the intersection and union areas
    intersection = polygon1.intersection(polygon2).area
    union = precomputed_polygon_1_area + polygon2.area - intersection
    # Return the IoU value
    return (intersection / union) if union != 0. else 0.

def get_homography_transform(combined_masks, plot=False):
    contours, _ = cv2.findContours(combined_masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = max(contours, key=cv2.contourArea).squeeze(1)

    fitter = QuadrilateralFitter(polygon=contours)
    fitted_quadrilateral = np.array(fitter.fit(simplify_polygons_larger_than=10), dtype=np.float32)

    intial_guess_points = shapely.get_coordinates(fitter._initial_guess)
    intial_guess_points = np.unique(intial_guess_points, axis=0)
    intial_guess_points = order_points_clockwise(intial_guess_points)
    # remove duplicate points from the np array:

    iou_inital = iou(polygon1=fitter.convex_hull_polygon, polygon2=Polygon(fitter._initial_guess))
    iou_fitted = iou(polygon1=fitter.convex_hull_polygon, polygon2=Polygon(fitter.fitted_quadrilateral))

    if plot:
        fitter.plot()

    if iou_inital > iou_fitted:
        hull = intial_guess_points
    else:
        hull = fitted_quadrilateral
    clamped_hull = []

    for point_idx in range(len(hull)):
        x = np.clip(hull[point_idx][0], 0, combined_masks.shape[1] - 1)
        y = np.clip(hull[point_idx][1], 0, combined_masks.shape[0] - 1)

        if x == 0:
            y = max(0, y - 30)
        if y == 0:
            x = min(x + 30, combined_masks.shape[1] - 1)

        clamped_hull.append((x, y))
    clamped_hull = clamped_hull[::-1]
    corner_points = copy.deepcopy(clamped_hull)
    corner_points.sort(key=lambda p: p[0])
    # Determine top-left and bottom-left points
    left_points = corner_points[:2]
    top_left = min(left_points, key=lambda p: p[1])
    bottom_left = max(left_points, key=lambda p: p[1])
    # Determine top-right and bottom-right points
    right_points = corner_points[2:]
    top_right = min(right_points, key=lambda p: p[1])
    bottom_right = max(right_points, key=lambda p: p[1])
    corner_points = [top_left, bottom_left, bottom_right, top_right]

    # plt.imshow(cur_img)
    #
    # # Plot the corner points
    # for point in corner_points:
    #     plt.scatter(*point, color='red')  # Change color as needed
    #
    # plt.show()
    homography_transform = cv2.findHomography(np.array(corner_points), np.array(
        [(0, 0), (0, combined_masks.shape[0]), (combined_masks.shape[1], combined_masks.shape[0]),
         (combined_masks.shape[1], 0)]))[0]
    return homography_transform


def get_moved_object(scene_graphs_subset_keystate, moved_objects, keystate_idx):
    if moved_objects is not None:
        moved_object = moved_objects[keystate_idx]
    else:
        obj_movements_summed = {}
        for sg_local in scene_graphs_subset_keystate:
            if sg_local.object_movements is None:
                continue
            for obj in sg_local.object_movements.keys():
                if obj not in obj_movements_summed.keys():
                    obj_movements_summed[obj] = 0
                obj_movements_summed[obj] += np.linalg.norm(sg_local.object_movements[obj])
        max_movement = np.argmax(list(obj_movements_summed.values()))
        moved_object = list(obj_movements_summed.keys())[max_movement]
    return moved_object


def split_batch(batch):
    """Given a batch with gt keystates, split the batch into smaller batches of 40 keystates"""

    gt_keystates = batch["keystates"]
    split_batches = []
    if len(gt_keystates) > 50:
        # split the batch into smaller batches
        n_keystates_per_batch = 40
        split_indices = np.arange(0, len(gt_keystates), n_keystates_per_batch)
        logging.info(f"Splitting batch into {len(split_indices)} smaller batches")
        for i in range(len(split_indices)):

            start_idx = gt_keystates[split_indices[i]]
            start_idx_raw = split_indices[i]

            if i == len(split_indices) - 1:
                end_idx = gt_keystates[-1]
                end_idx_raw = len(gt_keystates)
            else:
                end_idx = gt_keystates[split_indices[i + 1]]
                end_idx_raw = split_indices[i + 1]

            split_batch = {key: batch[key][start_idx:end_idx] for key in batch.keys() if
                           key != "task_annotation" and key != "annotations" and "path" not in key and "keystate" not in key}  #

            split_batch["keystates"] = gt_keystates[start_idx_raw:end_idx_raw]
            split_batch["keystates"] = split_batch["keystates"] - start_idx

            split_batch["path"] = batch["path"][start_idx_raw:end_idx_raw]
            split_batch["task_annotation"] = batch["task_annotation"][start_idx_raw:end_idx_raw]

            split_batches.append(split_batch)
            # recompute keystates
        return split_batches
    else:
        return [batch]


def load_predefined_objects(cfg):
    predefined_objects = []
    if cfg.prior.use_predefined_object_list:
        with open(cfg.predefined_objects_path, "r") as f:
            predefined_objects = f.readlines()
            predefined_objects = [obj.strip() for obj in predefined_objects]
            object_names = [obj.split(":")[0] for obj in predefined_objects]
            object_colors = [obj.split(":")[1] for obj in predefined_objects]
            predefined_objects = [{"name": obj, "color": color} for obj, color in zip(object_names, object_colors)]
    return predefined_objects


def save_annotation(color_obj_dict, cur_pred_tasks, data_save_path, interacted_object, interacted_object_synonyms,
                    keystate_reasons, keystate_scores):
    os.makedirs(data_save_path, exist_ok=True)
    data_save_path = Path(data_save_path)
    with open(Path(data_save_path) / "prompt_reasons.pkl", "wb") as f:
        pickle.dump(keystate_reasons, f)
    with open(data_save_path / "reasons.pkl", "wb") as f:
        pickle.dump(keystate_reasons, f)
    with open(data_save_path / "colors.pkl", "wb") as f:
        pickle.dump(color_obj_dict, f)
    with open(data_save_path / "objects.pkl", "wb") as f:
        pickle.dump(interacted_object, f)
    with open(data_save_path / "synonyms.pkl", "wb") as f:
        pickle.dump(interacted_object_synonyms, f)
    predicted_tasks_text = "\n".join(cur_pred_tasks) + f"\nconfidence: {keystate_scores[0]}"
    logging.info(f"\033[92mSaving to {data_save_path}\033[0m")
    with open(data_save_path / "lang_NILS.txt", "w") as f:
        f.write(predicted_tasks_text)


def setup_logging():
    formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False


def get_intrinsics(images, cache_dir):
    
    from PIL import Image
    
    cache_dir = os.path.join(cache_dir, "intrinsics_cache.pkl")

    if os.path.exists(cache_dir):

        with open(cache_dir, "rb") as f:
            intrinsics = pickle.load(f)
    else:
        model = torch.hub.load('ShngJZ/WildCamera', "WildCamera", pretrained=True)
        model = model.to("cuda")

        intrinsics = []
        for image in images:
            intrinsic, _ = model.inference(Image.fromarray(np.array(image)), wtassumption=False)
            intrinsics.append(intrinsic)
        intrinsics = np.mean(intrinsics, axis=0)
        with open(cache_dir, "wb") as f:
            pickle.dump(intrinsics, f)

        del model
        torch.cuda.empty_cache()

    intrinsic_parameters = {
        'width': images[0].shape[1],
        'height': images[0].shape[0],
        'fx': intrinsics[0, 0],
        'fy': intrinsics[1, 1],
        'cx': intrinsics[0, 2],
        'cy': intrinsics[1, 2],

    }

    return intrinsic_parameters


def get_box_dense_agreement(detection_model, cur_class_detections, cur_class_scores, cur_dense_logits):
    box_scores = []
    mask_filters = []
    for box, detection_score in zip(cur_class_detections, cur_class_scores):
        x_min, y_min, x_max, y_max = box.astype(int)
        selected_dense_score = cur_dense_logits[y_min:y_max + 1, x_min:x_max + 1]
        selected_dense_score[selected_dense_score < 0.2] = 0
        # normalize by area:
        box_area = (x_max - x_min) * (y_max - y_min)

        selected_dense_score = selected_dense_score / box_area
        mask_filter = torch.zeros_like(cur_dense_logits)
        mask_filter[y_min:y_max + 1, x_min:x_max + 1] = 1
        mask_filters.append(mask_filter)
        mean_dense_score = torch.sum(selected_dense_score)

        if "owl" in detection_model.__class__.__name__.lower():
            score = (mean_dense_score + detection_score * 2) / 3
        elif "dino" in detection_model.__class__.__name__.lower():
            score = (mean_dense_score * 2 + detection_score) / 3
        else:
            raise NotImplementedError(f"No weights for detector {detection_model.__class__.__name__.lower()}")

        box_scores.append(score)
    scores_old = box_scores
    box_scores = torch.nan_to_num(torch.stack(box_scores), nan=0.0)
    best_box_idx = np.argmax(box_scores)
    return best_box_idx, box_scores, mask_filters


def order_points_clockwise(pts):
    as_np = isinstance(pts, np.ndarray)
    if not as_np:
        pts = np.array(pts, dtype=np.float32)
    # Calculate the center of the points
    center = np.mean(pts, axis=0)

    # Compute the angles from the center
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

    # Sort the points by the angles in ascending order
    sorted_pts = pts[np.argsort(angles)]

    if not as_np:
        sorted_pts = tuple(tuple(pt) for pt in sorted_pts)
    return sorted_pts


def vis_scene_graphs(scene_graphs, images, out_path, plot=True):
    """Write scene graphs below image"""
    for idx, scene_graph in enumerate(scene_graphs):
        image = images[idx]

        sg_str = scene_graph.__str__()

        # Write scene graph below image in white rectangle with matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.text(0, 1, sg_str, color='white', backgroundcolor='black', fontsize=12, verticalalignment='top')
        ax.axis('off')

        if plot:
            plt.show()


def get_mask_agreement(masks, threshold=0.6):
    counts = np.sum(masks, axis=0)
    score = counts / len(masks)
    agreement = score > threshold
    return agreement


def find_points_close_to_line(line, seg_mask, dist_threshold=2, plot=False):
    """Given a line in x1,y1,x2,y2 format, find points in the mask that are close to the line"""

    x1, y1, x2, y2 = line

    # sample points along the line:
    t = np.linspace(0, 1, 100)
    line_points = np.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], axis=-1)

    points = np.argwhere(seg_mask > 0)
    # permute points to x,y format
    points = points[:, ::-1]

    # Calculate the Euclidean distance from each point in the mask to each sampled point on the line
    distances = np.linalg.norm(points[:, None, :] - line_points[None, :, :], axis=-1)

    # Find the minimum distance for each point in the mask

    min_distances = distances.min(axis=-1)

    # Create a boolean mask where the minimum distances are less than the threshold
    close_mask = min_distances < dist_threshold

    # Use this boolean mask to index the points in the mask that are close to the line
    close_points = points[close_mask]

    # get the two points with the largest distance between them

    max_dist = 0
    max_dist_points = None
    for point1 in close_points:
        for point2 in close_points:
            dist = np.linalg.norm(point1 - point2)
            if dist > max_dist:
                max_dist = dist
                max_dist_points = (point1, point2)

    # plot points on mask as red circles with cv2:
    if plot and max_dist_points is not None:
        mask = seg_mask.copy().astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for point in max_dist_points:
            cv2.circle(mask, tuple(point), 4, (255, 0, 0), -1)

        plt.imshow(mask)
        plt.show()
    return close_points, max_dist_points


def get_lines(image, alg="houghes_p", plot=False):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.copy()

    if plot:
        plt.imshow(edges)
        plt.show()

    if alg == "houghes_p":
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=30, maxLineGap=200)

        if plot:
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                plt.imshow(image)
                plt.show()

        return lines
    elif alg == "houghes":
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)

        if plot:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            plt.imshow(image)
            plt.show()

        return lines
    elif alg == "lsd":
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(image_gray)[0].squeeze(1)

        print(lines.shape)
        line_lengths = np.sqrt((lines[:, 2] - lines[:, 0]) ** 2 + (lines[:, 3] - lines[:, 1]) ** 2)

        lines = lines[line_lengths > 10]

        if plot:
            drawn_img = lsd.drawSegments(image_gray, lines)
            plt.imshow(drawn_img)
            plt.show()

        return lines
    else:
        raise ValueError("Invalid algorithm. Algorithm must be one of 'houghes_p', 'houghes', 'lsd'")


def cartesian_to_polar(line):
    x1, y1, x2, y2 = line
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    r = abs(C) / np.sqrt(A ** 2 + B ** 2)
    theta = np.arctan2(B, A)
    return r, theta


def fit_line_to_points(lines):
    """Given a set of lines in x1,y1,x2,y2 format, fit a line to the points with RANSAC"""

    # Convert lines to points
    points = []
    for line in lines:
        x1, y1, x2, y2 = line
        points.append((x1, y1))
        points.append((x2, y2))
    points = np.array(points)

    # Fit a line to the points using RANSAC
    ransac = RANSACRegressor()

    ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1])

    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    line = np.array([0, intercept, 1000, slope * 1000 + intercept])

    return line


def cluster_lines_polar(polar_lines_normalized, eps=0.15, min_samples=2):
    """Cluster lines based on their polar coordinates"""

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(polar_lines_normalized)

    # Group lines by their cluster labels
    clustered_lines = {}
    cluster_indices = {}
    for i, label in enumerate(labels):
        if label not in clustered_lines:
            clustered_lines[label] = []
            cluster_indices[label] = []
        clustered_lines[label].append(polar_lines_normalized[i])
        cluster_indices[label].append(i)

    return clustered_lines, cluster_indices


def line_length(x1, y1, x2, y2):
    """
    Calculate the length of a line segment.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def line_orientation(x1, y1, x2, y2):
    """
    Calculate the orientation (angle) of the line with the horizontal axis.
    """
    return np.arctan2((y2 - y1), (x2 - x1))


def line_proximity(line1, line2):
    """
    Calculate the minimum distance between two line segments.
    """

    def point_line_distance(px, py, x1, y1, x2, y2):
        """
        Calculate the distance from a point to a line segment.
        """
        line_mag = line_length(x1, y1, x2, y2)
        if line_mag == 0:
            return line_length(px, py, x1, y1)

        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
        u = max(0, min(1, u))
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        return line_length(px, py, ix, iy)

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    return min(
        point_line_distance(x1, y1, x3, y3, x4, y4),
        point_line_distance(x2, y2, x3, y3, x4, y4),
        point_line_distance(x3, y3, x1, y1, x2, y2),
        point_line_distance(x4, y4, x1, y1, x2, y2),
    )


def compare_lines(line1, line2, max_length, max_distance):
    """
    Compare two lines based on length, orientation, and proximity and generate a score.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Length score
    length1 = line_length(x1, y1, x2, y2)
    length2 = line_length(x3, y3, x4, y4)
    length_diff = abs(length1 - length2)
    length_score = 1 - (length_diff / max_length)  # Normalize and invert so closer lengths have higher scores

    # Orientation score
    orientation1 = line_orientation(x1, y1, x2, y2)
    orientation2 = line_orientation(x3, y3, x4, y4)
    orientation_diff = abs(orientation1 - orientation2)
    if orientation_diff > np.pi:
        orientation_diff = 2 * np.pi - orientation_diff  # Adjust for the periodicity of angles
    orientation_score = 1 - (orientation_diff / np.pi)  # Normalize and invert so closer angles have higher scores

    # Proximity score
    proximity = line_proximity(line1, line2)

    if proximity > 20:
        return 0
    proximity_score = 1 - (proximity / max_distance)  # Normalize and invert so closer proximities have higher scores

    # Combined score with equal weights
    combined_score = (length_score * 0.1 + orientation_score * 0.6 + proximity_score * 0.3)

    return combined_score


def line_projection_overlap(line1, line2, tolerance):
    """
    Calculate the overlap of projections of two lines on the x and y axes with a given tolerance.
    """
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2

    def project_onto_axis(p1, p2):
        return min(p1, p2), max(p1, p2)

    x1_min, x1_max = project_onto_axis(x11, x12)
    y1_min, y1_max = project_onto_axis(y11, y12)
    x2_min, x2_max = project_onto_axis(x21, x22)
    y2_min, y2_max = project_onto_axis(y21, y22)

    def ranges_overlap(min1, max1, min2, max2, tol):
        return max(0, min(max1 + tol, max2 + tol) - max(min1 - tol, min2 - tol))

    x_overlap = ranges_overlap(x1_min, x1_max, x2_min, x2_max, tolerance)
    y_overlap = ranges_overlap(y1_min, y1_max, y2_min, y2_max, tolerance)

    return x_overlap, y_overlap


def compare_lines_with_overlap(line1, line2, tolerance):
    """
    Compare two lines based on their overlap with a given tolerance and output a score.
    """
    x_overlap, y_overlap = line_projection_overlap(line1, line2, tolerance)

    length1 = line_length(*line1)
    length2 = line_length(*line2)

    max_length = max(length1, length2)

    x_overlap_score = x_overlap / max_length
    y_overlap_score = y_overlap / max_length

    overlap_score = (x_overlap_score + y_overlap_score) / 2

    return overlap_score


def compute_line_hull_match(line, hulls):
    """
    Compute the match score between a line and convex hulls.
    """
    x1, y1, x2, y2 = line
    line_points = np.array([(x1, y1), (x2, y2)], dtype=np.float32)

    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    score = 0
    scores = []

    for hull in hulls:

        hull = hull.squeeze(1)

        # get hull line segments:
        for i in range(len(hull)):
            # print(hull[i])
            x1, y1 = hull[i]
            x2, y2 = hull[(i + 1) % len(hull)]
            hull_line = np.array([x1, y1, x2, y2], dtype=np.float32)
            # score += compare_lines(line, hull_line, max_length, max_distance)
            scores.append(compare_lines(line, hull_line, 1000, 40))

        distances = cv2.pointPolygonTest(hull, tuple(line_points[0]), True), cv2.pointPolygonTest(hull,
                                                                                                  tuple(line_points[1]),
                                                                                                  True)

        min_distance = min(abs(distances[0]), abs(distances[1]))
        score += max(0, (line_length - min_distance))  # Higher score for closer matches
        # scores.append(score)

    score = max(scores)

    return score


def rank_lines_by_match(lines, hulls):
    """
    Rank the lines based on their match with the convex hulls.
    """
    line_scores = []
    for line in lines:
        score = compute_line_hull_match(line, hulls)
        line_scores.append((line, score))

    ranked_lines = sorted(line_scores, key=lambda x: x[1], reverse=True)
    return ranked_lines


def plot_contours_and_lines(mask, contours, lines):
    """
    Plot the contours and lines on the segmentation mask.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')

    # Plot contours
    for contour in contours:
        contour = contour.squeeze(1)
        plt.plot(contour[:, 0], contour[:, 1], color='red', linewidth=2)

    # Plot lines
    for line in lines:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2)

    plt.title('Contours and Lines')
    plt.axis('off')
    plt.show()


def interleave_detections(class_mask_dict_transformed, det_mask_subset_frames, n_frames, has_masks=True):
    class_mask_dicts_transformed_new = {}

    if has_masks:
        img_shape = (n_frames, list(class_mask_dict_transformed.values())[0]["mask"].shape[1],
                     list(class_mask_dict_transformed.values())[0]["mask"].shape[2])

    for cls, cls_dict in class_mask_dict_transformed.items():
        new_cls_dict = {}
        boxes = np.zeros((n_frames, 4), dtype=int)
        scores = np.zeros((n_frames,))
        ind = np.array(range(0, n_frames))
        bool_indexer = np.isin(ind, det_mask_subset_frames)
        boxes[bool_indexer] = cls_dict["box"]
        scores[bool_indexer] = cls_dict["score"]
        if has_masks:
            masks = np.zeros(img_shape)
            masks[bool_indexer] = cls_dict["mask"]
            new_cls_dict["mask"] = masks
        new_cls_dict["box"] = boxes
        new_cls_dict["score"] = scores
        class_mask_dicts_transformed_new[cls] = new_cls_dict
    class_mask_dict_transformed = class_mask_dicts_transformed_new
    return class_mask_dict_transformed


def cut_box_from_matrix(box, matrix):
    box = box.astype(int)
    x_min, y_min, x_max, y_max = box
    return matrix[y_min:y_max + 1, x_min:x_max + 1]




def clean_outlier_labels(object_manager, cur_masks):
    # Update class mask dict

    # todo: when adding new mask, check if old mask is occluded currently -> new mask liekyl to be noise, compare sscores
    # see episode_0000900.npz, red block (mask4)

    cur_masks = cur_masks.cpu().numpy()
    object_manager.calc_class_to_obj_id()

    class_to_obj = object_manager.class_to_obj_id

    for class_id, obj in class_to_obj.items():
        if len(obj) > 1:
            # Check if masks are in contact
            mask_ids = np.array([object_manager.obj_to_tmp_id[obj_id] for obj_id in obj])
            mask_scores = np.array(
                [np.mean(np.array(o.class_scores)[np.where(np.array(o.category_ids) == class_id)]) for o in obj])
            mask_scores = mask_scores + np.array(
                [np.sum(np.array(o.class_scores)[np.where(np.array(o.category_ids) == class_id)]) for o in obj]) * 0.5

            overall_class_counts = np.array([np.array(o.class_scores).shape[0] for o in obj])
            best_class_count = np.array([len(np.where(np.array(o.category_ids) == class_id)[0]) for o in obj])
            consistency = best_class_count / overall_class_counts
            scaled_consistency = (consistency - 0.5) / 0.5


            #mask_scores = mask_scores*0.9 + scaled_consistency*0.1


            # todo: best score can be 0 if mask occluded
            best_score = np.argmax(mask_scores)

            n_class_detections = [np.array(o.class_scores)[np.where(np.array(o.category_ids) == class_id)].shape[0] for o in obj]

            masks = cur_masks[mask_ids - 1]
            if masks[best_score].sum() <= 10:
                continue
            overlap = check_masks_contact(masks, best_score)
            # Remove masks that are not in contact
            mask_idx_to_remove = np.where(~overlap)[0]
            for idx in mask_idx_to_remove:
                # Skip occluded masks
                if masks[idx].sum() <= 0:
                    continue
                #if mask has a lot of detections, do not remove
                if n_class_detections[idx] > 5:
                    continue
                obj[idx].del_class_id(class_id)

                #reduce score:
                # for o in obj:
                #     o.class_scores = [s-0.001 for s in o.class_scores]
                #     o.scores = [s-0.001 for s in o.scores]

        object_manager.calc_class_to_obj_id()

    #check for disconnected components
    class_to_obj = object_manager.class_to_obj_id


def get_return_masks_dict(prob, image_np, object_labels, obj_to_tmp_id, class_to_objs, need_resize=False, shape=None,
                          ):
    class_mask_dict = {}

    if need_resize:
        prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,
               0]



    # Probability mask -> index mask
    mask = torch.argmax(prob, dim=0)



    for class_id, objects in class_to_objs.items():
        # merge masks
        cur_object_mask = torch.zeros_like(mask)
        deva_mask_ids = []
        logits = []
        for obj in objects:
            tmp_obj_id = obj_to_tmp_id[obj]
            cur_object_mask[mask == tmp_obj_id] = 1

            # cur_logits = (prob_softmax[tmp_obj_id][mask == tmp_obj_id])
            #
            # normed_logits = torch.mean(cur_logits)




            deva_mask_ids.append(obj.id)


        cur_object_mask = cur_object_mask.cpu().numpy()

        cur_object_mask, _ = remove_small_regions(cur_object_mask.astype(bool), 50, mode="holes")
        cur_object_mask, _ = remove_small_regions(cur_object_mask.astype(bool), 500, mode="holes")



        cur_object_mask = cur_object_mask.astype(np.uint8)

        score = np.sum(
            [np.mean(np.array(obj.class_scores)[np.where(np.array(obj.category_ids) == class_id)]) for obj in objects])

        overall_class_counts = np.sum([np.array(o.class_scores).shape[0] for o in objects])
        best_class_count = np.sum([len(np.where(np.array(o.category_ids) == class_id)[0]) for o in objects])
        consistency = best_class_count / overall_class_counts
        scaled_consistency = (consistency - 0.5) / 0.5

        score = score*0.9 + scaled_consistency*0.1




        class_mask_dict[object_labels[class_id]] = {"mask": cur_object_mask, "score": score,"deva_mask_ids":deva_mask_ids}


        #remap keys to lang keys:

    return class_mask_dict