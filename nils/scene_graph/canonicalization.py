import logging
import os

import cv2
import numpy as np
from scipy.stats import zscore

from nils.specialist_models.sam2.utils.amg import remove_small_regions
from nils.utils.utils import get_lines, cartesian_to_polar, cluster_lines_polar, fit_line_to_points




def get_best_surface_box(detection_model,images,surface_prompts,vlm_key = None,area_weight= 0.5):
    detector_detections = detection_model.detect_objects(images, surface_prompts)
    # get best detection for each image

    best_boxes = []
    class_scores = {key: [] for key in surface_prompts}
    best_class_boxes = {key: [] for key in surface_prompts}
    best_class_areas = {key: [] for key in surface_prompts}
    for detections in detector_detections:

        unique_classes = np.unique(detections.class_id)
        undetected_classes = [class_id for class_id in range(len(surface_prompts)) if
                              class_id not in unique_classes]
        for class_id in unique_classes:
            cur_class_boxes = detections.xyxy[detections.class_id == class_id]
            cur_class_scores = detections.confidence[detections.class_id == class_id]
            best_box_idx = np.argmax(cur_class_scores)
            best_box = cur_class_boxes[best_box_idx]
            best_box_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])

            # if best_box_area > 0.95 * (images[0].shape[0] * images[0].shape[1]):
            #     best_class_boxes[surface_prompts[class_id]].append(np.array([0, 0, 0, 0]))
            #     continue

            best_class_boxes[surface_prompts[class_id]].append(cur_class_boxes[best_box_idx])
            class_scores[surface_prompts[class_id]].append(cur_class_scores[best_box_idx])

            best_class_areas[surface_prompts[class_id]].append(best_box_area)

        for undetected_class in undetected_classes:
            best_class_boxes[surface_prompts[undetected_class]].append(np.array([0, 0, 0, 0]))
    avg_class_scores = {key: np.mean(np.array(class_scores[key])) for key in class_scores.keys()}
    avg_class_areas = {key: np.mean(np.array(best_class_areas[key])) for key in best_class_areas.keys()}
    count_score = {key: len(class_scores[key]) / len(images) for key in class_scores.keys()}

    max_possible_area = images[0].shape[0] * images[0].shape[1]

    normalized_areas = {key: np.array(avg_class_areas[key]) / max_possible_area for key in best_class_areas.keys()}

    combined_scores = {key: avg_class_scores[key] * 0.5 + normalized_areas[key] * area_weight + count_score[key] * 0.1 for
                       key in avg_class_scores.keys()}

    if vlm_key is not None:

        if normalized_areas[vlm_key] >= 0.4:
            if count_score[vlm_key] >= 0.5:
                combined_scores[vlm_key] += 0.15
            elif count_score[vlm_key] >= 0.25:
                combined_scores[vlm_key] += 0.02

    for key in combined_scores.keys():
        if np.isnan(combined_scores[key]):
            combined_scores[key] = 0

    best_class = max(combined_scores, key=combined_scores.get)

    best_boxes = np.array(best_class_boxes[best_class])

    return best_boxes,best_class



def get_surface_object_masks(detection_model,segmentation_model, m3d, images, surface_prompts, intrinsics, seg_threshold=0.6,
                             area_weight = 0.5):
    intrinsics = np.array([intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]])

    best_boxes, best_class = get_best_surface_box(detection_model, images, surface_prompts)


    valid_box_indices = best_boxes.sum(axis=1) > 0
    mask_images = images[valid_box_indices]
    valid_boxes = best_boxes[valid_box_indices]

    # get surface normals:
    depth, normals, conf = m3d.predict(np.array(mask_images), intrinsics)

    up_normals = normals[:, :, :, 1] > 0.6

    indices = list(range(len(images)))
    indices = np.array(indices)[valid_box_indices]

    # get masks

    masks, ious = segmentation_model.segment(mask_images, valid_boxes, threshold=seg_threshold)

    masks = masks.squeeze(1)

    # filter outlier masks by area:
    mask_areas = np.array([mask.sum() for mask in masks])

    mask_areas = mask_areas / (images[0].shape[0] * images[0].shape[1])
    zscores = zscore(mask_areas)
    noisy_masks = np.abs(zscores) > 1.5

    masks = np.array(masks)[~noisy_masks]

    indices = indices[~noisy_masks]

    valid_box_indices = np.array([True if idx in indices else False for idx in range(len(images))])

    # copmbine to singel mask with or reduce
    combined_mask = np.logical_or.reduce(masks)

    # combine normal and surface mask:
    # final_mask = np.logical_and(combined_mask,valid_normals_mask)

    return masks, ious, np.array(up_normals), valid_box_indices, best_class


def get_lines_from_surface_masks(surface_masks, max_size_to_remove):
    all_lines = []
    for mask in surface_masks:

        mask = remove_small_regions(mask, area_thresh=max_size_to_remove, mode="islands")[0]
        mask = remove_small_regions(mask, area_thresh=max_size_to_remove, mode="holes")[0]

        surface_mask_rgb = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)

        lines = get_lines(surface_mask_rgb, plot=False)
        if lines is not None:
            lines = lines.squeeze(1)
            all_lines.append(lines)

    all_lines = [item for sublist in all_lines for item in sublist]

    return  all_lines



def cluster_lines(lines,width,height):
    # line_clusters_angles = cluster_lines(all_lines)
    line_angles_to_horizontal = []
    line_angles_to_vertical = []

    for line in lines:
        delta_x = line[2] - line[0]
        delta_y = line[3] - line[1]
        angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
        angle_vertical = np.arctan2(delta_x, delta_y) * 180 / np.pi
        line_angles_to_vertical.append(angle_vertical)
        line_angles_to_horizontal.append(angle)

    all_lines_polar = [cartesian_to_polar(l) for l in lines]
    distances = np.array([r for r, theta in all_lines_polar]).reshape(-1, 1)
    max_possible_dist = np.sqrt(width ** 2 + height ** 2)
    distances = distances / max_possible_dist
    normalized_angles = np.array([theta for r, theta in all_lines_polar]).reshape(-1, 1) / np.pi
    all_lines_polar = np.concatenate([distances, normalized_angles], axis=1)

    clusters_polar, indices = cluster_lines_polar(all_lines_polar, eps=0.02)

    line_clusters = {}
    for idx, cluster in clusters_polar.items():
        line_clusters[idx] = [lines[i] for i in indices[idx]]

    return line_clusters


def get_best_line_from_clusters(line_clusters):
    line_candidates = []
    angles = []
    pred_line_indices = []
    for idx, cluster in enumerate(line_clusters):
        pred_line_indices.append(idx)

        fitted_line = fit_line_to_points(cluster).astype(int)
        delta_x = fitted_line[2] - fitted_line[0]
        delta_y = fitted_line[3] - fitted_line[1]
        angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

        line_candidates.append(fitted_line)
        angles.append(angle)

    if len(angles) == 0:
        logging.warning("Could not fit line to any cluster")
    else:

        best_line_idx = np.abs(angles).argmin()
        fitted_line = line_candidates[best_line_idx]


    return fitted_line,best_line_idx


def overlay_lines_on_mask(mask,lines,save_dir,fitted_line = None):
    surface_mask_rgb = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
    surface_mask_rgb_cp = surface_mask_rgb.copy()

    for line in lines:
        cv2.line(surface_mask_rgb_cp, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

    if fitted_line is not None:
        cv2.line(surface_mask_rgb_cp, (fitted_line[0], fitted_line[1]), (fitted_line[2], fitted_line[3]),
                 (255, 0, 255),
                 5)

    # plt.imshow(surface_mask_rgb_cp)
    # plt.show()

    # save image:
    cv2.imwrite(os.path.join(save_dir, "lines.jpg"), surface_mask_rgb_cp)


def plot_transformed_axes_on_image(intrinsic_parameters,transform,cur_img,vis_path):
    viz_axes_vals = np.float32([[0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1]]).T

    # make homogenous
    viz_axes = np.concatenate([viz_axes_vals, np.ones((1, 4))], axis=0)

    intrinsic_matrix = np.array([[intrinsic_parameters["fx"], 0, intrinsic_parameters["cx"]],
                                 [0, intrinsic_parameters["fy"], intrinsic_parameters["cy"]], [0, 0, 1]])

    viz_axes_camera_frames = np.dot(transform, viz_axes)
    viz_axes_new_coordinate_system = viz_axes_camera_frames[:3, :]

    # revecs is identity
    rvecs = np.eye(3).astype(np.float32)
    rvecs = cv2.Rodrigues(rvecs)[0]
    tvecs = np.zeros(3).astype(np.float32)[:, None]




    imgpts, jac = cv2.projectPoints(viz_axes_new_coordinate_system, rvecs, tvecs,
                                    intrinsic_matrix.astype(np.float32), None)

    imgpts = imgpts.astype(int)

    img_center = np.array([cur_img.shape[1] / 2, cur_img.shape[0] / 2])
    translation = img_center - imgpts[0]

    imgpts = imgpts + translation

    imgpts = imgpts.astype(int)

    corners = imgpts[0].astype(int)
    corner = tuple(corners[0].ravel())

    img = np.array(cur_img).copy()

    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (255, 0, 0), 5)

    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[3].ravel()), (0, 0, 255), 5)

    # save_img:
    cv2.imwrite(os.path.join(vis_path, "axes.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))