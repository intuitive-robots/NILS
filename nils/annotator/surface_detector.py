# surface_transform_config.py
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from nils.scene_graph.canonicalization import cluster_lines, get_best_line_from_clusters, get_lines_from_surface_masks, get_surface_object_masks, overlay_lines_on_mask, plot_transformed_axes_on_image
from nils.scene_graph.scene_graph_utils import get_planar_transformation
from nils.specialist_models.clipseg import masks_to_boxes
from nils.specialist_models.sam2.utils.amg import remove_small_regions
from nils.utils.utils import find_points_close_to_line, get_homography_transform, get_mask_agreement

@dataclass
class SurfaceTransformConfig:
    use_depth: bool = True
    canonicalize: bool = False
    ransac_residual_threshold: float = 0.35
    ransac_max_trials: int = 200
    small_region_threshold: float = 0.1  # As percentage of image size
    normal_agreement_threshold: float = 0.45
    surface_mask_agreement_threshold: float = 0.5
    normal_surface_mask_threshold: float = 0.2
    zscore_outlier_threshold: float = 1.2
    min_ransac_inliers: int = 4
    dilation_kernel_size: Tuple[int, int] = (5, 5)
    dilation_iterations: int = 2

# surface_transform.py
import cv2
import numpy as np
import torch
from scipy.stats import zscore
from sklearn.linear_model import RANSACRegressor
import os

class SurfaceTransformer:
    def __init__(
        self,
        config: SurfaceTransformConfig,
        detection_model,
        segmentation_model,
        m3d,
        object_manager,
        intrinsic_parameters: np.ndarray
    ):
        self.cfg = config
        self.detection_model = detection_model
        self.segmentation_model = segmentation_model
        self.m3d = m3d
        self.object_manager = object_manager
        self.intrinsic_parameters = intrinsic_parameters
        self.surface_transform_matrix = None

    def compute_surface_masks(self, average_images: List[np.ndarray], surface_object_prompts: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute surface masks and update surface object."""
        if self.object_manager.surface_object is None:
            try:
                surface_masks, ious, up_normals, valid_indices, best_surface_prompt = get_surface_object_masks(
                    self.detection_model,
                    self.segmentation_model,
                    self.m3d,
                    average_images,
                    surface_object_prompts,
                    self.intrinsic_parameters
                )

                max_size = surface_masks[0].shape[0] * surface_masks[0].shape[1]
                max_size_to_remove = self.cfg.small_region_threshold * max_size
                
                normal_agreement_mask = get_mask_agreement(up_normals, threshold=self.cfg.normal_agreement_threshold)
                combined_surface_masks = get_mask_agreement(surface_masks, threshold=self.cfg.surface_mask_agreement_threshold)
                combined_surface_masks = remove_small_regions(
                    combined_surface_masks, 
                    area_thresh=max_size_to_remove, 
                    mode="holes"
                )[0]
                
                total_agreement_mask = np.logical_and(normal_agreement_mask, combined_surface_masks)
                total_agreement_mask = remove_small_regions(
                    total_agreement_mask,
                    area_thresh=max_size_to_remove,
                    mode="islands"
                )[0]

                box = np.array(masks_to_boxes(torch.tensor(total_agreement_mask)[None])).astype(int)
                
                # Update object manager
                self.object_manager.add_surface_object(best_surface_prompt, "")
                self.object_manager.surface_object.boxes = box[0]
                
                return surface_masks, up_normals, combined_surface_masks

            except Exception as e:
                return None, None, None
        else:
            return (
                self.object_manager.surface_object.mask,
                self.object_manager.surface_object.normals,
                self.object_manager.surface_object.combined_mask
            )

    def process_depth_transforms(
        self,
        surface_masks: np.ndarray,
        up_normals: np.ndarray,
        depth_predictions: List[np.ndarray],
        image_indices: List[int],
        average_images: List[np.ndarray],
        valid_indices: np.ndarray,
        all_point_pairs: List,
        vis_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process depth information and compute transformations."""
        if not self.cfg.use_depth:
            return None, None

        # Process normals and masks
        dilated_up_normals = None
        if up_normals is not None:
            kernel = np.ones(self.cfg.dilation_kernel_size, np.uint8)
            dilated_up_normals = [
                cv2.dilate(normal.astype(np.uint8), kernel, iterations=self.cfg.dilation_iterations)
                for normal in up_normals
            ]
            normal_surface_masks = [
                np.logical_and(mask, normal)
                for mask, normal in zip(surface_masks, dilated_up_normals)
            ]
        else:
            normal_surface_masks = surface_masks

        # Process combined masks
        combined_masks = None
        if normal_surface_masks is not None:
            max_size_to_remove = self.cfg.small_region_threshold * surface_masks[0].shape[0] * surface_masks[0].shape[1]
            combined_masks = get_mask_agreement(normal_surface_masks, threshold=self.cfg.normal_surface_mask_threshold)
            combined_masks = remove_small_regions(combined_masks, area_thresh=max_size_to_remove, mode="islands")[0]
            combined_masks = remove_small_regions(combined_masks, area_thresh=max_size_to_remove, mode="holes")[0]

        # Compute transforms
        transforms = []
        valid_indices = np.where(valid_indices)[0]
        
        if len(all_point_pairs) == 0:
            all_point_pairs = [None] * len(valid_indices)

        for idx, i in enumerate(valid_indices):
            cur_depth = depth_predictions[image_indices[i]]
            cur_img = average_images[i]

            transformation, _, _, _ = get_planar_transformation(
                cur_img, cur_depth, None,
                self.intrinsic_parameters, "x", combined_masks,
                edge_points=all_point_pairs[idx]
            )

            if transformation is not None:
                transforms.append(transformation)

        # Compute final transform
        final_transform = self._compute_final_transform(transforms, all_point_pairs)
        
        # Compute homography if needed
        homography_transform = None
        if combined_masks is not None:
            homography_transform = get_homography_transform(combined_masks)
            if homography_transform is not None:
                cur_img = average_images[-1]  # Use last image
                warped_img = cv2.warpPerspective(
                    np.array(cur_img),
                    homography_transform,
                    (combined_masks.shape[1], combined_masks.shape[0])
                )
                cv2.imwrite(
                    os.path.join(vis_path, "warped_img.jpg"),
                    cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)
                )

        # Visualize results
        if final_transform is not None:
            plot_transformed_axes_on_image(
                self.intrinsic_parameters,
                final_transform,
                average_images[-1],
                vis_path
            )

        return final_transform, homography_transform

    def _compute_final_transform(
        self,
        transforms: List[np.ndarray],
        all_point_pairs: List
    ) -> Optional[np.ndarray]:
        """Compute final transformation matrix using RANSAC and averaging."""
        if len(transforms) == 0:
            return None

        transforms = np.array(transforms)
        
        # Filter outliers using z-score
        zscores = zscore(transforms[:, :3, :3])
        transform_filter = np.any(abs(zscores > self.cfg.zscore_outlier_threshold), axis=(1, 2))
        initial_transform = np.mean(transforms[~transform_filter], axis=0)

        # RANSAC for point pairs
        none_point_pairs = [False if pair is None else True for pair in all_point_pairs]
        if len(none_point_pairs) < self.cfg.min_ransac_inliers or np.sum(none_point_pairs) < self.cfg.min_ransac_inliers:
            return initial_transform

        transforms_filtered = transforms[none_point_pairs]
        transforms_filtered = transforms_filtered[:, :3, :3]
        flattened_transforms = transforms_filtered.reshape(transforms_filtered.shape[0], -1)
        X_dummy = np.ones((flattened_transforms.shape[0], 1))

        ransac = RANSACRegressor(
            random_state=42,
            residual_threshold=self.cfg.ransac_residual_threshold,
            max_trials=self.cfg.ransac_max_trials
        )
        ransac.fit(X_dummy, flattened_transforms)
        inlier_mask = ransac.inlier_mask_

        if np.sum(inlier_mask) < self.cfg.min_ransac_inliers:
            return initial_transform

        # Compute final transform from RANSAC inliers
        inliers = np.where(inlier_mask)[0]
        inlier_transforms = transforms_filtered[inliers]
        final_transform = np.mean(inlier_transforms, axis=0)

        final_transform_4x4 = np.eye(4)
        final_transform_4x4[:3, :3] = final_transform
        
        return final_transform_4x4

    def compute_transform(
        self,
        average_images: List[np.ndarray],
        surface_object_prompts: List[str],
        depth_predictions: List[np.ndarray],
        image_indices: List[int],
        vis_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Main method to compute surface transformation."""
        # Compute surface masks
        surface_masks, up_normals, combined_surface_masks = self.compute_surface_masks(
            average_images,
            surface_object_prompts
        )

        # Handle canonicalization if enabled
        all_point_pairs = []
        if self.cfg.canonicalize and surface_masks is not None:
            all_point_pairs = self._compute_canonicalization(surface_masks, combined_surface_masks, average_images, vis_path)

        valid_indices = np.array(list(range(len(average_images))))

        # Process depth and compute transforms
        final_transform, homography_transform = self.process_depth_transforms(
            surface_masks,
            up_normals,
            depth_predictions,
            image_indices,
            average_images,
            valid_indices,
            all_point_pairs,
            vis_path
        )

        self.surface_transform_matrix = homography_transform
        return final_transform, homography_transform

    def _compute_canonicalization(
        self,
        surface_masks: np.ndarray,
        combined_surface_masks: np.ndarray,
        average_images: List[np.ndarray],
        vis_path: str
    ) -> List:
        """Compute canonicalization for surface masks."""
        max_size_to_remove = self.cfg.small_region_threshold * surface_masks[0].shape[0] * surface_masks[0].shape[1]
        all_lines = get_lines_from_surface_masks(surface_masks, max_size_to_remove)

        if len(all_lines) == 0:
            return []

        line_clusters = cluster_lines(all_lines, average_images[0].shape[0], average_images[0].shape[1])
        line_clusters.pop(-1, None)
        
        line_clusters = [val for key, val in line_clusters.items()]
        cluster_sizes = [len(cluster) for cluster in line_clusters]
        top_k_clusters = np.argsort(cluster_sizes)[-3:][::-1]
        
        top3_line_clusters = [line_clusters[idx] for idx in top_k_clusters]
        best_line, best_line_cluster_idx = get_best_line_from_clusters(top3_line_clusters)

        if best_line is None:
            return []

        overlay_lines_on_mask(
            combined_surface_masks,
            top3_line_clusters[best_line_cluster_idx],
            max_size_to_remove,
            vis_path,
            best_line
        )

        all_point_pairs = []
        for mask in surface_masks:
            _, max_dist_points = find_points_close_to_line(best_line, mask, plot=False)
            all_point_pairs.append(max_dist_points)

        return all_point_pairs