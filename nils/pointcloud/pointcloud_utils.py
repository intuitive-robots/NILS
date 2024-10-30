import copy
import math
import os
from typing import Optional, Sequence, cast

import numpy as np
import open3d as o3d
import scipy
import torch
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA


def depth_frame_to_camera_space_xyz(
        depth_frame: torch.Tensor, mask: Optional[torch.Tensor], fov: float = 90
) -> torch.Tensor:
    """Transforms a input depth map into a collection of xyz points (i.e. a
    point cloud) in the camera's coordinate frame.

    # Parameters
    depth_frame : A square depth map, i.e. an MxM matrix with entry `depth_frame[i, j]` equaling
        the distance from the camera to nearest surface at pixel (i,j).
    mask : An optional boolean mask of the same size (MxM) as the input depth. Only values
        where this mask are true will be included in the returned matrix of xyz coordinates. If
        `None` then no pixels will be masked out (so the returned matrix of xyz points will have
        dimension 3x(M*M)
    fov: The field of view of the camera.

    # Returns

    A 3xN matrix with entry [:, i] equalling a the xyz coordinates (in the camera's coordinate
    frame) of a point in the point cloud corresponding to the input depth frame.
    """
    assert (
            len(depth_frame.shape) == 2 and depth_frame.shape[0] == depth_frame.shape[1]
    ), f"depth has shape {depth_frame.shape}, we only support (N, N) shapes for now."

    resolution = depth_frame.shape[0]
    if mask is None:
        mask = torch.ones_like(depth_frame, dtype=torch.bool)

    # pixel centers
    camera_space_yx_offsets = (
            torch.stack(torch.where(mask))
            + 0.5  # Offset by 0.5 so that we are in the middle of the pixel
    )

    # Subtract center
    camera_space_yx_offsets -= resolution / 2.0

    # Make "up" in y be positive
    camera_space_yx_offsets[0, :] *= -1

    # Put points on the clipping plane
    camera_space_yx_offsets *= (2.0 / resolution) * math.tan((fov / 2) / 180 * math.pi)

    # noinspection PyArgumentList
    camera_space_xyz = torch.cat(
        [
            camera_space_yx_offsets[1:, :],  # This is x
            camera_space_yx_offsets[:1, :],  # This is y
            torch.ones_like(camera_space_yx_offsets[:1, :]),
        ],
        axis=0,
    )

    return camera_space_xyz * depth_frame[mask][None, :]


def get_iou_bbox(pred_box, gt_box):
    """pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1. get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = ((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]) +
           (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    return iou, inters


def get_bbox_intersection_area(pred_box, gt_box):
    """pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1. get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    return inters


def is_inside_2d(scr_box, target_box, thresh=0.9):
    src_x1, src_y1, src_x2, src_y2 = scr_box
    target_x1, target_y1, target_x2, target_y2 = target_box
    intersection_area = max(0, min(src_x2, target_x2) - max(src_x1, target_x1)) * \
                        max(0, min(src_y2, target_y2) - max(src_y1, target_y1))

    # Calculate source box area
    src_box_area = (src_x2 - src_x1) * (src_y2 - src_y1)

    # Calculate intersection ratio
    intersection_ratio = intersection_area / src_box_area

    return intersection_ratio >= thresh


def get_iou(pred_box, gt_box):
    """pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1. get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = ((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]) +
           (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    return iou, inters


def closest_point_on_plane(point, plane_coefficients):
    """
    Find the closest point on a plane to a given point.

    Args:
    point (array-like): The coordinates of the point (x0, y0, z0).
    plane_coefficients (array-like): The coefficients of the plane (a, b, c, d).

    Returns:
    np.ndarray: The coordinates of the closest point on the plane.
    """
    x0, y0, z0 = point
    a, b, c, d = plane_coefficients

    # Define the plane normal vector
    normal = np.array([a, b, c])

    # Calculate the signed distance from the point to the plane
    t = (a * x0 + b * y0 + c * z0 + d) / (a ** 2 + b ** 2 + c ** 2)

    # Find the closest point on the plane
    closest_point = np.array([x0, y0, z0]) - t * normal

    return closest_point



def canonicalize_point_cloud_with_given_pc(pcd, pcd_ref, start_point_3d, end_point_3d,
                                           axis="x"):
    pcd = copy.deepcopy(pcd)
    canonicalized = True

    max_extend = np.linalg.norm(pcd_ref.get_max_bound() - pcd_ref.get_min_bound())

    n_points_ref = len(np.array(pcd_ref.points))
    # plane_model, inliers = pcd_ref.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    plane_model, inliers = pcd_ref.segment_plane(distance_threshold=max_extend / 150, ransac_n=min(n_points_ref, 6),
                                                 num_iterations=4000)
    
    selected_plane_pcd = pcd_ref.select_by_index(inliers)

    # remove outlierss

    points = np.asarray(selected_plane_pcd.points)
    center_point = start_point_3d if start_point_3d is not None else np.mean(np.array(pcd.points), axis=0)

    
    
    normal = plane_model[:3] / np.linalg.norm(plane_model[:3])
    
    if np.dot(plane_model[:3], [0, -1, 0]) < 0:
        normal = -normal
    new_y = normal / np.linalg.norm(normal)


    if start_point_3d is not None and end_point_3d is not None:
        start_point_plane = closest_point_on_plane(start_point_3d, plane_model)
        end_point_plane = closest_point_on_plane(end_point_3d, plane_model)
        right_vector = end_point_plane - start_point_plane
        right_vector /= np.linalg.norm(right_vector)
        
        if np.dot(right_vector, [1, 0, 0]) < 0:
            right_vector = -right_vector
        
        new_x = right_vector
        angle = np.arccos(np.clip(np.dot(normal, new_x), -1.0, 1.0))
        angle_deg = np.degrees(angle)
        projection = np.dot(new_x, normal) * normal
        perp_right = new_x - projection

        new_x = perp_right / np.linalg.norm(perp_right)

        if angle_deg < 80:
            new_x = np.cross(new_y, [0, 0, -1])
            
        new_z = np.cross(new_x, new_y)
        new_z /= np.linalg.norm(new_z)
        new_x = np.cross(new_y, new_z)
        new_x = new_x / np.linalg.norm(new_x)
        
    else:
        new_x = np.cross(new_y, [0, 0, -1])
        new_x = new_x / np.linalg.norm(new_x)
        new_z = np.cross(new_x, new_y)
        new_z = new_z / np.linalg.norm(new_z)
        
        
    final_transform = np.identity(4)
    final_transform[:3, :3] = np.vstack((new_x, new_y, new_z)).T
    final_transform[:3, 3] = center_point
        
    pcd.transform(final_transform)
    final_inverse = np.linalg.inv(final_transform)
    
    return final_transform, selected_plane_pcd,final_inverse

    return pcd, canonicalized, fallback_transformation, selected_plane_pcd, combined_transformation, combined_inverse


def normalize(vector):
    """Normalize a vector."""
    return vector / np.linalg.norm(vector)


def align_vectors_y_axis(v1, v2):
    """Calculate the rotation matrix around the Y-axis to align v1 with v2."""
    # Project the vectors onto the XZ-plane
    v1_proj = np.array([v1[0], 0, v1[2]])
    v2_proj = np.array([v2[0], 0, v2[2]])

    # Normalize the projections
    v1_proj = normalize(v1_proj)
    v2_proj = normalize(v2_proj)

    # Calculate the angle between the projections
    angle = angle_between_vectors(v1_proj, v2_proj)

    # Determine the direction of rotation
    cross_product = np.cross(v1_proj, v2_proj)
    if cross_product[1] < 0:  # If the Y component of the cross product is negative, invert the angle
        angle = -angle

    angle = angle + 0.1

    # Create the rotation matrix around the Y-axis
    rotation_matrix = rotation_matrix_y(angle)

    return rotation_matrix, angle


def angle_between_vectors(v1, v2):
    """Calculate the angle between two vectors in radians."""
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))


def rotation_matrix_y(angle):
    """Create a rotation matrix for a given angle around the Y-axis."""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([
        [cos_angle, 0, sin_angle, 0],
        [0, 1, 0, 0],
        [-sin_angle, 0, cos_angle, 0],
        [0, 0, 0, 1]
    ])


def canonicalize_point_cloud(pcd, canonicalize_threshold=0.3):
    # Segment the largest plane, assumed to be the floor
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

    canonicalized = False
    print(len(inliers) / len(pcd.points))
    if len(inliers) / len(pcd.points) > canonicalize_threshold:
        canonicalized = True

        # Ensure the plane normal points upwards
        print(np.dot(plane_model[:3], [0, 1, 0]))
        if np.dot(plane_model[:3], [0, 1, 0]) < 0:
            plane_model = -plane_model

        # Normalize the plane normal vector
        normal = plane_model[:3] / np.linalg.norm(plane_model[:3])

        # Compute the new basis vectors
        new_y = normal
        new_x = np.cross(new_y, [0, 0, -1])
        new_x /= np.linalg.norm(new_x)
        new_z = np.cross(new_x, new_y)

        # Create the transformation matrix
        transformation = np.identity(4)
        transformation[:3, :3] = np.vstack((new_x, new_y, new_z)).T
        transformation[:3, 3] = -np.dot(transformation[:3, :3], pcd.points[inliers[0]])

        # Apply the transformation
        pcd.transform(transformation)

        # Additional 180-degree rotation around the Z-axis
        rotation_z_180 = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                                   [np.sin(np.pi), np.cos(np.pi), 0],
                                   [0, 0, 1]])
        pcd.rotate(rotation_z_180, center=(0, 0, 0))

        return pcd, canonicalized, transformation
    else:
        return pcd, canonicalized, None


def is_inside(src_pts, target_pts, thresh=0.5):
    try:
        hull = scipy.spatial.ConvexHull(target_pts)
    except:
        return False
    # print("vertices of hull: ", np.array(hull.vertices).shape)
    # hull_vertices = np.array([[0,0,0]])
    hull_vertices = np.zeros((src_pts.shape[1]))[None]
    for v in hull.vertices:
        hull_vertices = np.vstack((hull_vertices, target_pts[v]))
    hull_vertices = hull_vertices[1:]

    num_src_pts = len(src_pts)
    # Don't want threshold to be too large (specially with more objects, like 4, 0.9*thresh becomes too large)
    thresh_obj_particles = thresh * num_src_pts
    src_points_in_hull = in_hull(src_pts, hull_vertices)
    # print("src pts in target, thresh: ", src_points_in_hull.sum(), thresh_obj_particles)
    if src_points_in_hull.sum() > thresh_obj_particles:
        return True
    else:
        return False


def in_hull(p, hull):
    """Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, scipy.spatial.Delaunay):
        hull = scipy.spatial.Delaunay(hull)

    return hull.find_simplex(p) >= 0


def get_pcd_dist(pcd_A, pcd_B):
    dist_pcd1_to_pcd2 = np.asarray(pcd_A.compute_point_cloud_distance(pcd_B))
    dist_pcd2_to_pcd1 = np.asarray(pcd_B.compute_point_cloud_distance(pcd_A))
    combined_distances = np.concatenate((dist_pcd1_to_pcd2, dist_pcd2_to_pcd1))
    min_dist = np.min(combined_distances)
    avg_dist = np.mean(combined_distances)

    return avg_dist, min_dist


def get_seg_mask_dist(seg_mask_A, seg_mask_B):
    pts_A = np.array(np.where(seg_mask_A))
    pts_B = np.array(np.where(seg_mask_B))

    dists = np.linalg.norm(pts_A[:, None, :] - pts_B[:, :, None], axis=0)

    min_dist = np.min(dists)
    return min_dist


def is_merge(node_1, node_2):
    if node_1.pcd is None or node_2.pcd is None:
        dist = get_seg_mask_dist(node_1.seg_mask, node_2.seg_mask)

    else:

        if len(np.array(node_1.pcd)) == 0 or len(np.array(node_2.pcd)) == 0:
            return True
        else:

            dist = get_pcd_dist(node_1.pcd, node_2.pcd)
    if dist < 0.01:
        return True, dist
    else:
        return False, dist


################
# FOR DEBUGGNG #
################
# The below functions are versions of the above which, because of their reliance on
# numpy functions, cannot use GPU acceleration. These are possibly useful for debugging,
# performance comparisons, or for validating that the above GPU variants work properly.


def _cpu_only_camera_space_xyz_to_world_xyz(
        camera_space_xyzs: np.ndarray,
        camera_world_xyz: np.ndarray,
        rotation: float,
        horizon: float,
):
    # Adapted from https://github.com/devendrachaplot/Neural-SLAM.

    # view_position = 3, world_points = 3 x N
    # NOTE: camera_position is not equal to agent_position!!

    # First compute the transformation that points undergo
    # due to the camera's horizon
    psi = -horizon * np.pi / 180
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    # fmt: off
    horizon_transform = np.array(
        [
            [1, 0, 0],  # unchanged
            [0, cos_psi, sin_psi],
            [0, -sin_psi, cos_psi, ],
        ],
        np.float64,
    )
    # fmt: on

    # Next compute the transformation that points undergo
    # due to the agent's rotation about the y-axis
    phi = -rotation * np.pi / 180
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    # fmt: off
    rotation_transform = np.array(
        [
            [cos_phi, 0, -sin_phi],
            [0, 1, 0],  # unchanged
            [sin_phi, 0, cos_phi], ],
        np.float64,
    )
    # fmt: on

    # Apply the above transformations
    view_points = (rotation_transform @ horizon_transform) @ camera_space_xyzs

    # Translate the points w.r.t. the camera's position in world space.
    world_points = view_points + camera_world_xyz[:, None]
    return world_points


def _cpu_only_depth_frame_to_camera_space_xyz(
        depth_frame: np.ndarray, mask: Optional[np.ndarray], fov: float = 90
):
    """"""
    assert (
            len(depth_frame.shape) == 2 and depth_frame.shape[0] == depth_frame.shape[1]
    ), f"depth has shape {depth_frame.shape}, we only support (N, N) shapes for now."

    resolution = depth_frame.shape[0]
    if mask is None:
        mask = np.ones(depth_frame.shape, dtype=bool)

    # pixel centers
    camera_space_yx_offsets = (
            np.stack(np.where(mask))
            + 0.5  # Offset by 0.5 so that we are in the middle of the pixel
    )

    # Subtract center
    camera_space_yx_offsets -= resolution / 2.0

    # Make "up" in y be positive
    camera_space_yx_offsets[0, :] *= -1

    # Put points on the clipping plane
    camera_space_yx_offsets *= (2.0 / resolution) * math.tan((fov / 2) / 180 * math.pi)

    camera_space_xyz = np.concatenate(
        [
            camera_space_yx_offsets[1:, :],  # This is x
            camera_space_yx_offsets[:1, :],  # This is y
            np.ones_like(camera_space_yx_offsets[:1, :]),
        ],
        axis=0,
    )

    return camera_space_xyz * depth_frame[mask][None, :]


def _cpu_only_depth_frame_to_world_space_xyz(
        depth_frame: np.ndarray,
        camera_world_xyz: np.ndarray,
        rotation: float,
        horizon: float,
        fov: float,
):
    camera_space_xyz = _cpu_only_depth_frame_to_camera_space_xyz(
        depth_frame=depth_frame, mask=None, fov=fov
    )

    world_points = _cpu_only_camera_space_xyz_to_world_xyz(
        camera_space_xyzs=camera_space_xyz,
        camera_world_xyz=camera_world_xyz,
        rotation=rotation,
        horizon=horizon,
    )

    return world_points.reshape((3, *depth_frame.shape)).transpose((1, 2, 0))


def _cpu_only_project_point_cloud_to_map(
        xyz_points: np.ndarray,
        bin_axis: str,
        bins: Sequence[float],
        map_size: int,
        resolution_in_cm: int,
        flip_row_col: bool,
):
    """Bins points into  bins.

    Adapted from https://github.com/devendrachaplot/Neural-SLAM.

    # Parameters
    xyz_points : (x,y,z) point clouds as a np.ndarray of shape (... x height x width x 3). (x,y,z)
        should be coordinates specified in meters.
    bin_axis : Either "x", "y", or "z", the axis which should be binned by the values in `bins`
    bins: The values by which to bin along `bin_axis`, see the `bins` parameter of `np.digitize`
        for more info.
    map_size : The axes not specified by `bin_axis` will be be divided by `resolution_in_cm / 100`
        and then rounded to the nearest integer. They are then expected to have their values
        within the interval [0, ..., map_size - 1].
    resolution_in_cm: The resolution_in_cm, in cm, of the map output from this function. Every
        grid square of the map corresponds to a (`resolution_in_cm`x`resolution_in_cm`) square
        in space.
    flip_row_col: Should the rows/cols of the map be flipped

    # Returns
    A collection of maps of shape (... x map_size x map_size x (len(bins)+1)), note that bin_axis
    has been moved to the last index of this returned map, the other two axes stay in their original
    order unless `flip_row_col` has been called in which case they are reversed (useful if you give
    points as often rows should correspond to y or z instead of x).
    """
    bin_dim = ["x", "y", "z"].index(bin_axis)

    start_shape = xyz_points.shape
    xyz_points = xyz_points.reshape([-1, *start_shape[-3:]])
    num_clouds, h, w, _ = xyz_points.shape

    if not flip_row_col:
        new_order = [i for i in [0, 1, 2] if i != bin_dim] + [bin_dim]
    else:
        new_order = [i for i in [2, 1, 0] if i != bin_dim] + [bin_dim]

    uvw_points: np.ndarray = np.stack([xyz_points[..., i] for i in new_order], axis=-1)

    num_bins = len(bins) + 1

    isnotnan = ~np.isnan(xyz_points[..., 0])

    uvw_points_binned = np.concatenate(
        (
            np.round(100 * uvw_points[..., :-1] / resolution_in_cm).astype(np.int32),
            np.digitize(uvw_points[..., -1:], bins=bins).astype(np.int32),
        ),
        axis=-1,
    )

    maxes = np.array([map_size, map_size, num_bins]).reshape((1, 1, 1, 3))

    isvalid = np.logical_and.reduce(
        (
            (uvw_points_binned >= 0).all(-1),
            (uvw_points_binned < maxes).all(-1),
            isnotnan,
        )
    )

    uvw_points_binned_with_index_mat = np.concatenate(
        (
            np.repeat(np.arange(0, num_clouds), h * w).reshape(-1, 1),
            uvw_points_binned.reshape(-1, 3),
        ),
        axis=1,
    )

    uvw_points_binned_with_index_mat[~isvalid.reshape(-1), :] = 0
    ind = np.ravel_multi_index(
        uvw_points_binned_with_index_mat.transpose(),
        (num_clouds, map_size, map_size, num_bins),
    )
    ind[~isvalid.reshape(-1)] = 0
    count = np.bincount(
        ind.ravel(),
        isvalid.ravel().astype(np.int32),
        minlength=num_clouds * map_size * map_size * num_bins,
    )

    return count.reshape([*start_shape[:-3], map_size, map_size, num_bins])
