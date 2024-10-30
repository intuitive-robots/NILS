import copy

import cv2
import numpy as np
import open3d as o3d
import sympy
import torch

from nils.pointcloud.pointcloud import (
    apply_mask_to_image,
    create_point_cloud_from_rgbd,
)
from nils.pointcloud.pointcloud_utils import (
    canonicalize_point_cloud_with_given_pc,
)



def sRT_to_4x4(scale, R, T):
    trf = torch.eye(4)
    trf[:3, :3] = R * scale
    trf[:3, 3] = T.ravel()  # doesn't need scaling
    return trf


def pixel_grid(H, W):
    return np.mgrid[:W, :H].T.astype(np.float32)


def fast_pnp(pts3d,focal, pp=None, niter_PnP=10):


    H, W, THREE = pts3d.shape
    assert THREE == 3


    pixels = pixel_grid(H, W)

    msk = np.ones((H, W), dtype=np.uint8).astype(bool)


    if pp is None:
        pp = (W/2, H/2)

    K = np.float32([(focal[0], 0, pp[0]), (0, focal[1], pp[1]), (0, 0, 1)])

    success, R, T, inliers = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
                                                iterationsCount=niter_PnP, reprojectionError=5,
                                                flags=cv2.SOLVEPNP_SQPNP)

    score = len(inliers)
    R = cv2.Rodrigues(R)[0]  # world to cam

    R, T = map(torch.from_numpy, (R, T))


    w_t_cam = sRT_to_4x4(1, R, T)
    return w_t_cam, score





def get_planar_transformation(image, depth, ref_object,intrinsic_parameters,axis, mask = None,edge_points = None):
    
    width = image.shape[1]
    height = image.shape[0]

    pcd_orig = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.array(image)),
            o3d.geometry.Image(np.array(depth).astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        ),
        o3d.camera.PinholeCameraIntrinsic(**intrinsic_parameters)
    )

    pcd_orig_shape = np.array(pcd_orig.points).reshape(image.shape[0], image.shape[1], 3)

    #pcd_orig_shape = np.array(pcd_orig.points).reshape(image.shape[0], image.shape[1], 3)
    if edge_points is not None:
        start_point_3D = pcd_orig_shape[edge_points[0][1], edge_points[0][0]]
        end_point_3D = pcd_orig_shape[edge_points[1][1], edge_points[1][0]]
    else:
        start_point_3D = None
        end_point_3D = None



    scale = np.linalg.norm(np.array(pcd_orig.points).std(axis=0)) * 1.0 + 1e-6

    pcd_orig = pcd_orig.voxel_down_sample(voxel_size=max(0.001, scale / 40))
    pcd_orig = pcd_orig.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]



    ref_obj = ref_object
    if ref_obj is not None:
        mask = ref_obj["mask"]
        mask_binary = mask > 0
        masked_rgb = apply_mask_to_image(np.array(image), mask_binary)
        masked_depth = apply_mask_to_image(depth, mask_binary)
    else:
        if mask is not None:
            mask = mask.astype(bool)
            masked_rgb = apply_mask_to_image(np.array(image), mask)
            masked_depth = apply_mask_to_image(depth, mask)
        else:
            masked_rgb = np.array(image)
            masked_depth = np.array(depth)
    pcd = create_point_cloud_from_rgbd(masked_rgb, masked_depth, intrinsic_parameters)



    if len(np.array(pcd.points)) == 0:
        if ref_obj is None:
            return None,intrinsic_parameters, None
        mask = np.zeros_like(image[:, :, 0])
        # fill mask with ones at bb:
        mask[ref_obj["box"][1]:ref_obj["box"][3], ref_obj["box"][0]:ref_obj["box"][2]] = 1
        mask_binary = mask > 0
        masked_rgb = apply_mask_to_image(np.array(image), mask_binary)
        masked_depth = apply_mask_to_image(depth, mask_binary)
        pcd = create_point_cloud_from_rgbd(masked_rgb, masked_depth, intrinsic_parameters)





    scale = np.linalg.norm(np.array(pcd.points).std(axis=0)) * 1.0 + 1e-6
    pcd = pcd.voxel_down_sample(voxel_size=max(0.001, scale / 40))




    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.0)
    inlier_cloud = pcd.select_by_index(ind)
    project_pcd = inlier_cloud
    if len(np.array(project_pcd.points)) > 0:

        transformation,plane_pcd,combined_inverse = canonicalize_point_cloud_with_given_pc(pcd_orig,
                                                                                               project_pcd,start_point_3D,
                                                                                                         end_point_3D,
                                                                                                         axis = axis)
        
    else:
        transformation = None
    return transformation, intrinsic_parameters,plane_pcd,combined_inverse

def get_extended_relations(norm_vector, threshold = 0.62):
    """Retrieve the extended relations for a given normal vector, including front left behind left etc..."""

    relations = ["on the"]


    if norm_vector[2] > threshold:
        relations.append("front")
    elif norm_vector[2] < -threshold:
        relations.append("behind")
    if norm_vector[0] > threshold:
        relations.append("right")
    elif norm_vector[0] < -threshold:
        relations.append("left")
    if norm_vector[1] > threshold:
        relations.append("below")
    elif norm_vector[1] < -threshold:
        relations.append("above")

    if len(relations) == 0:
        return None
    else:
        relations.append("of")
        return " ".join(relations)

    return relations
def get_bbox_center(bbox):
    if isinstance(bbox,o3d.geometry.AxisAlignedBoundingBox):
        return bbox.get_center()
    
    if bbox.ndim == 2:
        x = (bbox[:, 0] + bbox[:, 2]) / 2
        y = (bbox[:, 1] + bbox[:, 3]) / 2
        return np.array([x, y])
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return np.array([x, y])

def get_closest_point_on_mask(mask,point):

    mask_points = np.array(np.where(mask)).T

    #Shift y and x :
    mask_points = mask_points[:,::-1]

    dists = np.linalg.norm(mask_points-point[None],axis = 1)

    return mask_points[np.argmin(dists)]


def get_corner_points(bbox):
    if isinstance(bbox,o3d.geometry.AxisAlignedBoundingBox):
        return np.array(bbox.get_box_points())
    else:
        return bbox



def get_bbox_from_segmentation(seg_mask):

    x_min = np.min(np.where(seg_mask)[1])
    x_max = np.max(np.where(seg_mask)[1])
    y_min = np.min(np.where(seg_mask)[0])
    y_max = np.max(np.where(seg_mask)[0])

    return np.array([x_min,y_min,x_max,y_max])



def get_object_movements_from_scene_graphs(scene_graphs, tracking_model, visible_objects=None):
    object_movements_nl = ["placeholder"]

    object_movements_lst = []

    for i in range(1, len(scene_graphs)):
        scene_graph = scene_graphs[i - 1]
        next_scene_graph = scene_graphs[i]

        # object_movements = next_scene_graph.prev_movements(next_scene_graph, tracking_model)
        object_movements = next_scene_graph.prev_movement

        object_movements_lst.append(object_movements)

        object_movements_nl.append(scene_graph.get_nl_movements(object_movements, visible_objects))

    return object_movements_nl, object_movements_lst


def get_max_object_movement_from_scene_graphs(scene_graphs, object_movements_precomputed):
    max_object_movements = {}

    for object_movements in object_movements_precomputed:
        # scene_graph = scene_graphs[i - 1]
        # next_scene_graph = scene_graphs[i]

        # object_movements = scene_graph.get_object_movements(next_scene_graph)

        for key, value in object_movements.items():
            if value is not None:
                if key not in max_object_movements.keys():
                    max_object_movements[key] = value
                else:
                    stacked_values = np.stack([max_object_movements[key], value])
                    max_object_movements[key] = stacked_values[np.argmax([np.linalg.norm(stacked_values, axis=1)]), :]

    return max_object_movements


def filter_proximity_objects(object_distances):
    relevant_frames = -2  # last two frames to determine closest objects

    closest_objects = {}
    min_dists = []
    close_objects = []
    for dist_dict in object_distances[relevant_frames:]:
        obj, dists = np.array(list(dist_dict.keys())), np.array(list(dist_dict.values()))
        dists[dists == None] = np.inf
        dists_sorted = np.argsort(dists)
        min_dist = dists_sorted[1]
        close_objects.append(obj[dists_sorted[:2]])
        closest_objects[obj[min_dist]] = dists[min_dist]
        min_dists.append(dists[min_dist])

    mean_min_dist = np.mean(min_dists)
    close_objects = np.concatenate(close_objects, axis=0)

    close_objects = [{key: value for key, value in dist.items() if
                      value is not None and value < max(mean_min_dist, 30) or key in close_objects} for dist in
                     object_distances]

    return close_objects


def get_visible_objects(scene_graphs):
    """Filters objects that were not visible for most of the frames
    Args:
        scene_graphs:

    Returns:

    """
    obj_visibility_counts = {}
    for scene_graph in scene_graphs:
        for node in scene_graph.nodes:
            if node.name not in obj_visibility_counts.keys():
                obj_visibility_counts[node.name] = 0
            if node.visible:
                obj_visibility_counts[node.name] += 1
    total_frames = len(scene_graphs)
    visible_objects = [key for key, value in obj_visibility_counts.items() if value / total_frames > 0.4]

    return visible_objects


def get_gripper_object_distances_from_scene_graphs(scene_graphs, visible_objects=None):
    nl_object_distances = []
    object_distances = []
    for scene_graph in scene_graphs:
        object_distance = scene_graph.get_gripper_object_distances()

        object_distances.append(object_distance)

    filtered_object_distances = filter_proximity_objects(object_distances)

    for object_distance in filtered_object_distances:
        nl_object_distance = scene_graph.get_nl_gripper_object_dists(object_distance, visible_objects)
        nl_object_distances.append(nl_object_distance)
        s = 1

    return nl_object_distances


def get_scene_graph_edge_changes(scene_graphs):
    edge_changes = set()
    prev_edge_changes = set()
    for i in range(1, len(scene_graphs)):
        scene_graph = scene_graphs[i - 1]
        next_scene_graph = scene_graphs[i]

        #edge_changes.append(scene_graph.get_edge_changes(next_scene_graph))

        edge_change = set(next_scene_graph.edges.values()) - set(
            scene_graph.edges.values())
        prev_edge_change = set(scene_graph.edges.values()) - set(
            next_scene_graph.edges.values())
        
        edge_change_strings = set([edge.get_string() for edge in edge_change])
        prev_edge_change_strings = set([edge.get_string() for edge in prev_edge_change])
        edge_changes = edge_changes.union(edge_change_strings)
        prev_edge_changes = prev_edge_changes.union(prev_edge_change_strings)
        
        
    if len(edge_changes) == 0:
        return None

    return (prev_edge_changes,edge_changes)


#https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1fbd43f3827fffeb76641a9c5ab5b625eb5a75ba




def transform_bbox(bbox, M):
    """Transform a bounding box using a perspective transform matrix.

    Args:
    - bbox: List or tuple containing the bounding box coordinates (x1, y1, x2, y2).
    - M: Perspective transform matrix of shape 3x3.

    Returns:
    - Transformed bounding box coordinates (x1', y1', x2', y2').
    """
    # Convert bbox to homogeneous coordinates (x, y, 1)
    bbox_homogeneous = np.array([[bbox[0], bbox[1], 1],
                                 [bbox[2], bbox[1], 1],
                                 [bbox[0], bbox[3], 1],
                                 [bbox[2], bbox[3], 1]])

    # Apply perspective transformation
    transformed_bbox_homogeneous = np.dot(M, bbox_homogeneous.T).T

    # Convert back to Cartesian coordinates and normalize
    transformed_bbox_cartesian = transformed_bbox_homogeneous[:, :2] / transformed_bbox_homogeneous[:, 2][:, np.newaxis]

    # Get transformed bounding box coordinates (x1', y1', x2', y2')
    transformed_bbox = (np.min(transformed_bbox_cartesian[:, 0]),
                        np.min(transformed_bbox_cartesian[:, 1]),
                        np.max(transformed_bbox_cartesian[:, 0]),
                        np.max(transformed_bbox_cartesian[:, 1]))

    return np.array(transformed_bbox).astype(int)


def get_averaged_movement_between_sg(init_scene_graphs, final_scene_graphs):
    init_scene_graphs_avg = average_scene_graphs(init_scene_graphs)
    final_scene_graphs_avg = average_scene_graphs(final_scene_graphs)

    movement = get_movement_between_sg(init_scene_graphs_avg, final_scene_graphs_avg)

    return movement

def get_nl_movement_3d(object_movements):

    movement_nl = ""

    for obj,movement in object_movements.items():
        if movement is not None:
            if abs(movement[2]) > 0.3:
                if movement[2] > 0.3:
                    movement_nl += f"{obj} moved up by {movement[2]} meters. "
                elif movement[2] < 0.3:
                    movement_nl += f"{obj} moved down by {movement[2]} meters. "
            if abs(movement[0]) > 0.3:
                if movement[0] > 0.3:
                    movement_nl += f"{obj} moved right by {movement[0]} meters. "
                elif movement[0] < 0.3:
                    movement_nl += f"{obj} moved left by {movement[0]} meters. "
            if abs(movement[1]) > 0.3:
                if movement[1] > 0.3:
                    movement_nl += f"{obj} moved forward by {movement[1]} meters. "
                elif movement[1] < 0.3:
                    movement_nl += f"{obj} moved backward by {movement[1]} meters. "
    return movement_nl

def get_nl_movement_2d(object_movements):
    movement_nl = ""

    for obj,movement in object_movements.items():
        if movement is not None:
            if movement[1] > 0.3:
                movement_nl += f"{obj} moved right by {movement[1]} meters. "
            elif movement[1] < 0.3:
                movement_nl += f"{obj} moved left by {movement[1]} meters. "
            if movement[0] > 0.3:
                movement_nl += f"{obj} moved forward by {movement[0]} meters. "
            elif movement[0] < 0.3:
                movement_nl += f"{obj} moved backward by {movement[0]} meters. "


def get_nl_movement(object_movements):
    if list(object_movements.values())[0].shape[0] == 3:
        return get_nl_movement_3d(object_movements)
    else:
        return get_nl_movement_2d(object_movements)



def get_movement_between_sg(sg,future_sg):
    movements = {}
    for node in sg.nodes:
        other_node = future_sg.get_node(node.name)

        if other_node.pos is None or node.pos is None:
            movements[node.name] = np.array([0,0,0])
            continue
        movement = other_node.pos - node.pos
        movements[node.name] = movement

    return movements



def get_majority_edge(count_dict,threshold = 4):

    keys = [(edge.start.name, edge.end.name) for  count, edge in count_dict.values()]
    edges = [edge for count, edge in count_dict.values()]
    counts = [count for count, edge in count_dict.values()]

    unique_tuples = {}
    max_scores = {}

    for tuple_element, score, index in zip(keys, counts, range(len(keys))):
        if score < threshold:
            continue
        if tuple_element not in unique_tuples:
            unique_tuples[tuple_element] = index
            max_scores[tuple_element] = score
        elif score > max_scores[tuple_element]:
            max_scores[tuple_element] = score
            unique_tuples[tuple_element] = index

    selected_indices = [unique_tuples[tuple_element] for tuple_element in unique_tuples]

    return {keys[i]: edges[i] for i in selected_indices}




def average_scene_graph_edges(scene_graphs, count_threshold):
    count_dict = {}
    for scene_graph in scene_graphs:
        count_edges(scene_graph,count_dict)

    #only select edges over threshold:
    threshold = count_threshold
    #edges = filter_edges_by_count(count_dict, threshold)
    edges = get_majority_edge(count_dict,threshold)


    return edges

def average_scene_graphs(scene_graphs):
    
    edges = average_scene_graph_edges(scene_graphs, min(len(scene_graphs) // 2,2))



    projected_sg = copy.deepcopy(scene_graphs[0])
    try:
        for node in projected_sg.nodes:
            node_boxes = [sg.get_node(node.name).bbox2d for sg in scene_graphs]
            node_pcds = [sg.get_node(node.name).pcd for sg in scene_graphs]
            node_pos = [sg.get_node(node.name).pos for sg in scene_graphs]

            node.pos = np.mean(node_pos, axis=0)
            combined_pcd = np.sum(node_pcds, axis=0)
            combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.005)
            node.visible = np.mean([sg.get_node(node.name).visible for sg in scene_graphs]) > 0.5
            node.bbox = np.mean(node_boxes, axis=0).astype(int)
            new_pcd = combined_pcd
            #new_pcd.points = o3d.utility.Vector3dVector(np.mean(np.array(node_pcds.points), axis=0))
            node.pcd = new_pcd
            node.seg_mask = np.mean([sg.get_node(node.name).seg_mask for sg in scene_graphs], axis=0) > 0.5
    except Exception as e:
        print(e)


    projected_sg.edges = edges

    return projected_sg












def count_edges(scene_graph,count_dict):
    for edge in scene_graph.edges.values():
        if (edge.start.name, edge.end.name, edge.edge_type) not in count_dict:
            count_dict[(edge.start.name, edge.end.name, edge.edge_type)] = (0, edge)
        count_dict[(edge.start.name, edge.end.name, edge.edge_type)] = (
        count_dict[(edge.start.name, edge.end.name, edge.edge_type)][0] + 1, edge)

def filter_edges_by_count(count_dict, threshold):
    final_edges = []
    for k, v in count_dict.items():
        if v[0] >=threshold:
            final_edges.append(v[1])
    return final_edges

def get_nlp_from_scene_graphs(scene_graphs, tracking_model):
    nlp_descriptions = []

    visible_objects = get_visible_objects(scene_graphs)

    object_movements_nl, object_movements_lst = get_object_movements_from_scene_graphs(scene_graphs, tracking_model,
                                                                                       visible_objects)

    max_object_movements = get_max_object_movement_from_scene_graphs(scene_graphs, object_movements_lst)
    max_object_movements = {key: value for key, value in max_object_movements.items() if key != "robot"}
    max_overall_movement_idx = np.linalg.norm(np.array(list(max_object_movements.values())), axis=1).argmax()
    max_overall_movement = list(max_object_movements.values())[max_overall_movement_idx]


    sg_edge_changes = get_scene_graph_edge_changes(scene_graphs)
    
    objects_in_gripper = [sg.robot_interacted_object for sg in scene_graphs]

    use_object_distances = False
    if use_object_distances: 
        object_distances = get_gripper_object_distances_from_scene_graphs(scene_graphs, visible_objects)
    else:
        object_distances = None

    for scene_graph in scene_graphs:
        nlp_descriptions.append(scene_graph.get_nl_description())

    nlp_descriptions = {"object_movements": object_movements_nl, "object_distances": object_distances,
                        "scene_description": nlp_descriptions, "sg_edge_changes": sg_edge_changes[1] if sg_edge_changes is not None else None,
                        "objects_in_gripper": objects_in_gripper}

    return nlp_descriptions



def denoise_scene_graphs(scene_graphs, avg_interval=9):
    denoised_sg = copy.deepcopy(scene_graphs)
    assert avg_interval % 2 == 1, "Average interval must be odd"

    left_right_offset = avg_interval // 2
    for idx in range(len(scene_graphs)):

        if idx < left_right_offset:
            window = scene_graphs[0:idx + left_right_offset + 1]
        elif idx >= len(scene_graphs) - left_right_offset:
            window = scene_graphs[idx - left_right_offset:]
        else:
            window = scene_graphs[idx - left_right_offset:idx + left_right_offset + 1]

        averaged_edges = average_scene_graph_edges(window, len(window) // 2)

        denoised_sg[idx].edges = averaged_edges

    return denoised_sg




