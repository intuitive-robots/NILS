from copy import copy, deepcopy

import cv2
import networkx as nx
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.stats import zscore
from torchvision.ops import masks_to_boxes

import open3d as o3d

from nils.pointcloud.pointcloud import (
    CLOSE_DISTANCE,
    DEPTH_THRESH,
    IN_CONTACT_DISTANCE,
    INSIDE_THRESH,
    IS_NOISE_THRESH,
    NORM_THRESH_FRONT_BACK,
    NORM_THRESH_LEFT_RIGHT,
    NORM_THRESH_UP_DOWN,
    OCCLUDE_RATIO_THRESH,
    ON_TOP_OF_THRESH,
    apply_mask_to_image,
    create_point_cloud_from_rgbd,
    get_obj_depth,
    get_obj_points,
)
from nils.pointcloud.pointcloud_utils import (
    canonicalize_point_cloud,
    get_pcd_dist,
    is_inside,
    is_inside_2d,
    is_merge,
)
from nils.scene_graph.scene_graph_utils import (
    get_bbox_center,
    get_bbox_from_segmentation,
    get_closest_point_on_mask,
    get_corner_points, get_extended_relations,
)
from nils.utils.visualizer import Visualizer

MOVEMENT_THRESHOLD = 0.03


class SceneDistGraph:
    def __init__(self, mask_areas=None, image_np=None, heuristics=None, pcd_transformation=None, depth_map=None,
                 gripper_pos=None):
        self.robot_interacted_object = None
        self.objects = {}

        self.nodes = []
        self.node_map = {}
        self.total_nodes = []
        self.edges = {}

        self.gripper_pos = gripper_pos

        self.robot_node = None

        self.heuristics = heuristics
        self.pcd_transformation = pcd_transformation

        self.flow_raw = None  # Flow from this frame to next frame
        self.object_movements = None  # Object movements from this frame to next frame
        self.prev_movement = None

        self.flow_mask = None
        self.flow_mask_bwd = None

        self.object_movements_bwd = None

        self.depth_map = None

        self.mask_areas = mask_areas
        self.scaling_factor = 200

        self.image_np = image_np

        self.gripper_cam_labels = None

    def get_objects_near_robot(self):
        robot_node = self.get_node("robot")

        for other_node in self.nodes:
            if other_node.name != "robot":
                if other_node.seg_mask is None and "region" not in other_node.name:
                    continue
                else:
                    dist = self.get_point_object_distance(robot_node.pos, other_node.seg_mask)
                    if dist is None:
                        continue
                    else:
                        if dist < 0.8:
                            return True

    def get_nl_description(self, max_movement_object=None, name_objects=False, visible_objects=None):
        output = ""
        visited = []
        if name_objects:
            for node in set(self.nodes):
                output += (node.get_name() + ", ")

        robot_holds = None
        if len(output) != 0:
            output = output[:-2] + ". "
        for edge_key, edge in self.edges.items():
            start_name, end_name = edge_key
            if end_name == "robot" and edge.edge_type == "inside":
                robot_holds = start_name
                continue
            edge_key_2 = (end_name, start_name)
            if visible_objects is not None and (start_name not in visible_objects or end_name not in visible_objects):
                continue
            if edge_key not in visited and edge_key_2 not in visited:
                if max_movement_object is not None:
                    if start_name == max_movement_object or end_name == max_movement_object:
                        output += edge.start.name + " is " + edge.edge_type + " " + edge.end.name
                        output += ". "
                else:
                    output += edge.start.name + " is " + edge.edge_type + " " + edge.end.name

                    output += ". "
            visited.append(edge_key)
        output = output[:-1]

        if robot_holds is not None:
            output += "The robot holds " + robot_holds + ". "

        return output

    def get_object_relations(self):
        output = ""
        visited = []
        for edge_key, edge in self.edges.items():
            start_name, end_name = edge_key
            edge_key_2 = (end_name, start_name)
            if edge_key not in visited and edge_key_2 not in visited:
                output += edge.start.name + " is " + edge.edge_type + " " + edge.end.name
                output += ". "
            visited.append(edge_key)
        output = output[:-1]

        return output

    def annotate_image(self, moved_object, filtered_objects):

        visualizer = Visualizer(self.image_np, metadata=None, scale=1.0)

        for node in self.nodes:
            cur_mask = node.seg_mask
            if node.name == moved_object:

                img_annotated = visualizer.draw_binary_mask_with_number(cur_mask, text=node.name, label_mode=1,
                                                                        alpha=0.2,
                                                                        anno_mode=["Mark", "Mask", "Box"], color="red")
            else:
                if node.name not in filtered_objects and node.name != "robot":
                    continue
                if node.visible:
                    img_annotated = visualizer.draw_binary_mask_with_number(cur_mask, text=node.name, label_mode=1,
                                                                            alpha=0.2,
                                                                            anno_mode=["Mark"])
        return img_annotated.get_image()

    def get_observed_objects(self):
        return [node.name for node in self.nodes]

    def remove_in_gripper_edges(self):
        for key in list(self.edges.keys()):
            if "robot" in key:
                del self.edges[key]

    def get_held_object(self):

        for key, edge in self.edges.items():
            if "robot" in key and key[0] != "nothing" and edge.edge_type == "inside":
                return key[0]
        return "nothing"

    def add_node_wo_edge(self, node):
        self.total_nodes.append(node)

    def get_node(self, obj):
        if obj not in self.node_map:
            return None
        return self.node_map[obj]

    def in_contact(self, obj, robot_pos):

        for node in self.nodes:
            if node.name == obj:
                node_mask = node.seg_mask
                if node_mask is None or not node.visible:
                    return False
                node_mask_dilated = cv2.dilate(node_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=4)
                if node_mask_dilated[robot_pos[1], robot_pos[0]] == 1:
                    return True
                else:
                    return False

    def add_node(self, new_node):
        merge = False
        for idx, node in enumerate(self.nodes):
            if new_node.name == node.name:
                merge, dist = is_merge(new_node, node)
                if merge:
                    if node.pcd is None:
                        node.seg_mask = np.logical_or(node.seg_mask, new_node.seg_mask)
                    else:
                        node.pcd = torch.cat((node.pcd, new_node.pcd), 0)
                    self.nodes[idx] = node
                    # self.add_object_state(new_node, self.event, mode="gt")
                    return new_node
        if not merge:
            # if new_node.name != self.in_robot_gripper(): # Check if in gripper?????
            if True:
                for node in self.total_nodes:
                    # if node.name != new_node.name and node.visible and new_node.visible:
                    if node.name != new_node.name:
                        # if new_node.obj.interactable() and new_node.obj.movable_to_container():
                        self.add_edge(node, new_node)
                        # if node.obj.interactable() and node.obj.movable_to_container():
                        self.add_edge(new_node, node)
                        # self.add_edge(new_node, node)
            self.nodes.append(new_node)
            self.node_map[new_node.name] = new_node
            # self.add_object_state(new_node, self.event, mode="gt")
        return new_node

    def add_object_in_gripper_edge(self, use_gripper_cam=False):

        gripper_object_dists = self.get_gripper_object_distances()

        for obj, dist in gripper_object_dists.items():
            if dist is not None and dist < INSIDE_THRESH and "region" not in obj:
                self.edges[(obj, "robot")] = Edge(self.get_node(obj), self.get_node("robot"), dist, "inside")

    def add_edge(self, node, new_node):

        if node.name == "robot" or new_node.name == "robot":
            return
        pos_1 = node.pos
        pos_2 = new_node.pos

        if node.type == "region":
            s = 1

        if pos_1 is None or pos_2 is None:
            return
        cam_arr = pos_1 - pos_2
        norm_vector = cam_arr / np.linalg.norm(cam_arr)

        box_A, box_B = np.array(node.corner_pts), np.array(new_node.corner_pts)
        box_A_unscaled, box_B_unscaled = np.array(node.corner_pts), np.array(new_node.corner_pts)

        if node.seg_mask is None:
            seg_mask_a = None
            seg_mask_a_dilated = None
        else:
            seg_mask_a = node.seg_mask
            seg_mask_a_dilated = cv2.dilate(seg_mask_a.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

        if new_node.seg_mask is None:
            seg_mask_b = None
            seg_mask_b_dilated = None
        else:

            seg_mask_b = new_node.seg_mask
            seg_mask_b_dilated = cv2.dilate(seg_mask_b.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

        # scale to (0,1)
        scale = np.array(
            [node.seg_mask.shape[1], node.seg_mask.shape[0], node.seg_mask.shape[1], node.seg_mask.shape[0]])

        is_3d = node.pcd is not None and new_node.pcd is not None

        if not is_3d:

            box_A = box_A / scale
            box_B = box_B / scale

            scale_xy = np.array([node.seg_mask.shape[1], node.seg_mask.shape[0]])

            if node.name == "robot":
                dist = self.get_point_object_distance(node.pos, new_node.seg_mask)
            else:
                is_touching = np.any(np.logical_and(seg_mask_a_dilated, seg_mask_b_dilated))
                if is_touching:
                    dist = 0.0
                else:
                    dist = self.get_point_object_distance(node.pos, new_node.seg_mask, scale=scale_xy)

            # transform y,x to x,y
            box_A_pts, box_B_pts = np.array(np.where(node.seg_mask))[::-1], np.array(np.where(new_node.seg_mask))[::-1]

            box_A_pts, box_B_pts = box_A_pts / np.array([node.seg_mask.shape[1], node.seg_mask.shape[0]])[
                                               :, None], box_B_pts / np.array(
                [new_node.seg_mask.shape[1], new_node.seg_mask.shape[0]])[:, None]

            if dist is None:
                return
            # dist = dist / node.seg_mask.shape[1]
        else:

            if len(node.pcd.points) == 0 or len(new_node.pcd.points) == 0:
                return

            added_node = self.add_edges_3d(node, new_node)
            if added_node:
                return

        # IN CONTACT

        if node.seg_mask.sum() == 0 or new_node.seg_mask.sum() == 0:
            return

        box_A = masks_to_boxes(torch.tensor(node.seg_mask[None, :, :]))[0].numpy()
        box_B = masks_to_boxes(torch.tensor(new_node.seg_mask[None, :, :]))[0].numpy()

        box_B_unscaled = box_B
        box_A_unscaled = box_A
        pos_1 = get_bbox_center(box_A)
        pos_2 = get_bbox_center(box_B)

        cam_arr = pos_1 - pos_2
        norm_vector = cam_arr / np.linalg.norm(cam_arr)

        box_A = box_A / scale
        box_B = box_B / scale
        scale_xy = np.array([node.seg_mask.shape[1], node.seg_mask.shape[0]])

        # get pos by box center

        is_touching = np.any(np.logical_and(seg_mask_a_dilated, seg_mask_b_dilated))
        if self.depth_map is not None:
            max_depth = self.depth_map.max()
            depth_A = (node.seg_mask * self.depth_map)
            depth_A = depth_A[depth_A != 0].mean()

            depth_B = (new_node.seg_mask * self.depth_map)
            depth_B = depth_B[depth_B != 0].mean()

            is_touching = is_touching and ((depth_A / max_depth - depth_B / max_depth) < 0.1)
        if is_touching:
            dist = 0.0
        else:
            dist = self.get_point_object_distance(pos_1, new_node.seg_mask, scale=scale_xy)

        scale_xy = np.array([node.seg_mask.shape[1], node.seg_mask.shape[0]])

        if node.name == "robot":
            dist = self.get_point_object_distance(node.pos, new_node.seg_mask)

        # transform y,x to x,y
        box_A_pts, box_B_pts = np.array(np.where(node.seg_mask))[::-1], np.array(np.where(new_node.seg_mask))[::-1]

        box_A_pts, box_B_pts = box_A_pts / np.array([node.seg_mask.shape[1], node.seg_mask.shape[0]])[
                                           :, None], box_B_pts / np.array(
            [new_node.seg_mask.shape[1], new_node.seg_mask.shape[0]])[:, None]

        box_A_pts = box_A_pts.T
        box_B_pts = box_B_pts.T

        if dist is None:
            return

        if dist < IN_CONTACT_DISTANCE:

            # if is_inside(src_pts=box_B_pts, target_pts=box_A_pts, thresh=INSIDE_THRESH):

            # added_node  = self.add_edges_3d(node,new_node)
            if is_inside_2d(box_B, box_A, thresh=0.85):
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "inside")

            # elif box_A_pts.ndim == 2:
            #
            #     box_A_x, box_A_y = box_A_pts[:, 0], box_A_pts[:, 1]
            #     box_B_x, box_B_y = box_B_pts[:, 0], box_B_pts[:, 1]
            #
            #     intersection_area = get_bbox_intersection_area(box_B_unscaled,box_A_unscaled)
            #
            #     count_above_b_a = np.sum((box_B_y < box_A[1]) & (box_A[0] <= box_B_x) & (box_B_x <= box_A[2]))
            #     count_above_a_b = np.sum((box_A_y < box_B[1]) & (box_B[0] <= box_A_x) & (box_A_x <= box_B[2]))
            #
            #
            #
            #
            #     #count_above_b_a = sum(1 for x, y in box_B_pts if y < box_A[3] and (x >= box_A[0] and x <= box_A[2])) #less because of np indexing
            #     #count_above_a_b = sum(1 for x, y in box_A_pts if y < box_A[3] and (x >= box_B[0] and x <= box_B[2]))
            #     if count_above_b_a  + 0.4 * intersection_area>  len(box_B_pts) * ON_TOP_OF_THRESH and "region" not in node.name:
            #         self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "above")
            #     elif count_above_a_b  +  0.4 * intersection_area>  len(box_A_pts) * ON_TOP_OF_THRESH and "region" not in new_node.name:
            #         self.edges[(node.name, new_node.name)] = Edge(node, new_node, dist, "above")
            #
            # elif len(np.where((box_B_pts[:, 0] < box_A[2, 2]) & (box_B_pts[:, 0] > box_A[0, 0]) &
            #                   (box_B_pts[:, 2] < box_A[4, 2]) & (box_B_pts[:, 2] > box_A[0, 2]))[0]) > len(
            #     box_B_pts) * ON_TOP_OF_THRESH:
            #     if len(np.where(box_B_pts[:, 1] > box_A[4, 1])[0]) > len(box_B_pts) * ON_TOP_OF_THRESH:
            #
            #         self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "on top of")
            #     elif len(np.where(box_A_pts[:, 1] > box_B[4, 1])[0]) > len(box_A_pts) * ON_TOP_OF_THRESH:
            #
            #         self.edges[(node.name, new_node.name)] = Edge(node, new_node, dist, "on top of")

            if np.any(np.logical_and(seg_mask_a_dilated, seg_mask_b_dilated)):
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "touching")

        # CLOSE TO
        if dist < CLOSE_DISTANCE and (new_node.name, node.name) not in self.edges and (not new_node.global_node):

            if abs(norm_vector[1]) > NORM_THRESH_UP_DOWN:
                if norm_vector[1] > 0:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "behind of")
                else:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "in front of")
            if abs(norm_vector[0]) > NORM_THRESH_LEFT_RIGHT:
                if norm_vector[0] < 0:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "on the right of")
                else:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "on the left of")

            # derive blocking from previous mask size (e.g. some kind of average of mask area over timespan)
            # Check if one object has a certain prop of
            elif new_node.name == "robot" or node.name == "robot":
                return
            elif new_node.name not in self.mask_areas or node.name not in self.mask_areas:
                return
            # elif ( node.seg_mask.sum() <  self.mask_areas[node.name]* OCCLUDE_RATIO_THRESH and new_node.seg_mask.sum() > self.mask_areas[new_node.name]* (0.8)):
            #     self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "blocking")
            # elif (node.seg_mask.sum() >  self.mask_areas[node.name]* (0.8) and new_node.seg_mask.sum() < self.mask_areas[new_node.name]* OCCLUDE_RATIO_THRESH):
            #     self.edges[(node.name, new_node.name)] = Edge(node, new_node, dist, "blocking")

            # elif abs(norm_vector[
            #              2]) > NORM_THRESH_FRONT_BACK and new_node.bbox2d is not None and node.bbox2d is not None and new_node.depth is not None and node.depth is not None:
            #     iou, inters = get_iou_bbox(new_node.bbox2d, node.bbox2d)
            #     occlude_ratio = inters / ((node.bbox2d[2] - node.bbox2d[0]) * (node.bbox2d[3] - node.bbox2d[1]))
            #     # print("new_node, node: ", new_node.name, node.name)
            #     # print("iou, occlude_ratio: ", iou, occlude_ratio)
            #     if occlude_ratio > OCCLUDE_RATIO_THRESH and len(
            #             np.where(new_node.depth <= np.min(node.depth))[0]) > len(new_node.depth) * DEPTH_THRESH:
            #         self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "blocking")

    def add_edges_3d(self, node, new_node):

        IN_CONTACT_DISTANCE = self.heuristics["IN_CONTACT_DISTANCE_3D"]
        CLOSE_DISTANCE = self.heuristics["CLOSE_DISTANCE_3D"]

        pos_1 = node.pos
        pos_2 = new_node.pos

        points1 = np.asarray(node.pcd.points)
        points2 = np.asarray(new_node.pcd.points)


        pcd2_tree = o3d.geometry.KDTreeFlann(new_node.pcd)
        min_distance = float('inf')
        closest_point1 = None
        closest_point2 = None

        # Search for the closest points using KD-Tree
        for point1 in points1:
            [_, idx, _] = pcd2_tree.search_knn_vector_3d(point1, 1)
            point2 = points2[idx[0]]
            distance = np.linalg.norm(point1 - point2)
            if distance < min_distance:
                min_distance = distance
                closest_point1 = point1
                closest_point2 = point2

        norm_vector_closest = closest_point2 - closest_point1
        norm_vector_closest = norm_vector_closest / np.linalg.norm(norm_vector_closest)

        box_A_pts, box_B_pts = np.array(node.pcd.points), np.array(new_node.pcd.points)
        box_A, box_B = np.array(node.corner_pts), np.array(new_node.corner_pts)
        cam_arr = pos_2 - pos_1
        norm_vector = cam_arr / np.linalg.norm(cam_arr)

        norm_vector_combined = norm_vector_closest + cam_arr
        norm_vector_combined = norm_vector_combined / np.linalg.norm(norm_vector_combined)

        mean_dist, dist = get_pcd_dist(node.pcd, new_node.pcd)
        added_node = False
        if dist < IN_CONTACT_DISTANCE:

            if is_inside(src_pts=box_B_pts, target_pts=box_A_pts, thresh=0.3):

                self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "inside")
                added_node = True
            elif len(np.where((box_B_pts[:, 0] < box_A[4, 0]) & (box_B_pts[:, 0] > box_A[0, 0]) &
                              (box_B_pts[:, 2] < box_A[4, 2]) & (box_B_pts[:, 2] > box_A[0, 2]))[0]) > len(
                box_B_pts) * ON_TOP_OF_THRESH:
                if len(np.where(box_B_pts[:, 1] > box_A_pts[:, 1].mean())[0]) > len(box_B_pts) * ON_TOP_OF_THRESH:

                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "on top of")
                    added_node = True
                elif len(np.where(box_A_pts[:, 1] > box_B_pts[:, 1].mean())[0]) > len(box_A_pts) * ON_TOP_OF_THRESH:

                    self.edges[(node.name, new_node.name)] = Edge(node, new_node, dist, "on top of")
                    added_node = True

        # CLOSE TO
        # if dist < CLOSE_DISTANCE and (new_node.name, node.name) not in self.edges and (not new_node.global_node):
        if dist < CLOSE_DISTANCE and (not new_node.global_node) and not added_node:

            if abs(norm_vector[1]) > NORM_THRESH_UP_DOWN:
                if norm_vector[1] > 0:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "above")
                    added_node = True
                else:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "below")
                    added_node = True
            if abs(norm_vector[0]) > NORM_THRESH_LEFT_RIGHT:
                if norm_vector[0] > 0:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "on the right of")
                    added_node = True
                else:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "on the left of")
                    added_node = True

            if abs(norm_vector[2] > NORM_THRESH_FRONT_BACK):
                if norm_vector[2] > 0:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "in front of")
                    added_node = True
                if norm_vector[2] < 0:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "behind of")
                    added_node = True

            if not added_node:
                relation = get_extended_relations(norm_vector)

                if relation is not None:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, relation)
                    added_node = True
                else:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, dist, "close to")
                    added_node = True

        return added_node

    def __eq__(self, other):

        if not ((set(self.nodes) == set(other.nodes)) and (set(self.edges.values()) == set(other.edges.values()))):

            diff = set(self.edges.values()).symmetric_difference(set(other.edges.values()))

            diff_nodes = set(self.nodes).symmetric_difference(set(other.nodes))

            if len(diff_nodes) > 0:
                return False
            actual_diff = True

            non_movable_change = False

            for d in diff:

                if d.start.obj.movable_to_container:
                    return False
            return True
        else:
            return True

        # remove nodes that were not visibile in this or other scene graph
        not_visible_nodes = [node for node in self.nodes + other.nodes if not node.visible]

        nodes_to_compare_self = [node for node in self.nodes if node not in not_visible_nodes]
        nodes_to_compare_other = [node for node in other.nodes if node not in not_visible_nodes]

        # remove edges that contain nodes that were not visible in this or other scene graph
        not_visible_nodes_names = [node.name for node in not_visible_nodes]
        edges_to_compare_self = {edge_key: edge for edge_key, edge in self.edges.items() if
                                 edge.start.name not in not_visible_nodes_names and edge.end.name not in not_visible_nodes_names}
        edges_to_compare_other = {edge_key: edge for edge_key, edge in other.edges.items() if
                                  edge.start.name not in not_visible_nodes_names and edge.end.name not in not_visible_nodes_names}

        if not ((set(nodes_to_compare_self) == set(nodes_to_compare_other)) and (
                set(edges_to_compare_self.values()) == set(edges_to_compare_other.values()))):

            diff = set(edges_to_compare_self.values()).symmetric_difference(set(edges_to_compare_other.values()))
            actual_diff = True
            for d in diff:

                if np.linalg.norm(self.prev_movement[d.start.name]) > 1 or d.edge_type == "inside":
                    return False
            return True
        else:
            return True

    def __str__(self):
        visited = []
        res = ""
        # res += "Object Relations:\n"
        for edge_key, edge in self.edges.items():
            name_1, name_2 = edge_key
            edge_key_reversed = (name_2, name_1)
            if (edge_key not in visited and edge_key_reversed not in visited) or edge.edge_type in ['on top of',
                                                                                                    'inside',
                                                                                                    'occluding']:
                res += edge.get_string()
                res += "\n"
            visited.append(edge_key)
        return res

    def is_occluded(self, node):

        # node_names = np.array([node.name for node in self.nodes])
        # robot_idx = np.where(node_names == "robot")[0].item()
        # robot_node = self.nodes[robot_idx]

        # robot_obj_dist = np.linalg.norm(robot_node.pos - node.pos)

        # if robot_obj_dist < 30:
        #    return False

        if node.seg_mask.sum() < self.mask_areas[node.name] * OCCLUDE_RATIO_THRESH:
            return True
        else:
            return False

    def is_noise(self, node):

        if node.seg_mask.sum() > self.mask_areas[node.name] * IS_NOISE_THRESH:
            return True
        else:
            return False

    def get_object_positions(self):

        positions = {}
        for node in self.nodes:
            positions[node.name] = node.pos
        return positions

    def get_object_movements(self, next_scene_graph, use_flow_always=True):

        object_movements = {}

        next_scene_graph_names = np.array([node.name for node in next_scene_graph.nodes])

        if self.flow_raw is None:
            raise ValueError("Flow not calculated for this scene graph")
            flow_input = np.stack([self.image_np, next_scene_graph.image_np], axis=0)
            flow_masks_fwd, flow_masks_bwd, flow_fwd, flow_bwd = flow_predictor.calc_optical_flow(flow_input, 1)
            self.flow_mask = flow_masks_fwd
            self.flow_mask_bwd = flow_masks_bwd
            self.flow_raw = flow_fwd

        if self.object_movements is not None and next_scene_graph.prev_movement is not None:
            return self.object_movements

        for node in self.nodes:
            if node in next_scene_graph.nodes:

                # Init scene graph
                if node.seg_mask is None or node.seg_mask.sum() == 0 or "region" in node.name:
                    object_movements[node.name] = np.array([0., 0.])
                    continue

                next_node = next_scene_graph.nodes[np.where(next_scene_graph_names == node.name)[0].item()]

                if use_flow_always or (self.is_occluded(next_node) and not self.is_noise(next_node)):

                    flow = self.flow_raw[0, node.seg_mask.astype(np.bool), :]

                    z_scores = np.abs(zscore(flow, axis=0))
                    outlier_mask = np.any(z_scores > 2, axis=1)
                    filtered_vectors = flow[~outlier_mask]

                    mean_dir = np.mean(filtered_vectors, axis=0)

                    object_movements[node.name] = mean_dir
                else:
                    if node.pos is not None and next_node.pos is not None:
                        object_movements[node.name] = next_node.pos - node.pos
                    else:
                        object_movements[node.name] = None

            else:
                object_movements[node.name] = None

        if self.object_movements is None:
            self.object_movements = object_movements
        if next_scene_graph.prev_movement is None:
            next_scene_graph.prev_movement = object_movements

        return object_movements

    def get_movement_direction(self, object_movement):

        movement_dict = {}
        if object_movement is not None:
            if object_movement[0] > MOVEMENT_THRESHOLD * self.scaling_factor:
                movement_dict["right"] = int(object_movement[0])
            elif object_movement[0] < -MOVEMENT_THRESHOLD * self.scaling_factor:
                movement_dict["left"] = int(object_movement[0])
            if object_movement[1] <= -MOVEMENT_THRESHOLD * self.scaling_factor:
                movement_dict["up"] = int(object_movement[1])
            elif object_movement[1] > MOVEMENT_THRESHOLD * self.scaling_factor:
                movement_dict["down"] = int(object_movement[1])

        return movement_dict

    # def get_nl_movements(self, object_movements):
    #
    #     output ="Object Movements:\n {"
    #     #object_movements = self.get_object_movements(next_scene_graph)
    #     for obj, movement in object_movements.items():
    #         if movement is not None:
    #             if movement[0] > MOVEMENT_THRESHOLD * self.scaling_factor:
    #                 output += obj + f" moved right by {int(movement[0])}. "
    #             elif movement[0] < -MOVEMENT_THRESHOLD * self.scaling_factor:
    #                 output += obj + f" moved left by {int(movement[0])}. "
    #             if movement[1] <= -MOVEMENT_THRESHOLD * self.scaling_factor: ##Reversed,as indexing is reversed in
    #                 output += obj + f" moved up by {int(movement[1])}. "
    #             elif movement[1] > MOVEMENT_THRESHOLD * self.scaling_factor:
    #                 output += obj + f" moved down by {int(movement[1])}. "
    #     return output
    def get_nl_movements(self, object_movements, visible_objects=None):

        output = "Object Movements:\n {"
        # object_movements = self.get_object_movements(next_scene_graph)
        for obj, movement in object_movements.items():
            if movement is not None:
                if movement[0] > MOVEMENT_THRESHOLD * self.scaling_factor:
                    output += obj + f"(right,{int(movement[0])}),"
                elif movement[0] < -MOVEMENT_THRESHOLD * self.scaling_factor:
                    output += obj + f"(left,{int(movement[1])}),"
                if movement[1] <= -MOVEMENT_THRESHOLD * self.scaling_factor:  ##Reversed,as indexing is reversed in
                    output += obj + f"(up,{int(movement[0])}),"
                elif movement[1] > MOVEMENT_THRESHOLD * self.scaling_factor:
                    output += obj + f"(down,{int(movement[1])}),"
        output += "}"
        return output

    def get_nl_gripper_object_dists(self, gripper_object_distances, visible_objects=None):
        output = "Distances to Robot Gripper:\n{"
        for obj, dist in gripper_object_distances.items():
            if dist is not None and "region" not in obj and (visible_objects is None or obj in visible_objects):
                output += obj + f": {round(dist, 1)},"
        output += "}"
        return output

    def get_point_object_distance(self, point, mask, scale=None):
        if mask is None:
            return None
        other_node_points = np.array(np.where(mask))[::-1]

        vec = other_node_points - point[:, None]
        if scale is not None:
            vec = vec / scale[:, None]
        point_mask_dists = np.linalg.norm(vec, axis=0)
        if len(point_mask_dists) == 0:
            return None
        return np.min(point_mask_dists)

    def get_gripper_object_distances(self, round_digits=1):
        gripper_object_distances = {}
        robot_node = self.get_node("robot")

        if robot_node is None:
            return gripper_object_distances
        for other_node in self.nodes:
            if other_node.name != "robot":
                if other_node.seg_mask is None:
                    gripper_object_distances[other_node.name] = None
                else:
                    if other_node.pcd is not None:
                        dist = other_node.pcd.compute_point_cloud_distance(robot_node.pcd)
                    dist = self.get_point_object_distance(robot_node.pos, other_node.seg_mask)
                    if dist is None:
                        gripper_object_distances[other_node.name] = None
                    else:
                        gripper_object_distances[other_node.name] = round(dist, round_digits)

        return gripper_object_distances

    def plot_center_points_on_img(self, image):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)

        for node in self.nodes:
            if node.pos is not None:
                x = node.pos[0]
                y = node.pos[1]

                circle = patches.Circle((x, y), radius=2, color='red')
                ax.add_patch(circle)
                ax.text(x, y, node.name, fontsize=10, color='red')
        plt.show()

    def plot_graph(self):
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.name)

        # Add edges to the graph
        for edge in self.edges.values():
            G.add_edge(edge.start.name, edge.end.name, weight=round(edge.dist, 3), edge_type=edge.edge_type)

        # Draw the graph
        # pos = nx.spring_layout(G)  # positions for all nodes

        pos = {node.name: node.pos / node.seg_mask.shape[0] for node in self.nodes}

        for obj_pos in pos.values():
            obj_pos[1] = 1 - obj_pos[1]

        labels = nx.get_edge_attributes(G, 'weight')

        relation = nx.get_edge_attributes(G, 'edge_type')

        merged_labels = {k: str(v) + " - " + str(relation[k]) for k, v in labels.items()}
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=100, edge_color='black', linewidths=1,
                font_size=15, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=merged_labels, font_size=5)

        plt.title("Scene Dist Graph Visualization")
        plt.show()

    def get_graph_obj_movement_image(self, obj, movement, plot=False):
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.name)

        # Add edges to the graph
        for edge in self.edges.values():
            G.add_edge(edge.start.name, edge.end.name, weight=round(edge.dist, 3), edge_type=edge.edge_type)

        # Draw the graph
        # pos = nx.spring_layout(G)  # positions for all nodes

        pos = {node.name: node.pos / node.seg_mask.shape[0] for node in self.nodes}

        for obj_pos in pos.values():
            obj_pos[1] = 1 - obj_pos[1]

        obj_start_pos = pos[obj]

        labels = nx.get_edge_attributes(G, 'weight')

        relation = nx.get_edge_attributes(G, 'edge_type')

        merged_labels = {k: str(v) + " - " + str(relation[k]) for k, v in labels.items()}

        fig, ax = plt.subplots()

        canvas = fig.canvas

        nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=100, edge_color='black', linewidths=1,
                font_size=15, arrows=True)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=merged_labels, font_size=5)

        ax.arrow(obj_start_pos[0], obj_start_pos[1], movement[0], movement[1], head_width=0.1, head_length=0.1,
                 fc='red', ec='red')

        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)

        if plot:
            plt.imshow(image)
            plt.show()

        return image


class CustomNode(object):
    def __init__(self, name, object_id=None, pos=None, corner_pts=None, bbox2d=None, pcd=None, depth=None,
                 global_node=False, seg_mask=None, score=None, type="object", visible=True, obj=None, state=None):
        self.name = name
        self.object_id = object_id
        self.bbox2d = bbox2d  # 2d bounding box (4x1)
        self.pos = pos  # object position, 2d or 3d
        self.corner_pts = corner_pts  # corner points of 3d bbox (8x3)
        self.pcd = pcd  # point cloud (px3)
        self.depth = depth
        self.name_w_state = None
        self.state = state
        self.global_node = global_node
        self.seg_mask = seg_mask
        self.score = score
        self.type = type
        self.visible = visible
        self.obj = obj

    def set_state(self, state):
        self.name_w_state = state

    def __str__(self):
        return self.get_name()

    def __hash__(self):
        return hash(self.get_name())

    def __eq__(self, other):

        return True if self.get_name() == other.get_name() and self.state == other.state else False

    def get_name(self):
        if self.name_w_state is not None:
            return self.name_w_state
        else:
            return self.name


class Edge(object):
    def __init__(self, start_node, end_node, dist, edge_type="none"):
        self.start = start_node
        self.end = end_node
        self.edge_type = edge_type
        self.dist = dist

    def __hash__(self):
        return hash((self.start, self.end, self.edge_type))

    def __eq__(self, other):
        if self.start == other.start and self.end == other.end and self.edge_type == other.edge_type:
            return True
        else:
            return False

    def __str__(self):
        return str(self.start) + "->" + self.edge_type + "->" + str(self.end)

    def get_string(self):

        res_string = ""
        if self.edge_type != "inside" and self.edge_type != "on top of":
            # return str(self.start) +  " is next to " + self.edge_type + " " + str(self.end)

            rel_str = str(self.start) + " " + self.edge_type + " " + str(self.end)

            return rel_str

        return str(self.start) + " is " + self.edge_type + " " + str(self.end)


def init_scene_dist_graph(objects, image_np, obj_mask_areas):
    init_dist_graph = SceneDistGraph(obj_mask_areas, image_np=image_np)

    for object in objects:
        node = CustomNode(object, pos=None, corner_pts=None, bbox2d=None, pcd=None, depth=None, seg_mask=None,
                          global_node=False, visible=False)
        init_dist_graph.add_node_wo_edge(node)

    for node in init_dist_graph.total_nodes:
        init_dist_graph.add_node(node)

    return init_dist_graph





def get_pc_transforms(image, depth_image):
    width = image.shape[1]
    height = image.shape[0]

    intrinsic_parameters = {
        'width': width,
        'height': height,
        'fx': 612,
        'fy': 661,
        'cx': 320,
        'cy': 117,
    }

    pcd_orig = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.array(image)),
            o3d.geometry.Image(np.array(depth_image)),
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        ),
        o3d.camera.PinholeCameraIntrinsic(**intrinsic_parameters)
    )
    # pcd_orig = pcd_orig.voxel_down_sample(voxel_size=0.005)
    # pcd_orig  = pcd_orig.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)[0]

    plane_model, inliers = pcd_orig.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=4000)

    inlier_cloud = pcd_orig.select_by_index(inliers)
    pcd_components, canonicalized, transformation = canonicalize_point_cloud(inlier_cloud)

    if not canonicalized:
        return None

    return transformation


def create_scene_dist_graph(obj_seg_mask_dict, obj_mask_areas, image_np, flow_prediction, cfg, gripper_pos=None,
                            tcp_center_screen=None, gt_depth=None,
                            center_point_type="bbox", last_scene_graph=None, transformation=None,
                            intrinsic_parameters=None,
                            ):
    object_point_clouds = {}
    heuristics = None

    if gt_depth is not None and cfg.use_depth:
        if transformation is None:
            transformation = get_pc_transforms(image_np, gt_depth)
            if transformation is None:
                if last_scene_graph is not None and last_scene_graph.pcd_transformation is not None:
                    transformation = last_scene_graph.pcd_transformation
                else:
                    return False
        width = image_np.shape[1]
        height = image_np.shape[0]

        if intrinsic_parameters is None:
            intrinsic_parameters = {
                'width': width,
                'height': height,
                'fx': 612,
                'fy': 661,
                'cx': 320,
                'cy': 117,
            }
        rotation_z_180 = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                                   [np.sin(np.pi), np.cos(np.pi), 0],
                                   [0, 0, 1]])

        whole_pcd = create_point_cloud_from_rgbd(np.array(image_np), np.array(gt_depth), intrinsic_parameters)
        whole_pcd.transform(transformation)
        # whole_pcd.rotate(rotation_z_180, center=(0, 0, 0))
        max_extend = np.linalg.norm(whole_pcd.get_max_bound() - whole_pcd.get_min_bound())

        heuristics = {
            "IN_CONTACT_DISTANCE_3D": max_extend / 200.,
            "CLOSE_DISTANCE_3D": max_extend / 30.,

        }

    gripper_location_3d = None

    for i, (obj, obj_dict) in enumerate(obj_seg_mask_dict.items()):

        if gt_depth is not None:

            mask = obj_dict["mask"]

            mask_binary = mask > 0

            masked_rgb = apply_mask_to_image(np.array(image_np), mask_binary)
            masked_depth = apply_mask_to_image(np.array(gt_depth), mask_binary)

            pcd = create_point_cloud_from_rgbd(masked_rgb, masked_depth, intrinsic_parameters)

            scale = np.linalg.norm(np.array(pcd.points).std(axis=0)) * 3.0 + 1e-6
            pcd = pcd.voxel_down_sample(voxel_size=max(0.001, scale / 20))
            # print(pcd)
            if np.array(pcd.points).shape[0] > 62:
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=60, std_ratio=1.0)
                inlier_cloud = pcd.select_by_index(ind)
            else:
                inlier_cloud = pcd

            inlier_cloud.transform(transformation)
            if obj == "robot_location":
                gripper_location_3d = inlier_cloud.get_center()


            else:
                object_point_clouds[obj] = inlier_cloud
        obj_seg_map = object_point_clouds
    else:
        obj_seg_map = obj_seg_mask_dict

    if "robot_location" in obj_seg_map:
        del obj_seg_map["robot_location"]

    scene_dist_graph = SceneDistGraph(obj_mask_areas, image_np=image_np, heuristics=heuristics,
                                      pcd_transformation=transformation,
                                      depth_map=gt_depth, gripper_pos=gripper_location_3d)

    obj_centers = {}

    for obj, obj_dict in obj_seg_map.items():

        if obj in object_point_clouds:
            pcd = object_point_clouds[obj]
        else:
            pcd = None

        mask = obj_dict["mask"]
        score = obj_dict["score"]
        if obj_dict["obj"].is_region():
            bbox = obj_dict["box"]

            if pcd is not None:

                if obj == "robot" and gripper_location_3d is not None:
                    center_point = gripper_location_3d
                    corner_pts = get_corner_points(bbox)
                else:
                    bbox_3d = pcd.get_axis_aligned_bounding_box()
                    center_point = bbox_3d.get_center()
                    corner_pts = np.array(bbox_3d.get_box_points())
            else:
                center_point = get_bbox_center(bbox)
                corner_pts = get_corner_points(bbox)

            node = CustomNode(obj,
                              pos=center_point,
                              corner_pts=corner_pts,
                              bbox2d=bbox,
                              pcd=pcd if pcd is not None else None,
                              depth=None,
                              state=None,  # obj_dict["state"],
                              seg_mask=mask,
                              score=None,
                              global_node=True,
                              type="region",
                              obj=obj_dict["obj"])

            scene_dist_graph.add_node_wo_edge(node)

            obj_centers[obj] = center_point
        else:
            if mask.sum() == 0 or np.isnan(score):
                if last_scene_graph is not None:
                    last_node = last_scene_graph.get_node(obj)
                    if last_node is not None:
                        node = copy(last_node)
                        # node.seg_mask = mask
                        node.visible = False
                        scene_dist_graph.add_node_wo_edge(node)
                        continue
                    else:
                        continue
                else:
                    continue

            if "box" not in obj_dict:
                bbox_2d = get_bbox_from_segmentation(mask)
            else:
                bbox_2d = obj_dict["box"]
            if pcd is not None:
                bbox_3d = pcd.get_axis_aligned_bounding_box()
                pos = np.array(pcd.points).mean(axis=0)
                corner_pts = np.array(bbox_3d.get_box_points())
                obj_centers[obj] = pos
            else:
                pos = get_bbox_center(bbox_2d)
                closest_mask_point = get_closest_point_on_mask(mask, pos)

                if center_point_type == "mask":
                    center_point = closest_mask_point
                elif center_point_type == "bbox":
                    center_point = pos
                else:
                    raise ValueError("Invalid center point type")

                pos = center_point
                corner_pts = get_corner_points(bbox_2d)

                obj_centers[obj] = pos

            if tcp_center_screen is not None and obj == "robot":
                center_point = tcp_center_screen

                corner_pts = get_corner_points(bbox_2d)
            node = CustomNode(obj,
                              pos=pos,
                              corner_pts=corner_pts,
                              bbox2d=bbox_2d,
                              pcd=pcd if pcd is not None else None,
                              depth=gt_depth,
                              seg_mask=mask,
                              state=None,  # obj_dict["state"],
                              score=score,
                              type="object",

                              visible=True,
                              obj=obj_dict["obj"]
                              )

            scene_dist_graph.add_node_wo_edge(node)
    # if regions is not None:
    #     for region,bbox in regions.items():
    #         center_point = get_bbox_center(bbox)
    #         corner_pts = get_corner_points(bbox)
    # 
    #         seg_mask = np.zeros((image_np.shape[0],image_np.shape[1]))
    # 
    #         #fill bbox with ones
    #         seg_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
    # 
    #         node = CustomNode(region,
    #                     pos=center_point,
    #                     corner_pts=corner_pts,
    #                     bbox2d=bbox,
    #                     pcd=None,
    #                     depth=None,
    #                     seg_mask=seg_mask,
    #                     score=None,
    #                     global_node=True,
    #                     type ="region")
    # 
    #         scene_dist_graph.add_node_wo_edge(node)

    # for obj, obj_dict in obj_seg_map.items():
    #     for node in scene_dist_graph.total_nodes:
    #         if node.name == obj:
    #                 scene_dist_graph.add_node(node)
    #

    for node in scene_dist_graph.total_nodes:
        scene_dist_graph.add_node(node)

    if flow_prediction is not None:
        scene_dist_graph.flow_raw = flow_prediction[None]
    else:
        scene_dist_graph.flow_raw = None
    if last_scene_graph is not None:
        movement_dict = {obj: cur_obj_dict["movement"] for obj, cur_obj_dict in obj_seg_mask_dict.items()}
        # last_scene_graph.object_movements = movement_dict
        scene_dist_graph.object_movements = movement_dict
        scene_dist_graph.prev_movement = last_scene_graph.object_movements if last_scene_graph.object_movements is not None else movement_dict
        # last_scene_graph.get_object_movements(scene_dist_graph,
        #                                         cfg.use_flow_always)

    return scene_dist_graph, obj_centers
