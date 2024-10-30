import logging

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from sklearn.cluster import DBSCAN, HDBSCAN
from torchvision.ops import box_iou

from nils.utils.utils import filter_consecutive_trues_np


class Object:
    def __init__(self, name,states, movable_to_container, container, interactable, nl_name = None, color = None):

        self.states = states
        if not states:
            self.states = None

        self.movable_to_container = movable_to_container
        self.container = container
        self.interactable = interactable
        
        
        self.name = name
        
        self.synonyms = []
        
        self.nl_name = nl_name
        self.color = color
        
        self.predicted_states = []
        self.predicted_states_indices = []
        self.state_confidences = []
        
        self.cropped_images = []

        self.pos_3d = None
        
        
        self.boxes = None
        self.mask = None
        self.confidence = None
        self.object_movement = None
        self.object_movement_dirs = None
        
        self.mean_area = None
        
        
        self.detection_prompt = None
        self.ov_seg_prompt =None
        
        self.state_prompt = None
        
    
        
        
    
    def get_last_known_box(self, index):
        viable_indices = np.where((self.confidence != None) & (~np.isnan(self.confidence)))[0]
        if len(viable_indices) == 0:
            return None
        last_known_index = viable_indices[np.argmin(np.abs(viable_indices - index))]
        return self.boxes[last_known_index]
    
    def get_last_known_mask(self, index,range = -1):
        viable_indices = np.where((self.confidence != None) & (~np.isnan(self.confidence)))[0]
        if len(viable_indices) == 0:
            return None
        last_known_index = viable_indices[np.argmin(np.abs(viable_indices - index))]
        
        if range != -1:
            dist = np.abs(last_known_index - index)
            if dist > range:
                return None,None
            
        
        return self.mask[last_known_index], last_known_index
    
    
        
    def process(self):

        nan_filter = (self.confidence != None) & (~np.isnan(self.confidence))
        
        
        
        areas_to_compare = (self.boxes[nan_filter][:, 2] - self.boxes[nan_filter][:, 0]) * (self.boxes[nan_filter][:, 3] - self.boxes[nan_filter][:, 1])
        
        if self.mask is not None:
            mask_areas = np.sum(self.mask[nan_filter], axis=(1, 2))
            if mask_areas.sum() > 0:
                areas_to_compare = mask_areas
        confidences_to_compare = self.confidence[nan_filter]
        
        
        #Smaller more tolerance due to occlusion
        area_filter = (zscore(areas_to_compare) > 2) | (zscore(areas_to_compare) < -3.2)
        
        confidence_filter = (zscore(confidences_to_compare) > 2.5) | (zscore(confidences_to_compare) < -0.7)

        final_filter = area_filter | confidence_filter
        
        nan_filter[nan_filter] = final_filter

        if nan_filter.sum() > 0:
            
            self.boxes = self.boxes.astype(float)
            self.boxes[nan_filter] = np.zeros_like(self.boxes[0],dtype=int)
            if self.mask is not None:
                self.mask[nan_filter] = np.zeros_like(self.mask[0],dtype=bool)
            self.confidence[nan_filter] = None

        self.boxes = self.boxes.astype(int)
        if self.mask is not None:
            self.mask = self.mask.astype(bool)
        
    
    def add_detections(self, boxes, mask, confidence):
        
        #confidences are used as indicator if obejct is detected. None -> Undetected
        
        
        if isinstance(confidence,float):
            confidence = np.array([confidence] * len(boxes))
        
        confidence[confidence == 0] = None
        
        if self.boxes is None:
            self.boxes = boxes.astype(int)
            self.mask = mask
            self.confidence = confidence
        else:
            self.boxes = boxes.astype(int)
            self.mask = mask
            self.confidence = confidence
            
    def is_region(self):
        return not self.movable_to_container
            
    
    def get_mean_area(self):
        
        nan_filter = (self.confidence != None) & (~np.isnan(self.confidence))

        mask_areas = np.sum(self.mask[nan_filter], axis=(1, 2))

        mask_areas = mask_areas[mask_areas > 1]

        z_scores = zscore(mask_areas)

        outlier_idx = ~(abs(z_scores) > 2.5)

        filtered_idx = np.where(outlier_idx)[0]

        mean_area = mask_areas[filtered_idx].mean()
        
        self.mean_area = mean_area

        return mean_area
    
    
    
    def get_movement_direction(self,flow, robot_masks = None):
        

        mean_dirs = []
        for i in range(0, len(self.mask)):
            if self.confidence[i] == 0 or self.mask[i].sum()== 0:
                mean_dirs.append(np.array([0,0]))
            else:
                cur_flow = flow[i]
                # if self.mask[i].sum() > 1000:
                #     cur_mask = cv2.erode(self.mask[i].astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
                # else:
                #     cur_mask = self.mask[i]
                
                #if is region, get mask from bounding box
                if self.is_region():
                    cur_mask = np.zeros_like(self.mask[i])
                    cur_mask[self.boxes[i][1]:self.boxes[i][3],self.boxes[i][0]:self.boxes[i][2]] = True
                    
                    #remove dilated robot mask
                    robot_mask_dilated = cv2.dilate(robot_masks[i].astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
                    cur_mask = np.logical_and(cur_mask,~robot_mask_dilated)
                else:
                    cur_mask = self.mask[i]
                    robot_mask_dilated = cv2.dilate(robot_masks[i].astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)

                    cur_mask = np.logical_and(cur_mask,~robot_mask_dilated)
                    #cur_mask = self.mask[i]
                
                
                
                #Dont remove robot mask from small objects
                # if robot_masks is not None and self.mask[i].sum() > 1000:
                # 
                #     robot_mask_dilated = cv2.dilate(robot_masks[i].astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
                #     cur_mask = np.logical_and(cur_mask,~robot_mask_dilated)

                cur_flow_vis = cur_flow.copy()
                cur_flow_vis[~cur_mask] = [0,0]
                cur_flow_vis = np.linalg.norm(cur_flow_vis, axis=-1)
                cur_flow = cur_flow[cur_mask,:]


                
                vec_lengths = np.linalg.norm(cur_flow, axis=1)
                
                highest_idx = np.argsort(vec_lengths)[-200:]
                #mean_dir = np.mean(cur_flow[highest_idx], axis=0))
                #cur_flow = cur_flow[highest_idx]

                # z_scores_movement = zscore(vec_lengths, axis = 0)
                # total_movement_filter= np.abs(z_scores_movement) > 2
                z_scores = np.abs(zscore(cur_flow, axis=0))
                outlier_mask = np.any(z_scores > 2, axis=1)

                highest_idx = np.argsort(vec_lengths)[-200:]

                #cur_flow = cur_flow[highest_idx]

                #zscores = zscore(vec_lengths[highest_idx])
                #outlier_mask = zscores > 1
                
                
                filtered_vectors = cur_flow[~outlier_mask]

                if np.any(filtered_vectors.std(axis = 0)> 5) :
                    z_scores = np.abs(zscore(filtered_vectors, axis=0))
                    outlier_mask = np.any(z_scores > 2, axis=1)
                    filtered_vectors = filtered_vectors[~outlier_mask]
                    
                #filtered_vectors = cur_flow

                
                #Filter masks where dilated robot mask and object not in contact -> probaby noise, object not moving due to robot interaction
                

                robot_mask_dilated_filter = cv2.dilate(robot_masks[i].astype(np.uint8), np.ones((5, 5), np.uint8),
                                                iterations=2).astype(bool)
                
                overlap = np.logical_and(robot_mask_dilated_filter, cur_mask)
                if overlap.sum() < 20:
                    filtered_vectors = np.array([[0,0]])
                
                mean_dir = np.mean(filtered_vectors, axis=0)
                mean_dirs.append(mean_dir)
        mean_dirs = np.array(mean_dirs)
        mean_dirs[np.any(np.isnan(mean_dirs), axis=1)] = [0,0]
        return np.array(mean_dirs)
                


    def get_movement_3d(self):

        self.pos_3d = np.array(self.pos_3d).astype(float)
        pos_3d_interpolated = pd.DataFrame(self.pos_3d).interpolate(limit=1).rolling(3,min_periods=1,center = True).mean().values

        movement = np.diff(pos_3d_interpolated, axis = 0)
        movement = np.vstack([movement,np.array([0,0,0])])
        nan_filter = (self.confidence != None) & (~np.isnan(self.confidence))

        nan_filter = nan_filter[1:]

        movement[:-1][~nan_filter] = np.array([0,0,0])

        return movement


    def get_movement_directions_box(self):
        
        box_centers = (self.boxes[:, :2] + self.boxes[:, 2:]) / 2
        
        box_centers[box_centers.sum(axis = 1) == 0] = np.array([None,None])
        #interpolate box centers:

        box_centers = pd.DataFrame(box_centers).interpolate(limit=1).rolling(3,min_periods=1,center = True).mean().values
        
        box_center_movement = np.diff(box_centers, axis = 0)
        box_center_movement = np.vstack([box_center_movement,np.array([0,0])])
        
        nan_filter = (self.confidence != None) & (~np.isnan(self.confidence))
        
        nan_filter = nan_filter[1:]
        
        box_center_movement[:-1][~nan_filter] = np.array([0,0])
        

        return box_center_movement
    
    
    
    def get_state_prompt(self):
        if self.state_prompt is None:
            return self.name
        else:
            return self.state_prompt
    
    def get_movement(self,flow = None, robot_masks = None):
        
        #positions = get_bbox_center(self.boxes)
        
        #nan_mask = (self.confidence  != None  | np.isnan(self.confidence))
        
        #masks = self.mask[nan_mask]
        
        if self.pos_3d is not None:
            mean_dirs_3d = self.get_movement_3d()
        mean_movement_3d = np.linalg.norm(mean_dirs_3d,axis = 1)
        self.object_movement_3d = mean_movement_3d


        
        
        
        if flow is not None:
            mean_dirs = self.get_movement_direction(flow, robot_masks)
            

          
            
            
            

            #mean_dirs_smooth = pd.DataFrame(mean_dirs).astype(int).interpolate(limit=1).rolling(3,min_periods=1,center = True).mean().values
            
            mean_dirs_box = self.get_movement_directions_box()

            if self.pos_3d is not None:
                mean_dirs_3d = self.get_movement_3d()


            mean_dirs_box_filter = np.abs(mean_dirs_box) > np.abs(mean_dirs).max(axis = 0) * 10
            mean_dirs_box[mean_dirs_box_filter] = 0

            if not self.is_region():
                self.object_movement_dirs = np.mean([mean_dirs,mean_dirs_box],axis = 0)
            else:
                self.object_movement_dirs = mean_dirs
       
            
            
            mean_movement = np.linalg.norm(mean_dirs,axis = 1)
            mean_movement_3d = np.linalg.norm(mean_dirs_3d,axis = 1)
            
            consecutive_movement_filter_base = filter_consecutive_trues_np(mean_movement > 1.5, 2)

            # consecutive_movement_filter_base_3d = filter_consecutive_trues_np(mean_movement_3d > 0.1, 2)
            #
            # movement_frames_3d = mean_movement_3d > 0.1

            to_replace = (mean_movement > 1.5) & ~consecutive_movement_filter_base
            mean_movement[to_replace] = 0
            mean_movement_smooth = pd.Series(mean_movement).rolling(3,min_periods=1,center = True).mean().values
            
            mean_movement = mean_movement_smooth
          
        
            
            mean_movement_box = np.linalg.norm(mean_dirs_box,axis = 1)
            
            if self.is_region():
                self.object_movement = mean_movement
                return mean_movement
            
            #Normalize to 0-1
            
            orig_max = max(mean_movement.max(), mean_movement_box.max())
            orig_min = max(mean_movement.min(), mean_movement_box.min())
            
            mean_movement = (mean_movement - mean_movement.min()) / (mean_movement.max() - mean_movement.min())
            
            mean_movement_box = (mean_movement_box - mean_movement_box.min()) / (mean_movement_box.max() - mean_movement_box.min())
            
            
            #combine
            mean_movement = (mean_movement + mean_movement_box) / 2
            mean_movement = mean_movement.astype(np.float32)
            
            #rescale to orig scale:
            mean_movement = mean_movement * (orig_max - orig_min) + orig_min
            
            mean_movement = np.nan_to_num(mean_movement)
            
        else:
            mean_dirs_box = self.get_movement_directions_box()
            mean_movement_box = np.linalg.norm(mean_dirs_box,axis = 1)
            mean_movement = np.nan_to_num(mean_movement_box)



        self.object_movement = mean_movement
        self.object_movement_3d = mean_movement_3d
        return mean_movement
    
    
    def get_movement_box(self,start_frame, end_frame):
        
        valid_frames = np.where((self.confidence != None) & (~np.isnan(self.confidence)))[0]
        if len(valid_frames) == 0:
            return np.array([0,0])
        nearest_start_frame = valid_frames[np.argmin(np.abs(valid_frames - start_frame))]
        nearest_end_frame = valid_frames[np.argmin(np.abs(valid_frames - end_frame))]
        
        
        iou = box_iou(torch.tensor(self.boxes[nearest_start_frame][None]), torch.tensor(self.boxes[nearest_end_frame][None]))
        
        iou = iou.numpy()[0]
        
        center_start = (self.boxes[nearest_start_frame, :2] + self.boxes[nearest_start_frame, 2:]) / 2

        center_end = (self.boxes[nearest_end_frame, :2] + self.boxes[nearest_end_frame, 2:]) / 2

        movement = center_end - center_start

        length = np.linalg.norm(movement)
        
        if iou > 0.5:
            movement = np.array([0,0])
        
        
        return movement
    
    def get_nl_movement(self,start_frame,end_frame,keystate_idx = None):
        
        #get movememnt based on bounding box:
        
        #get nearerst non nan frame

        valid_frames = np.where((self.confidence != None) & (~np.isnan(self.confidence)))[0]
        if len(valid_frames) == 0:
            return ""
        nearest_start_frame = valid_frames[np.argmin(np.abs(valid_frames - start_frame))]
        nearest_end_frame = valid_frames[np.argmin(np.abs(valid_frames - end_frame))]
        
        center_start = (self.boxes[nearest_start_frame, :2] + self.boxes[nearest_start_frame, 2:]) / 2
        
        center_end = (self.boxes[nearest_end_frame, :2] + self.boxes[nearest_end_frame, 2:]) / 2


        if keystate_idx is not None:

            pos_start_3d = self.pos_3d[keystate_idx]
            pos_end_3d = self.pos_3d[keystate_idx + 1]
        else:
            pos_start_3d = self.pos_3d[nearest_start_frame]
            pos_end_3d = self.pos_3d[nearest_end_frame]

        if pos_start_3d[0] is None or pos_end_3d[0] is None:
           movement_3d = np.array([0,0,0])
        else:
            movement_3d = pos_end_3d - pos_start_3d
        
        movement = center_end - center_start
        
        length = np.linalg.norm(movement)

        movement_str = ""
        movement_thresh = 0.13
        movement_relations = []
        use_3d = True
        if use_3d:
            if abs(movement_3d[0]) > movement_thresh:
                if movement_3d[0] > movement_thresh:
                    movement_str += f"to the right "
                    movement_relations.append("to the right")
                elif movement_3d[0] < -movement_thresh:
                    movement_str +=  f"to the left"
                    movement_relations.append("to the left")
            if abs(movement_3d[2]) > movement_thresh:
                if movement_3d[2] > movement_thresh:
                    movement_str +=  f"forward"
                    movement_relations.append("forward")
                elif movement_3d[2] < -movement_thresh:
                    movement_str +=  f"backward"
                    movement_relations.append("backward")
            if abs(movement_3d[1]) > movement_thresh * 1.7:
                if movement_3d[1] > movement_thresh:
                    movement_str +=  f"up"
                    movement_relations.append("up")
                elif movement_3d[1] < -movement_thresh*1.7:
                    movement_str +=  f"down"
                    movement_relations.append("down")
        else:
            if abs(movement[0]) > 8:
                if movement[0] > 8:
                    movement_str += f"to the right "
                    movement_relations.append("to the right")
                elif movement[0] < -8:
                    movement_str +=  f"to the left"
                    movement_relations.append("to the left")
            if abs(movement[1]) > 8:
                if movement[1] > 8:
                    movement_str +=  f"forward"
                    movement_relations.append("forward")
                elif movement[1] < -8:
                    movement_str +=  f"backward"
                    movement_relations.append("backward")


        # if len(movement_str) >0:
        #     movement_str = f"{self.name} moved {movement_str} "
        #     return movement_str

        if len(movement_relations) > 0:
            return f"{self.name} moved {' and '.join(movement_relations)}"
        else:
            return ""

        
        return ""
        
        
        

        



class RobotObject(Object):
    def __init__(self, name = "robot", states = None, movable_to_container = False, container = False, interactable = False):
        super().__init__(name,states, movable_to_container, container, interactable)
        
        self.gripper_locations = None
        self.gripper_locations_3d = None
        
        
    def get_last_known_gripper_location(self, index,range = -1):
    
        gripper_locations = self.gripper_locations_3d if self.gripper_locations_3d is not None else self.gripper_locations
        
        viable_indices = np.where(gripper_locations.sum(axis=1) != 0)[0]
        if len(viable_indices) == 0:
            return None
            
        last_known_index = viable_indices[np.argmin(np.abs(viable_indices - index))]
        if range != -1:
            dist = np.abs(last_known_index - index)
            if dist > range:
                return None,None
            
        return gripper_locations[last_known_index], last_known_index
        
        
    


class SurfaceObject(Object):
    def __init__(self, name, states, movable_to_container, container, interactable, nl_name=None, color=None):
        super().__init__(name, states, movable_to_container, container, interactable, nl_name, color)


        self.normals = None
        self.combined_mask = None


class NonMovableObject(Object):
    def __init__(self, name,states, movable_to_container, container, interactable, nl_name = None, color = None):
        super().__init__(name,states, movable_to_container, container, interactable, nl_name, color)

    def process(self):
        
        super().process()
        
        # Override the preprocess method for NonMovableObject

        #object_masks = self.list_of_dict_of_dict_to_dict_of_dict_of_list(object_masks)
        
        if np.all((self.confidence == None) | np.isnan(self.confidence)):
            return
        
        nan_filter = (self.confidence != None) & (~np.isnan(self.confidence))
        

        
        boxes = self.boxes[nan_filter]
        confidences = self.confidence[nan_filter]
    
        

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        #Remove boxes that encompass almost whole scene
        area_filter = areas < 0.8 * self.mask.shape[1] * self.mask.shape[2]

        
        boxes = boxes[area_filter]
        confidences = confidences[area_filter]
        areas = areas[area_filter]
        
        if np.unique(areas).shape[0] == 1:
            self.boxes = np.repeat(boxes[0][None],len(self.boxes),axis = 0)
            self.confidence = np.repeat(confidences[0][None],len(self.confidence),axis = 0)
            return
        area_zscores = zscore(areas)
        area_filter = np.abs(area_zscores) < 2.5

        boxes_filtered = boxes[area_filter]
        confidences = confidences[area_filter]

        box_centers = (boxes_filtered[:, :2] + boxes_filtered[:, 2:]) / 2

        # cluster centers and detect outliers

        eps = 0.04 * self.mask.shape[1]

        # box_centers = (box_centers - box_centers.min()) / (box_centers.max() - box_centers.min())
        if len(box_centers) < 1:
            self.boxes = np.repeat(np.array([0,0,0,0])[None],len(self.boxes),axis = 0)
            self.confidence = np.repeat(0.0,len(self.confidence),axis = 0)
            return
        
        db = DBSCAN(eps=eps, min_samples=max(2,min(len(confidences) // 6,10))).fit(box_centers)
        labels = db.labels_
        unique_labels = np.unique(labels, return_counts=True)
        # remove noise label

        unique_labels = (unique_labels[0][unique_labels[0] != -1], unique_labels[1][unique_labels[0] != -1])
        
        #If no clusters found, reduce min samples
        if len(unique_labels[0]) == 0:
            db = DBSCAN(eps=eps, min_samples=max(len(confidences) // 7,2)).fit(box_centers)
            labels = db.labels_
            unique_labels = np.unique(labels, return_counts=True)
            unique_labels = (unique_labels[0][unique_labels[0] != -1], unique_labels[1][unique_labels[0] != -1])
        
        if len(unique_labels[0]) == 0:
            self.boxes = np.repeat(boxes_filtered[0][None],len(self.boxes),axis = 0)
            self.confidence = np.repeat(confidences[0][None],len(self.confidence),axis = 0)
            return
        max_count = max(unique_labels[1])
        normalized_count_score = unique_labels[1] / max_count
        max_conf_cluster = np.argmax([np.mean(confidences[labels == cluster]) for cluster in unique_labels[0]] + normalized_count_score)
        
        #majority_cluster = unique_labels[0][np.argmax(unique_labels[1])]
        majority_cluster = max_conf_cluster
        
        
        avg_boxes = np.array([np.mean(boxes_filtered[labels == cluster], axis=0).astype(int) for cluster in unique_labels[0]])
        
        
    
        avg_box = np.mean(boxes_filtered[labels == majority_cluster], axis=0).astype(int)

        ious = box_iou(torch.tensor(avg_box[None]), torch.tensor(avg_boxes))
        
        if "drawer" in self.name:
            avg_overlap_boxes = ious > 0.15
            #combine boxes with high overlap
            combined_box = avg_box
            for box in avg_boxes[np.array(avg_overlap_boxes[0])]:
                combined_box = np.array([min(combined_box[0], box[0]), min(combined_box[1], box[1]), max(combined_box[2], box[2]), max(combined_box[3], box[3])])
            avg_box = combined_box
        

        avg_confidence = np.mean(confidences[labels == majority_cluster])
        
        
        
        self.boxes = np.array([avg_box]).repeat(len(self.boxes), axis=0)
        self.confidence = np.array([avg_confidence]).repeat(len(self.confidence), axis=0)
    
    
    def get_nl_movement(self,start_frame,end_frame,keystate_idx = None):
        return ""
    
    
class ObjectManager:
    def __init__(self,object_dict, segmentation_prompts = None,detection_prompts = None,state_prompts = None,cfg = None):
        self.objects = {}

        self.segmentation_bg_prompts = []
         
        self.initialize(object_dict)
        
        self.robot = RobotObject()
        
        self.surface_object = None
        
       
         
        if segmentation_prompts is not None:
             self.set_ov_segmentation_prompts(segmentation_prompts)
            
        if detection_prompts is not None:
            self.set_detection_prompts(detection_prompts)
            
        if state_prompts is not None:
            self.set_state_prompts(state_prompts)
            
        self.cfg = cfg
            
        
    
    
    def add_synonyms(self,synonyms):
        for obj in self.objects:
            if obj in synonyms:
                self.objects[obj].synonyms = synonyms[obj]
    
    def get_object_dict_in_interval(self,start_frame,end_frame,object_name = None):
        
        movements = {}
        
        if object_name is not None:
            for obj in self.objects:
                if obj == object_name:
                    movements[obj] = self.objects[obj].get_movement_box(start_frame,end_frame)
                    
                    return movements
        else:
            for obj in self.objects:
                movements[obj] = self.objects[obj].get_movement_box(start_frame,end_frame)
            return movements
    
    def get_object_movements_in_interval(self,start_frame,end_frame,keystate_object = None,keystate_idx = None):
        movements = {}
        
        if keystate_object is not None:
            for obj in self.objects:
                if obj == keystate_object:
                    movements[obj] = self.objects[obj].get_nl_movement(start_frame,end_frame,keystate_idx)
                    
                    return movements
        else:
            for obj in self.objects:
                movements[obj] = self.objects[obj].get_nl_movement(start_frame,end_frame,keystate_idx)
            return movements
    

    def get_closest_valid_state(self,obj,frame_idx,dir = "left"):

        valid_state_indices = np.where((self.objects[obj].predicted_states != "None"))[0]

        if len(valid_state_indices) == 0:
            return None

        possible_state_indices = valid_state_indices <= frame_idx + 2 if dir == "left" else valid_state_indices >= frame_idx - 2

        if np.sum(possible_state_indices) == 0:
            return None
        closest_state_idx = valid_state_indices[possible_state_indices][np.argmin(np.abs(valid_state_indices[possible_state_indices] - frame_idx))]


        return self.objects[obj].predicted_states[closest_state_idx]






    def get_state_changes_in_interval(self,start_frame,end_frame):
        changes = {}
        for obj in self.objects:
            
            
            
            states = self.objects[obj].predicted_states[start_frame:end_frame]
            
            if len(np.unique(states)) > 1:
                start_state = states[0]
                if start_state == "None":
                    start_state = self.get_closest_valid_state(obj,start_frame,dir = "left")
                    if start_state is None:
                        return {}

                end_state = states[-1]
                if end_state == "None":
                    end_state = self.get_closest_valid_state(obj,end_frame,dir = "right")
                    if end_state is None:
                        return {}


                if start_state == end_state:
                    end_state = states[len(states) // 2]
                changes[obj] = (start_state,end_state)
            
        return changes
    
    def get_last_states(self):
        states = {}
        for obj in self.objects:
            if self.objects[obj].states is not None and len(self.objects[obj].predicted_states) > 0:
                states[obj] = self.objects[obj].predicted_states[-1]
        return states
    
    def get_object_mask_areas(self):
        areas = {}
        for obj in self.objects.values():
           
            areas[obj.name] = obj.get_mean_area()
        return areas
    



    def get_object_stats_frame(self,frame_idx):
        detections = {}
        for obj in self.objects:
            
            if self.objects[obj].states is None or len(self.objects[obj].predicted_states) == 0:
                state  = None
            else:
                state = self.objects[obj].predicted_states[frame_idx]
            
            cur_obj = self.objects[obj]
            
            if cur_obj.object_movement_dirs is None:
                cur_tstep_movement = None
                #cur_tstep_movement = cur_obj.object_movement[frame_idx]
            else:
                cur_tstep_movement = cur_obj.object_movement_dirs[frame_idx]
            detections[obj] = {"box":cur_obj.boxes[frame_idx],"mask":cur_obj.mask[frame_idx],"score":cur_obj.confidence[frame_idx],"obj": cur_obj,
                               "state":state, "movement":cur_tstep_movement}
            
        return detections

    def get_robot_stats_frame(self,frame_idx):
        detections = {}
        state = None


        if self.robot.object_movement_dirs is None:
            cur_tstep_movement = None
            #cur_tstep_movement = cur_obj.object_movement[frame_idx]
        else:
            cur_tstep_movement = self.robot.object_movement_dirs[frame_idx]

        detections["robot"] = {"box":self.robot.boxes[frame_idx],"mask":self.robot.mask[frame_idx],"score":self.robot.confidence[frame_idx],"obj": self.robot,
                               "state":state, "movement":cur_tstep_movement}
        return detections
    
    def get_all_obj_names(self):
        return list(self.objects.keys())
    
    def get_all_object_stats(self):
        all_stats = {}
        for obj in self.objects:
            cur_obj = self.objects[obj]
            all_stats[obj] = {"box":cur_obj.boxes,"mask":cur_obj.mask,"score":cur_obj.confidence, "obj": cur_obj}
        return all_stats
       
    def get_object_detection_prompts(self):
        prompts = {}
        
        # if self.surface_object is not None:
        #     prompts[self.surface_object.name] = f"{self.surface_object.color} {self.surface_object.name}"
            
        
        for obj in self.objects:
            cur_prompt = self.objects[obj].detection_prompt
            if cur_prompt is None:
                cur_prompt = obj
            prompts[obj] = cur_prompt
            
            if self.objects[obj].color is not None:
                prompts[obj] = f"{self.objects[obj].color} {cur_prompt}".strip()
        

        return prompts
    
    
    def get_object_locations_box_center(self):
        boxes = {}
        for obj in self.objects:
            boxes[obj] = self.objects[obj].boxes
            centers = (self.objects[obj].boxes[:, :2] + self.objects[obj].boxes[:, 2:]) / 2
            boxes[obj] = centers
        return boxes
    
    def get_object_boxes(self):
        boxes = {}
        for obj in self.objects:
            boxes[obj] = self.objects[obj].boxes
        return boxes


    def get_object_scores(self):
        scores = {}
        for obj in self.objects:
            scores[obj] = self.objects[obj].confidence
        return scores
    
    
    def get_object_segmentation_prompts(self):
        prompts = {}
        for obj in self.objects:
            cur_prompt = self.objects[obj].ov_seg_prompt
            if cur_prompt is None:
                cur_prompt = obj
            prompts[obj] = cur_prompt
        return prompts
    
    
    def get_region_objects(self):
        region_objects = {}
        for obj in self.objects:
            if self.objects[obj].is_region():
                region_objects[obj] = self.objects[obj]
        return region_objects
    
    
    def add_surface_object(self,name,color):
        
        self.surface_object = SurfaceObject(name,[],None,False,False,False,color)
        
    

    def add_object(self, name, properties):
        if properties['movable']:
            obj = Object(name,properties['states'], properties['movable'], properties['container'], properties['interactable'],
                         name, properties['color'])
        else:

                
            obj = NonMovableObject(name,properties['states'], properties['movable'], properties['container'], properties['interactable'],
                                   name, properties['color'])
        self.objects[name] = obj
    
    
    
    def initialize(self, objects_dict):
        for name, properties in objects_dict.items():
            #check for properties:
            if "states" not in properties:
                properties["states"] = None
            if "movable" not in properties:
                properties["movable"] = False
                
            if "container" not in properties:
                properties["container"] = False
            
            self.add_object(name, properties)
            
    
    def set_obj_masks(self,obj,masks):
        self.objects[obj].mask = masks
        
    
    def add_object_detections(self,detection_dict,is_voted_temporal = False):
        
        detection_dict = detection_dict.copy()
        
        predicted_classes = []

        process = not is_voted_temporal and self.cfg.enable_detection_refinment and len(detection_dict["robot"]["box"]) > 4
        
        for name, detection in detection_dict.items():
            
            

            if name =="robot":
                self.robot.add_detections(detection["box"],detection["mask"],detection["score"])
                if not is_voted_temporal:
                    self.robot.process()
           
                
            else:
                process = process or isinstance(self.objects[name],NonMovableObject)
                if self.objects[name].boxes is None:
                    self.objects[name].add_detections(detection["box"],detection["mask"],detection["score"])
                    if process:
                        self.objects[name].process()
                else:
                    self.objects[name].add_detections(detection["box"],detection["mask"],detection["score"])
                    if process:
                        self.objects[name].process()
                    logging.warning(f"Object {name} already has detections")
                predicted_classes.append(name)
        
        if is_voted_temporal:
            #set undetected classes scores to nan
            for name in self.objects:
                if isinstance(self.objects[name],NonMovableObject):
                    continue
                if name not in predicted_classes:
                    self.objects[name].confidence = np.array([None] * len(self.objects[name].confidence)).astype(float)

            #clean objects with low score and few detections:
            #self.clean_objects()
            

    def calc_object_consistency(self):

        for obj in self.objects.values():
            if (obj.mask.sum(axis =(1,2)).mean() - obj.mask.sum(axis =(1,2)).std()) < 0:
                obj.confidence = obj.confidence-0.25




    def clean_objects(self,threshold = 0.2,predefined_objects = []):


        predefined_object_names = [obj["name"] if isinstance(obj,dict) else obj for obj in predefined_objects]
        
        obj_to_clean = []


        #self.calc_object_consistency()



        #calc consistency score of detections

        for obj in self.objects:
            cur_obj = self.objects[obj]

            #cur_obj.process()


            nan_vals = ((np.isnan(self.objects[obj].confidence)) | (self.objects[obj].confidence == None))
            missing_count = nan_vals.sum()


            if missing_count > 0.5*len(self.objects[obj].confidence) or np.mean(self.objects[obj].confidence[~nan_vals]) < threshold:
                obj_to_clean.append(obj)


        cleaned_objects = []

        for obj in obj_to_clean:
            if obj in predefined_object_names:
                continue
            self.objects.pop(obj)
            cleaned_objects.append(obj)
        logging.info(f"Cleaning objects: {cleaned_objects}")




    def set_detection_prompts(self,prompts):
        for name, prompt in prompts.items():
            if name in self.objects:
                self.objects[name].detection_prompt = prompt
            else:
                self.segmentation_bg_prompts.append(prompt)
                
    def set_state_prompts(self,prompts):
        for name, prompt in prompts.items():
            if name in self.objects:
                self.objects[name].state_prompt = prompt

                
                
    def set_ov_segmentation_prompts(self,prompts):
        for name, prompt in prompts.items():
            if name in self.objects:
                self.objects[name].ov_seg_prompt = prompt
            else:
                self.segmentation_bg_prompts.append(prompt)
                
                
    def get_object_movements(self,flow):
        movements = {}
        for obj in self.objects:
            movements[obj] = self.objects[obj].get_movement(flow,self.robot.mask)
        return movements
    
    
        



