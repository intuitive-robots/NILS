import logging
import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from nils.annotator.keystate_predictors.keystate_predictor import (
    KeystatePredictor,
)
from nils.utils.utils import filter_consecutive_values_with_outlier_fill


class ObjectMovementKeystatePredictor(KeystatePredictor):
    
    def __init__(self, name,downsample_interval=1, gripper_close_movement_filter=True,movement_threshold=0.1,n_frames_movement=2, offset = 0):
        super(ObjectMovementKeystatePredictor, self).__init__(name,downsample_interval)

        self.gripper_close_movement_filter = gripper_close_movement_filter
        
        self.movement_threshold = 1.5
        self.n_frames_movement = n_frames_movement
        self.n_frames_movement = 2
        self.offset = offset

        self.max_score_movement = 8
        
        self.placeholder_prompt = "No object movement detected"
        
    
    def predict(self, data):

        batch = data["batch"]
        scene_graphs = data["scene_graphs"]
        object_manager = data["object_manager"]
        object_movements = data["object_movements"]
        
        scene_graphs_subset = np.linspace(0, len(batch["rgb_static"]) - 1,
                                          len(batch["rgb_static"]) // self.downsample_interval).astype(np.int32)
        batch_subset = {key: batch[key][scene_graphs_subset] for key in batch.keys()}

        # robot_pos_screen = get_tcp_center_screen(batch_subset, self.env)
        robot_pos_screen = None
        if "gripper_actions" not in batch_subset:
            gripper_actions = None
        else:
            gripper_actions = batch_subset["gripper_actions"]
        object_movement_keystates, scores = self.detect_movement(scene_graphs, gripper_actions, object_movements,object_manager)
        position_based = self.detect_movements_position_based(scene_graphs, gripper_actions, object_movements, object_manager=data["object_manager"])

        moved_objects = [obj for obj, frame in object_movement_keystates]
        object_movement_dict = object_movement_keystates
        object_movement_keystates = np.array([ks[1] for ks in object_movement_keystates])
        object_movement_keystates = (object_movement_keystates * self.downsample_interval)

        self.keystates = object_movement_keystates
        
        scores = np.array(scores)
        if len(scores) == 0:
            self.keystate_scores = np.array([])
            self.keystate_reasons = []

            return object_movement_keystates

        #scores = (scores) / (np.max(scores) + 1e-6)
        scores = (scores) / self.max_score_movement

        self.keystate_scores = scores
        
        
        self.keystate_reasons = moved_objects
        
        self.compute_keystate_reasons_nl()
        

        return object_movement_keystates
    
    def compute_keystate_reasons_nl(self):
        
        nl_reasons = []
        for keystate_reason in self.keystate_reasons:
            nl_reasons.append("The robot interacted with {} and it moved".format(keystate_reason))
        
        for obj in self.object_keystate_scores.keys():
            self.keystate_reasons_by_object_nl[obj] = ["Object movements: The {} moved.".format(obj)] * len(self.object_keystate_scores[obj])
        
        self.keystate_reasons_nl = nl_reasons
        
    def detect_movements_position_based(self, scene_graphs, gripper_actions, object_movements,object_manager):
        positions = {}
        
        height,width =  object_manager.robot.mask.shape[1:]
        scale = max(height,width)
        
        mask_areas = {}
        for sg in scene_graphs:
            for node in sg.nodes:
                
                if "robot" in node.name or object_manager.objects[node.name].is_region():
                    continue
                if node.name not in positions:
                    positions[node.name] = []
                if node.name not in mask_areas:
                    mask_areas[node.name] = []
                if node.seg_mask is None:
                    positions[node.name].append(np.array([0,0]))
                    mask_areas[node.name].append(0)
                    continue
                mask = node.seg_mask
                mask_center = np.mean(np.where(mask),axis = 1)
                mask_center = mask_center[::-1]
                
                #other_pos = node.pos    
                #smask_center = other_pos
                
                positions[node.name].append(mask_center)
               
               
                mask_areas[node.name].append(np.sum(mask))
        
        positions = {obj: np.array(pos) for obj, pos in positions.items()}
        mask_areas = {obj: np.array(area) for obj, area in mask_areas.items()}
        
        obj_keystates = {}
        for obj, pos in positions.items():
            
            mean_area = np.mean(mask_areas[obj])
            
            mask_fractions = mask_areas[obj][1:] / mask_areas[obj][:-1]
            mask_fractions = np.concatenate([mask_fractions,np.array([1])])
            mask_fraction_scores = np.abs(mask_fractions - 1)
            
            thresh = 0.1
            
            mask_fraction_scores = mask_fraction_scores > thresh
            
            mean_area_filter = mask_areas[obj] < 0.5 * mean_area
            mask_fraction_scores = mean_area_filter | mask_fraction_scores
            
            cur_pos = pos
            cur_pos = pd.DataFrame(cur_pos)
            cur_pos[mask_fraction_scores] = None
            cur_pos = cur_pos.interpolate(method="linear", axis=0)
            cur_pos = cur_pos.bfill()
            
            pos = cur_pos.values 
            #smoothed_pos = pos
            
            smoothed_pos = np.apply_along_axis(lambda m: np.convolve(m, np.ones(3) / 3, mode="same"), 0, pos)
            smoothed_pos[0] = pos[0]
            smoothed_pos[-1] = pos[-1]
            last_pos = smoothed_pos[0]
            
            last_box = object_manager.objects[obj].boxes[0]
            
            last_t_step = 0
            diff = np.linalg.norm(np.diff(smoothed_pos, axis=0),axis = 1)
            
            t_to_skip = 0
            for i,cur_pos in enumerate(smoothed_pos):
                cur_box = object_manager.objects[obj].boxes[i]
                if t_to_skip > 0:
                    t_to_skip -= 1
                    continue
                
                
                if np.linalg.norm(cur_pos - last_pos) > 0.07 * scale:
                    
                    #verifiy object position constant for next n frames:

                    print(f"Movement detected from tstep {last_t_step} to {i}")
                    print(np.linalg.norm(cur_pos - last_pos))
                    print(f"Movement detected: {obj}")
                    print(i)
                    next_non_moving = np.where(diff[i:] < 0.02 * scale)[0]
                    
                    
                    
                    if len(next_non_moving) == 0:
                        #take last step
                        next_non_moving = len(diff) - i
                    else:
                        
                        next_non_moving = next_non_moving[0]
                    print(next_non_moving + i)
                    
                    last_pos = smoothed_pos[i + next_non_moving]
                    
                    last_box = object_manager.objects[obj].boxes[i + next_non_moving]
                    
                    #Verify by IOU:
                    iou = box_iou(torch.tensor([last_box]),torch.tensor([cur_box])).numpy()[0]
                    if iou > 0.5:
                        continue
                        
                    
                    if obj not in obj_keystates:
                        obj_keystates[obj] = []
                    obj_keystates[obj].append(i + next_non_moving)
                    t_to_skip = next_non_moving
                    last_t_step = i + next_non_moving
        
        
        #merge with movement based keystates:
        merged_ks = {}
        merged_ks_scores = {}
        
        non_matched_score = 0.3
        matched_score = 0.4
        
        
        
        
        for obj, keystates in obj_keystates.items():

            if obj not in merged_ks:
                merged_ks[obj] = []
                merged_ks_scores[obj] = []

            near_threshold = 8
            flow_ks = np.array(self.keystates_by_object[obj])
            for ks in keystates:
                if len(flow_ks) == 0:
                    merged_ks[obj].append(ks)
                    merged_ks_scores[obj].append(non_matched_score)
                else:
                    nearest_dist = np.min(np.array(flow_ks) - ks)
                    nearest = np.argmin(np.abs(np.array(flow_ks) - ks))
                    if nearest_dist < near_threshold:
                        merged_ks[obj].append((flow_ks[nearest] + ks) // 2)
                        merged_ks_scores[obj].append(self.object_keystate_scores[obj][nearest] + matched_score)
                        self.object_keystate_scores[obj][nearest] += matched_score
                    else:
                        merged_ks[obj].append(ks)
                        merged_ks_scores[obj].append(non_matched_score)
                
        
        
        for obj,ks in merged_ks.items():
            self.keystates_by_object[obj] = ks
        for obj,ks in merged_ks_scores.items():
            self.object_keystate_scores[obj] = ks
        # self.object_keystate_scores = merged_ks_scores
        # self.keystates_by_object = merged_ks
        
        
                    
                
        s = 1

    def detect_movement(self, scene_graphs, gripper_actions, object_movements,object_manager):
        
        if self.gripper_close_movement_filter:
            if gripper_actions is None:
                gripper_close_frames = None
                logging.warning("Gripper actions not found and gripper close movement filter is enabled.")
            else:
                gripper_close_frames = [i for i, action in enumerate(gripper_actions) if action == -1]
                gripper_close_frames = np.array(gripper_close_frames)
            
        else:
            gripper_close_frames = None
            

        image_shape = object_manager.robot.mask.shape[1:]

        max_dim = max(image_shape)
        movement_thresh = 0.015 * max_dim

        #object_movements = self.object_manager.get_object_movements(all_flows)

        keystates = []
        moved_objects = []
        max_movement = []
        keystate_scores = []
        for obj, movement in object_movements.items():
            
            if not object_manager.objects[obj].interactable:
                continue
            
            
            # Only consider movement where gripper is closed:
            if obj not in self.keystates_by_object:
                self.keystates_by_object[obj] = []
            if gripper_close_frames is not None and self.gripper_close_movement_filter:
                movement[~gripper_close_frames] = 0

            percentile = 95
            percentile_val = np.percentile(movement, 95)
            threshold = max(percentile_val,0.15)
            alt_threshold = max(self.movement_threshold, np.mean(movement))


            # Calculate the 1st and 3rd quartiles (25th percentile and 75th percentile)
            Q1 = np.percentile(movement, 25)
            Q3 = np.percentile(movement, 75)

            # Calculate the Interquartile Range (IQR)
            IQR = Q3 - Q1

            # Define the threshold for movement. This is typically 1.5 * IQR above the 3rd quartile
            threshold = min(Q3 + 3 * IQR,movement_thresh)
            
            over_threshold = movement > threshold
            
           
            
            over_threshold_filled = filter_consecutive_values_with_outlier_fill(over_threshold, min_length=1, outlier_fill_tolerance=4)
            changes = np.diff(over_threshold.astype(int))
            start_indices = np.where(changes == 1)[0] + 1
            end_indices = np.where(changes == -1)[0]

            if over_threshold[-1]:
                end_indices = np.append(end_indices, len(over_threshold) - 1)

            if over_threshold[0]:
                start_indices = np.insert(start_indices, 0, 0)

            # Filter short movements:
            for start_indices, end_indices in zip(start_indices, end_indices):
                if end_indices - start_indices+1 >= self.n_frames_movement:
                    
                    keystate_scores.append(end_indices - start_indices)
                    keystates.append((obj, end_indices + self.offset))
                    
                    self.keystates_by_object[obj].append((end_indices + self.offset)*self.downsample_interval)
                    self.object_keystate_scores[obj].append(end_indices - start_indices)
                    self.keystate_reasons_by_object[obj].append(obj)
                    
                    moved_objects.append(obj)
                    max_movement.append(np.max(movement[start_indices:end_indices]))
        for obj in self.object_keystate_scores.keys():
            self.object_keystate_scores[obj] = np.array(self.object_keystate_scores[obj])
            if len(self.object_keystate_scores[obj]) > 0:
                self.object_keystate_scores[obj] = (self.object_keystate_scores[obj]) / self.max_score_movement

        for obj in self.keystates_by_object.keys():
            self.keystates_by_object[obj] = np.array(self.keystates_by_object[obj])


        return keystates,keystate_scores
