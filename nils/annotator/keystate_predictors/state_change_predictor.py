import time

import numpy as np
from matplotlib import pyplot as plt

from nils.annotator.keystate_predictors.keystate_predictor import (
    KeystatePredictor,
)
from nils.utils.plot import crop_images_with_boxes


class StateChangeKeystatePredictor(KeystatePredictor):
    
    def __init__(self,name, downsample_interval=1, verify_with_vlm = False):
        super(StateChangeKeystatePredictor, self).__init__(name,downsample_interval)
        

        
        self.verify_with_vlm = verify_with_vlm
        self.max_len_score = 20
        
        self.placeholder_prompt = "No object state change detected"
        
        
        
    
    def predict(self, data):

        object_manager = data["object_manager"]
        
        state_changed_indices = []
        state_change_reasons = []
        
        vlm = data["vlm"]
        
        images = data["batch"]["rgb_static"][::self.downsample_interval]
        
        
        
        robot_masks = object_manager.robot.mask

        keystate_scores = []
        for obj in object_manager.objects.values():
            
            if obj.states is not None:
                cropped_robot_masks = np.array(
                    [crop_images_with_boxes(robot_mask.astype(np.uint8), obj.boxes[i], resize=True, resize_dim= None) if not np.isnan(obj.confidence[i]) and not np.isnan(object_manager.robot.confidence[i])  else np.ones_like(robot_mask) for i, robot_mask in
                     enumerate(robot_masks)] )
                state_confidences = obj.state_confidences
                state_confidences_softmax = np.exp(state_confidences * 100 ) / np.sum(np.exp(state_confidences * 100), axis=1)[:, None]
                max_conf = np.max(state_confidences, axis=1)
                pred_states_nl = np.array(obj.predicted_states)
                pred_states_filtered_changes = pred_states_nl[:-1] != pred_states_nl[1:]

                state_changed_indices_obj = np.where(pred_states_filtered_changes != 0)[0]
                
                keystate_scores = []
                
                if len(state_changed_indices_obj) == 0:
                    continue
                
                keystates = []
                state_change_indices_new_obj = []
                state_change_indices_after_occlusion = []
                
                state_change_reasons_new_after_occlusion = []
                state_change_reasons_new = []
                scores_new = []
                
                occlusion_lengths = []
                
                
                
                
                len_unknown = 0
                first_valid = np.where(pred_states_nl != "None")[0][0]
                last_state = pred_states_nl[first_valid]
                last_valid_state = pred_states_nl[first_valid]
                for i in range(first_valid,len(pred_states_nl)):
                    if pred_states_nl[i] == "None":
                        last_state = "None"
                        len_unknown += 1
                        continue
                    if pred_states_nl[i] != last_state:
                    
                        if last_state == "None" and last_valid_state != pred_states_nl[i]:
                            future_state = pred_states_nl[i: i + 2 * self.downsample_interval]
                            if np.all(future_state == pred_states_nl[i]):
                                state_change_indices_after_occlusion.append(i)
                                state_change_reasons_new_after_occlusion.append({"object": obj.name, "index": i * self.downsample_interval,
                                              "reason": (last_valid_state, pred_states_nl[i])})
                                scores_new.append(max(1 / (len_unknown + 1) * 15,1))
                                print(len_unknown)
                            occlusion_lengths.append(len_unknown)
                            len_unknown = 0
                            last_valid_state = pred_states_nl[i]

                        elif last_state != "None" and last_state != pred_states_nl[i]:
                            
                            last_valid_state = pred_states_nl[i]
                            # check in the future if the state is still the same
                            future_state = pred_states_nl[i: i + 2* self.downsample_interval]
                            if np.all(future_state == pred_states_nl[i]):
                                state_change_indices_new_obj.append(i * self.downsample_interval)
                                state_change_reasons_new.append({"object": obj.name, "index": i * self.downsample_interval,
                                              "reason": (last_state, pred_states_nl[i])})
                                scores_new.append(1.0)
                        # unknown to state:

                        else:
                            len_unknown = 0
                        last_state = pred_states_nl[i]
                
                
                if len(occlusion_lengths) > 0:
                    state_change_indices_after_occlusion_adjusted = (np.array(state_change_indices_after_occlusion) - np.min(occlusion_lengths) // 3) * self.downsample_interval 
                else:
                    state_change_indices_after_occlusion_adjusted = np.array([])
                total_keystates = np.concatenate([state_change_indices_new_obj,state_change_indices_after_occlusion_adjusted]).astype(int)
                total_reasons = state_change_reasons_new + state_change_reasons_new_after_occlusion
                state_changed_indices.extend(list(total_keystates))
                state_change_reasons.extend(total_reasons)
                self.keystates_by_object[obj.name].extend(list(total_keystates))
                self.object_keystate_scores[obj.name].extend(scores_new)
                self.keystate_reasons_by_object[obj.name].extend(total_reasons)
                
                
                           
                            
                            
                    

                # keystate_scores.append(np.mean(max_conf[:state_changed_indices_obj[0]]))
                # for i in range(1, len(state_changed_indices_obj)):
                #     start_index = state_changed_indices_obj[i-1] 
                #     end_index = state_changed_indices_obj[i]
                #     cur_class_idx = np.argmax(state_confidences[start_index+1:end_index+1], axis=1)
                #     class_idx,counts = np.unique(cur_class_idx,return_counts=True)
                #     class_idx = class_idx[np.argmax(counts)]
                #     #score is diff to other states
                #     cur_scores = state_confidences_softmax[start_index+1:end_index+1][:,class_idx]
                #     other_scores = state_confidences_softmax[start_index+1:end_index+1][:,np.arange(len(state_confidences[0])) != class_idx]
                #     
                #     
                #     
                #     diff = cur_scores[:,None] - other_scores
                #     diff = np.mean(diff)
                #     score = diff * min((end_index-start_index) /self.max_len_score,1.)
                #     chunk = max_conf[start_index:end_index]
                #     confidence = np.mean(chunk)
                #     #keystate_scores.append(score)
                #     keystate_scores.append(1.0)
                # 
                # # Add the last chunk from the last index to the end of the value array
                # 
                # 
                # for i,idx in enumerate(state_changed_indices_obj):
                #     
                #     if self.verify_with_vlm:
                #         if vlm is None:
                #             raise ValueError("VLM is not provided")
                #         
                #         if i > 0:
                #             prev_keystate_obj = state_changed_indices_obj[i-1]
                #         else:
                #             prev_keystate_obj = 0
                #         prev_state, after_state = self.verify_state_change(obj, idx,vlm,prev_keystate_obj,cropped_robot_masks)
                #         if prev_state != after_state:
                #             state_changed_indices.append(idx)
                #             state_change_reasons.append({"object": obj.name, "index": idx * self.downsample_interval,
                #                               "reason": (prev_state, after_state)})
                #             self.keystates_by_object[obj.name].append(idx * self.downsample_interval)
                #             self.object_keystate_scores[obj.name].append(1)
                #             keystate_scores[i] = keystate_scores[i] + 1
                #     
                #     
                #     
                #     else:
                #         state_changed_indices.extend(list(state_changed_indices_obj))
                #         state_change_reasons.extend([{"object": obj.name, "index": i * self.downsample_interval,
                #                                       "reason": (pred_states_nl[i], pred_states_nl[i + 1])} for i in
                #                                      state_changed_indices_obj])
                #         self.keystates_by_object[obj.name].extend(list(state_changed_indices_obj * self.downsample_interval))
                #         self.object_keystate_scores[obj.name].extend(keystate_scores)
                    
                
                
                
        self.keystates = np.array(state_changed_indices) 
        
        self.keystate_scores = np.array(keystate_scores)
        
        self.keystate_reasons = state_change_reasons
        
        self.compute_keystate_reasons_nl()
        
        return self.keystates


        #keystate_preds["state_change"] = np.array(state_changed_indices) * self.downsample_interval

    def compute_keystate_reasons_nl(self):

        nl_reasons = []
        for i, keystate_reason in enumerate(self.keystate_reasons):
            
            obj = keystate_reason["object"]
            before_state, after_state = keystate_reason["reason"]
            
            nl_reasons.append(f"{obj} changed from {before_state} to {after_state}\n")

        self.keystate_reasons_nl = nl_reasons
        
        for obj in self.keystate_reasons_by_object.keys():
            
            for i, keystate_reason in enumerate(self.keystate_reasons_by_object[obj]):
                before_state, after_state = keystate_reason["reason"]
                self.keystate_reasons_by_object_nl[obj].append(f"State changes: {obj} changed from {before_state} to {after_state}\n")
                
            

        
        
    
    
    
    def verify_state_change(self,obj, keystate_index,vlm, prev_keystate_obj,robot_mask = None):
        
        
        # if len (self.keystates_by_object[obj.name]) > 0:
        #     prev_keystate_obj = self.keystates_by_object[obj.name][-1] // self.downsample_interval#
        # else:
        #     prev_keystate_obj = 0

        prev_keystate_obj = max(0,prev_keystate_obj - 2)
        
        cropped_images = obj.cropped_images
        state_retrieved = False
        
        #find min robot robot mask (cropped)
        masks_to_look_at = robot_mask[keystate_index: keystate_index+6]
        
        areas = [np.sum(mask) for mask in masks_to_look_at]
        kernel = np.ones(min(3,len(areas))) / min(3,len(areas))
        areas = np.convolve(areas,kernel,mode = "same")
        
        min_idx = np.argmin(areas)
        
        
        
        before_masks_to_look_at = robot_mask[max(prev_keystate_obj,keystate_index - 18):keystate_index]
        
        areas_before = [np.sum(mask) for mask in before_masks_to_look_at]
        kernel = np.ones(min(3,len(areas_before))) / min(3,len(areas_before))
        areas_before = np.convolve(areas_before,kernel,mode = "same")
        
        
        min_before_idx = np.argmin(areas_before)
        
        
        future_idx = min(keystate_index + min_idx, len(cropped_images) - 1)
        cropped_image_ks = cropped_images[future_idx]
        cropped_image_before = cropped_images[
            max(0,keystate_index - len(before_masks_to_look_at) + min_before_idx)]
        
        obj_state_prompt = obj.get_state_prompt()
        while not state_retrieved:
            try:
                before_state = vlm.get_object_states(cropped_image_before, obj_state_prompt,
                                                        obj.states)
                after_state = vlm.get_object_states(cropped_image_ks, obj_state_prompt,
                                                        obj.states)
                
                if after_state == "unknown":
                    after_state = vlm.get_object_states(cropped_images[max(future_idx+6,len(cropped_images)-1)], obj_state_prompt,
                                                        obj.states)

                state_retrieved = True
            except Exception as e:
                print(e)
                print("Retrying")
                time.sleep(60)
                
        if before_state == after_state:
            plt.imshow(cropped_image_before)
            plt.show()
            plt.imshow(cropped_image_ks)
            plt.show()
            print("No state change")
        if "unknown" in before_state or "unknown" in after_state:
            before_state = "unknown"
            after_state = "unknown"
            print("Unknown state")
        
        
        return before_state, after_state
        # if after_state != before_state:
        #     prompt_nl_reasons += f"{obj} changed its state from {before_state} to {after_state}\n"
    
    
    def get_keystate_reasons(self, combined_keystates, combined_keystates_threshold,keystate_objects):

        matched_keystate_reasons = []
        if self.keystate_reasons_nl is None or len(self.keystate_reasons_nl) == 0:
            return [None] * len(combined_keystates)
        for idx, combined_keystate in enumerate(combined_keystates):
            
            cur_obj = keystate_objects[idx]
            abs_dist = np.abs(combined_keystate -  self.keystates_by_object[cur_obj])
            dist = (combined_keystate -  self.keystates_by_object[cur_obj])
            # more tolreance for state change
            neg_dists = dist[dist <= 0]
            if len(neg_dists > 0) and np.min(abs_dist) >= combined_keystates_threshold:
                min_dist_bwd = np.argmax(neg_dists)
                match_threshold_bwd = -22
                min_dist_bwd_val = neg_dists[min_dist_bwd]
                if min_dist_bwd_val > match_threshold_bwd:
                    min_dist_idx = np.where(dist == min_dist_bwd_val)[0][0]
                    
                    # keystates_to_average.append(other_keystates[min_dist_idx])
                    reason = self.keystate_reasons_by_object_nl[cur_obj][min_dist_idx]
                    
                    matched_keystate_reasons.append(reason)
                else:
                    matched_keystate_reasons.append(self.placeholder_prompt)
            else:
                cur_obj = keystate_objects[idx]
                keystate_dists = np.abs(self.keystates_by_object[cur_obj] - combined_keystate)
                if len(keystate_dists) == 0:
                    matched_keystate_reasons.append(self.placeholder_prompt)
                    continue
                min_dist = np.min(keystate_dists)
                if min_dist < combined_keystates_threshold:
                    # reason = self.keystate_reasons_nl[np.argmin(keystate_dists)]
                    reason = self.keystate_reasons_by_object_nl[cur_obj][np.argmin(keystate_dists)]
                    matched_keystate_reasons.append(reason)
                else:
                    matched_keystate_reasons.append(self.placeholder_prompt)
                    
        return matched_keystate_reasons

        
       