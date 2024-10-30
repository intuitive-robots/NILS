import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nils.annotator.keystate_predictors.keystate_predictor import (
    KeystatePredictor,
)
from nils.utils.utils import fill_false_outliers_np



class GripperPosKeystatePredictor(KeystatePredictor):

    def __init__(self, name,downsample_interval=1, n_frames_proximity = 3, offset = 0):
        super(GripperPosKeystatePredictor, self).__init__(name, downsample_interval)

        self.n_frames_proximity = n_frames_proximity
        
        self.offset = 0

        self.max_score_proximity = 8
        
        self.placeholder_prompt = "Gripper proximity: No gripper proximity detected"





    def get_in_contact(self, data):

        gripper_locations = data["object_manager"].robot.gripper_locations

        object_centers = data["object_manager"].get_object_boxes()

        confidences = np.stack(
            [data["object_manager"].objects[obj].confidence for obj in data["object_manager"].objects.keys()], axis=1)

        nan_filter = np.isnan(confidences)

        #object_manager = data["object_manager"]

        #depth_map = data["depth_map"]

        object_boxes = {obj.name: obj.boxes for obj in data["object_manager"].objects.values()}

        objects = list(object_centers.keys())

        centers = np.stack(list(object_centers.values()), axis=1)
        boxes = np.stack(list(object_boxes.values()), axis=1)

        points = gripper_locations[:, np.newaxis, :]
        bboxes = boxes

        # for name,box in object_boxes.items():
        #
        #    dx = max(box[:,0]] - points[:,0])

        # dx =n np.maximum(bboxes[..., [0, 2]] - points[..., 0, np.newaxis], 0)
        dx = np.maximum.reduce(
            [bboxes[:, :, 0] - points[..., 0], points[..., 0] - bboxes[:, :, 2], np.zeros_like(bboxes[:, :, 0])])
        dy = np.maximum.reduce(
            [bboxes[:, :, 1] - points[..., 1], points[..., 1] - bboxes[:, :, 3], np.zeros_like(bboxes[:, :, 1])])

        distances = np.sqrt(dx ** 2 + dy ** 2)

        distances[nan_filter] = np.nan

        smoothed_distances = pd.DataFrame(distances).interpolate(method='linear', axis=0)
        smoothed_distances = smoothed_distances.rolling(3, center=True, min_periods=1).mean().values

        contact_threshold = np.maximum(np.sort(smoothed_distances, axis=0)[4, :], 3) * 2

        contact_threshold = np.minimum(20, contact_threshold)

        in_contact = smoothed_distances < contact_threshold
        in_contact_raw = distances < contact_threshold
        # Get objects in concact for at least self.n_frames_proximity frames:
        # in_contact_filtered =

        # in_contact_filtered = np.stack([filter_consecutive_trues_np(in_contact[:,i], self.n_frames_proximity) for i in range(in_contact.shape[1])],axis = 1)

        in_contact_filtered = np.stack(
            [fill_false_outliers_np(in_contact_raw[:, i], 5) for i in range(in_contact_raw.shape[1])], axis=1)


        return in_contact_filtered


        
    def predict(self, data):

        gripper_locations = data["object_manager"].robot.gripper_locations
        
        object_centers = data["object_manager"].get_object_boxes()
        
        confidences = np.stack([data["object_manager"].objects[obj].confidence for obj in data["object_manager"].objects.keys()], axis = 1)
        
        nan_filter = np.isnan(confidences)
        
        object_manager = data["object_manager"]

        depth_map = data["depth_map"]
        
        object_boxes = {obj.name: obj.boxes for obj in data["object_manager"].objects.values()}
        
        objects = list(object_centers.keys())
        
        centers = np.stack(list(object_centers.values()),axis = 1)
        boxes = np.stack(list(object_boxes.values()),axis = 1)

        points = gripper_locations[:, np.newaxis, :].astype(int)
        bboxes = boxes
        
        #for name,box in object_boxes.items():
        #    
        #    dx = max(box[:,0]] - points[:,0])
            
        
        #dx =n np.maximum(bboxes[..., [0, 2]] - points[..., 0, np.newaxis], 0)
        dx = np.maximum.reduce([bboxes[:,:,0] - points[..., 0], points[..., 0] - bboxes[:,:,2], np.zeros_like(bboxes[:,:,0])])
        dy = np.maximum.reduce([bboxes[:,:,1] - points[..., 1], points[..., 1] - bboxes[:,:,3], np.zeros_like(bboxes[:,:,1])])
        
        
        
        distances = np.sqrt(dx ** 2 + dy ** 2)
        
        distances[nan_filter] = np.nan
        
        
        
        smoothed_distances = pd.DataFrame(distances).interpolate(method='linear', axis=0)
        smoothed_distances = smoothed_distances.rolling(3,center=True,min_periods=1).mean().values
        
        contact_threshold = np.maximum(np.sort(smoothed_distances, axis=0)[4,:],3) * 2

        contact_threshold = np.minimum(20,contact_threshold)
        
        in_contact = smoothed_distances < contact_threshold
        in_contact_raw = distances < contact_threshold
        #Get objects in concact for at least self.n_frames_proximity frames:
        #in_contact_filtered = 
        
        
        
        #in_contact_filtered = np.stack([filter_consecutive_trues_np(in_contact[:,i], self.n_frames_proximity) for i in range(in_contact.shape[1])],axis = 1)
        
        in_contact_filtered = np.stack([fill_false_outliers_np(in_contact_raw[:,i],5) for i in range(in_contact_raw.shape[1])],axis = 1)
        
        keystates = []
        keystate_reasons = []
        keystate_scores = []
        for obj_idx in range(in_contact_filtered.shape[1]):

            if object_manager.objects[objects[obj_idx]].interactable:
                
                cur_obj_in_contact = in_contact_filtered[:,obj_idx].astype(int)
                
                if cur_obj_in_contact[-1] == 1:
                    cur_obj_in_contact[-1] = 0
                
                diff = np.diff(cur_obj_in_contact)
                
                obj_keystates = np.where(diff == -1)[0] +1
                
                
                #interaction_length = np.diff(np.where(diff == -1)[0])
                
                interaction_lengths = np.array(get_seq_lens(diff))
                
                if len(interaction_lengths) < len (obj_keystates):
                    interaction_lengths = np.concatenate([obj_keystates[[0]],interaction_lengths])
                
                assert len(interaction_lengths) == len(obj_keystates)
                

                indices_to_keep = []
                for idx,ks in enumerate(obj_keystates):
                    gripper_depth = depth_map[ks][points[ks,0,1],points[ks,0,0]]
                    mean_obj_depth = (depth_map[ks] * object_manager.objects[objects[obj_idx]].mask[ks])

                    if object_manager.objects[objects[obj_idx]].mask[ks].sum() < 40:
                        mean_obj_depth = 1000
                        diff = 1000
                    else:
                        min_obj_depth = np.min(mean_obj_depth[mean_obj_depth > 0])

                        diff = np.min(np.abs(gripper_depth - mean_obj_depth[mean_obj_depth > 0]))
                    thresh = depth_map.max() / 5
                    if diff < thresh:
                        indices_to_keep.append(idx)


                obj_keystates = obj_keystates[indices_to_keep]
                interaction_lengths = interaction_lengths[indices_to_keep]

                keystate_scores.append(interaction_lengths)
                
                keystates.append(obj_keystates)
                
                
                keystate_reasons.append([objects[obj_idx]] * len(obj_keystates))
                
                self.keystate_reasons_by_object[objects[obj_idx]].extend([objects[obj_idx]] * len(obj_keystates))
                
                self.keystates_by_object[objects[obj_idx]].extend((obj_keystates * self.downsample_interval) - self.offset)
                self.object_keystate_scores[objects[obj_idx]].extend(interaction_lengths)
                
        
        keystate_scores = np.concatenate(keystate_scores)
        if len(keystate_scores) == 0:
            self.keystates = np.array([])
            self.keystate_reasons = np.array([])
            self.keystate_scores = np.array([])
            return np.array([])
        keystate_scores = (keystate_scores) / (np.max(keystate_scores) + 1e-6)
        self.keystate_scores = keystate_scores
        
        for obj in self.object_keystate_scores.keys():
            self.object_keystate_scores[obj] = np.array(self.object_keystate_scores[obj])
            if len(self.object_keystate_scores[obj]) > 0:
                self.object_keystate_scores[obj] = np.minimum((self.object_keystate_scores[obj]) / self.max_score_proximity,1.)
        
        self.keystates = (np.concatenate(keystates) * self.downsample_interval) + self.offset 
        self.keystate_reasons = np.concatenate(keystate_reasons)
        
        self.compute_keystate_reasons_nl()

        
        return self.keystates

    



    def compute_keystate_reasons_nl(self):
        
        nl_reasons = []
        for i, keystate_reason in enumerate(self.keystate_reasons):
            nl_reasons.append("The robot gripper was close to the {}".format(keystate_reason))
        
        self.keystate_reasons_nl = nl_reasons
        
        for obj in self.object_keystate_scores.keys():
            self.keystate_reasons_by_object_nl[obj] = ["Gripper proximity: The robot gripper was close to the {}".format(obj)] * len(self.object_keystate_scores[obj])






def plot_distances(dist,contact_thresh,object_name):
    # scaler = StandardScaler()
    # distances_scaled = scaler.fit_transform(distances)
    # model = IsolationForest(contamination=0.01)
    # res = model.fit_predict(distances_scaled[:, -2].reshape(-1, 1))

    df = pd.DataFrame(dist)
    
    # visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    in_contact = dist < contact_thresh
    in_contact = fill_false_outliers_np(in_contact,5)
    
    a = df.loc[in_contact]  # anomaly
    ax.plot(df.index, df[0], color='black', label='Distances')
    ax.scatter(a.index, a[0], color='red', label='Contact Points')
    plt.legend()
    plt.title(object_name)
    plt.show()


def get_seq_lens(arr):
    sequences = []
    current_length = 0
    inside_sequence = False

    for num in arr:
        if num == 1:
            if inside_sequence:
                sequences.append(current_length)
                current_length = 0
            inside_sequence = True
        elif num == -1:
            if inside_sequence:
                sequences.append(current_length)
                current_length = 0
                inside_sequence = False
        elif num == 0:
            if inside_sequence:
                current_length += 1

    return sequences