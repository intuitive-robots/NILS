import numpy as np


class KeystatePredictor:
    def __init__(self, cfg,downsample_interval):
        self.cfg = cfg
        self.downsample_interval = downsample_interval
        
        self.keystates = None
        self.keystate_reasons = None

    def predict(self, batch, scene_graphs=None, labeled_gripper_cam_data=None):
        raise NotImplementedError("Subclasses should implement this method.")
    
    



class ObjectMovementKeystatePredictor(KeystatePredictor):
    def predict(self, data):

        batch = data["batch"]
        scene_graphs = data["scene_graphs"]
        
        scene_graphs_subset = np.linspace(0, len(batch["rgb_static"]) - 1,
                                          len(batch["rgb_static"]) // self.downsample_interval).astype(np.int32)
        batch_subset = {key: batch[key][scene_graphs_subset] for key in batch.keys()}
        
        

        # robot_pos_screen = get_tcp_center_screen(batch_subset, self.env)
        robot_pos_screen = None
        gripper_actions = batch_subset["actions"][:, -1]
        object_movement_keystates = self.detect_movement(scene_graphs, robot_pos_screen, gripper_actions)

        moved_objects = [obj for obj, frame in object_movement_keystates]
        object_movement_dict = object_movement_keystates
        object_movement_keystates = np.array([ks[1] for ks in object_movement_keystates])
        object_movement_keystates = (object_movement_keystates * self.downsample_interval)
        
        
        self.keystates = object_movement_keystates
        self.keystate_reasons = list(object_movement_dict.items())
        
        return object_movement_keystates
    

    def detect_movement(self, scene_graphs, gripper_actions):

        gripper_close_frames = [i for i, action in enumerate(gripper_actions) if action == -1]
        gripper_close_frames = np.array(gripper_close_frames)

        all_flows = np.concatenate([sg.flow_raw for sg in scene_graphs if sg.flow_raw is not None], axis=0)

        # No flow for last frame
        all_flows = np.concatenate([all_flows, np.zeros_like(all_flows[0][None, ...])], axis=0)

        object_movements = self.object_manager.get_object_movements(all_flows)

        keystates = []
        moved_objects = []
        max_movement = []
        for obj, movement in object_movements.items():
            # Only consider movement where gripper is closed:

            if gripper_close_frames is not None and self.cfg.gripper_close_movement_filter:
                movement[~gripper_close_frames] = 0
            over_threshold = movement > self.cfg.movement_threshold
            changes = np.diff(over_threshold.astype(int))
            start_indices = np.where(changes == 1)[0] + 1
            end_indices = np.where(changes == -1)[0]

            if over_threshold[-1]:
                end_indices = np.append(end_indices, len(over_threshold) - 1)

            if over_threshold[0]:
                start_indices = np.insert(start_indices, 0, 0)

            # Filter short movements:
            for start_indices, end_indices in zip(start_indices, end_indices):
                if end_indices - start_indices > self.cfg.n_frames_movement:
                    keystates.append((obj, end_indices + 1))
                    moved_objects.append(obj)
                    max_movement.append(np.max(movement[start_indices:end_indices]))

        return keystates

    
