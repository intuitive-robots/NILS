import logging
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

pnp_tasks = ["put", "pick", "take", "pnp", "icra", "rss", "many_skills", "lift_bowl", "move_drying", "wipe",
             "right_pepper", "topple", "upright", "flip", "set_table"]


class BridgeTest(Dataset):

    def __init__(self, path, sampling_rate=1, name="bridge_test", n_jobs=-1, task_number=0):
        self.path = path
        self.sampling_rate = sampling_rate
        self.name = name

        self.paths = self.get_traj_paths()

        if n_jobs > 1:
            n_trajs = len(self.paths)
            n_trajs_per_job = n_trajs // n_jobs
            start_idx = task_number * n_trajs_per_job
            end_idx = (task_number + 1) * n_trajs_per_job
            logging.info(f"Task {task_number} will process from {start_idx} to {end_idx}")
            self.paths = self.paths[start_idx:end_idx]


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        ret_dict = self.load_frames(path)

        return ret_dict

    def load_frames(self, folder):

        short_horizon_trajectories = os.listdir(folder)
        
        short_horizon_trajectories = [traj for traj in short_horizon_trajectories if "traj" in traj and "annotation" not in traj]
        

        n_trajs = len(short_horizon_trajectories)
        
        if n_trajs == 1:
            return None

        short_horizon_trajectories_idx = [int(traj.split("traj")[-1]) for traj in short_horizon_trajectories]
        short_horizon_trajectories_idx = np.argsort(short_horizon_trajectories_idx)
        short_horizon_trajectories = [short_horizon_trajectories[i] for i in short_horizon_trajectories_idx]

        short_horizon_trajectories = short_horizon_trajectories[:n_trajs]

        long_horizon_frames = []

        cur_dict = {"rgb_static": [], "actions": []}
        frame_idx = 0
        keystates = []
        paths = []
        frame_names = []
        for short_horizon_traj in short_horizon_trajectories[:10]:

            short_horizon_traj_path = os.path.join(folder, short_horizon_traj)
            paths.append(short_horizon_traj_path)
            frames_dir = os.path.join(short_horizon_traj_path, "images0")

            frames = os.listdir(frames_dir)
            frame_numbers = [int(frame.split("im_")[-1].split(".jpg")[0]) for frame in os.listdir(frames_dir)]
            frames_idx = np.argsort(frame_numbers)
            frames = np.array([frames[i] for i in frames_idx])

            n_frames_per_traj = len(frames) // self.sampling_rate
            frame_indices = np.linspace(0, len(frames) - 1, n_frames_per_traj).astype(int)

            keystates.append(frame_idx + len(frame_indices))
            
            for cur_idx, frame in enumerate(frames[frame_indices]):
                frame_path = os.path.join(frames_dir, frame)
                frame_image = np.array(Image.open(frame_path))
                long_horizon_frames.append(frame_image)
                frame_names.append(f"{short_horizon_traj}:{frame}")
                cur_dict["rgb_static"].append(frame_image)
                frame_idx += 1


            with open(os.path.join(short_horizon_traj_path, "obs_dict.pkl"), "rb") as f:
                state = pickle.load(f)["state"]

            state = np.array(state)
            state = state[frame_indices]
            gripper_closed = state[:, -1] < 0.7

            gripper_states = np.ones_like(state[:, -1])
            gripper_states[gripper_closed] = -1

            state[:, -1] = gripper_states

            cur_dict["actions"].append(state)

        cur_dict["rgb_static"] = np.stack(cur_dict["rgb_static"])
        cur_dict["gripper_actions"] = np.concatenate(cur_dict["actions"], axis=0)[:, -1]
        cur_dict["path"] = paths
        cur_dict["frame_names"] = frame_names
        cur_dict["keystates"] = keystates

        return cur_dict

    def get_traj_paths(self):
        base_path = Path(self.path)
        folders = []
        for lang_file in tqdm(base_path.rglob('images0'), desc="Finding image folders"):
            parent_dir = lang_file.parent
            if not os.path.exists(os.path.join(parent_dir, "images0")):
                continue
            folders.append(str(lang_file.parent.parent))

        return folders
