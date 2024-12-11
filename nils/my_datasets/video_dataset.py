import logging
import os
import pickle

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob

class VideoDataset(Dataset):
    def __init__(self, path, sampling_rate=1, name=None, n_jobs = -1, task_number=0):
        self.path = path


        video_files_in_path = glob.glob(os.path.join(path, "*.mp4"))
        self.trajectories = []
        self.paths = video_files_in_path
        
        
        n_jobs = min(n_jobs, len(self.paths))
        
                
        if n_jobs > 1:
            n_trajs = len(self.paths)
            n_trajs_per_job = n_trajs // n_jobs
            start_idx = task_number * n_trajs_per_job
            end_idx = (task_number + 1) * n_trajs_per_job
            logging.info(f"Task {task_number} will process from {start_idx} to {end_idx}")
            self.paths = self.paths[start_idx:end_idx]
        
        self.frame_names = []
        
        
        for f in self.paths:
            print(f"Found video file: {f}")

            cap = cv2.VideoCapture(os.path.join(path, f))
            frames = []
            frame_names = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (frame.shape[1] //2, frame.shape[0] //2))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                frames.append(frame)
                frame_names.append(f"{frame_idx}.jpg")
                frame_idx += 1


            frames = np.array(frames)

            print(f"Loaded {len(frames)} frames")

            self.sampling_rate = sampling_rate

            self.frames = np.stack(frames)[::self.sampling_rate]
            self.frame_names.append(frame_names[::self.sampling_rate])

            self.trajectories.append(self.frames)
        

        self.name = name
        

    def __getitem__(self, idx):
        
        
        frames = self.trajectories[idx]
        
        data = {}
        data["rgb_static"] = frames
        data["path"] = np.array([self.paths[idx]] * len(frames))
        data["frame_names"] = np.array(self.frame_names[idx])

        return data

    def __len__(self):
        return len(self.trajectories)
