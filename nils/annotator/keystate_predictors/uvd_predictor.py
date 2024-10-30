from __future__ import annotations

from typing import Tuple

import numpy
import numpy as np
import torch
import torchvision.transforms as T
from liv import load_liv
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.signal import argrelextrema
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm
import uvd

from nils.annotator.keystate_predictors.keystate_predictor import (
    KeystatePredictor,
)
from nils.scorers.VoltronKeyStatePredictor import (
    VoltronKeyStatePredictor,
)


def smooth_fn(data):
    kr = KernelRidge(kernel="rbf", gamma=0.08)
    X = np.array(range(0, data.shape[0])).reshape(-1, 1)
    kr.fit(X, data)

    res = kr.predict(X)
    return res


class UVDKeyStatePredictor(KeystatePredictor):

    def __init__(self,name,downsample_interval, device="cpu", backbone="LIV", min_interval=20):
        
        super().__init__(name,downsample_interval)

        if backbone == "LIV":
            self.vision_backbone = load_liv("resnet50")
            self.transform = T.Compose([T.ToTensor()])
        elif backbone =="Voltron":
            #pass
            self.model = VoltronKeyStatePredictor("v-dual",device=device)


        self.model_name =backbone

        self.device = "cpu"
        self.smooth_fn = smooth_fn
        self.min_interval = min_interval
        self.transform = T.Compose([T.ToTensor()])

    def preprocess(self, images):
        pass

    def get_embeddings(self, input_data) -> torch.Tensor:
        x = self.preprocess_liv(input_data)
        x = torch.split(x,64)
        embeddings = []
        with torch.no_grad():
            for batch in x:
                embeddings.append(self.vision_backbone(input=batch.to(self.device), modality="vision"))

            embeddings = torch.cat(embeddings)


        return embeddings.detach().cpu().numpy()

    def predict(self, data) -> numpy.ndarray:
        
        frames = data["batch"]["rgb_static"]
        keystates, distances = self.extract_keyframes_uvd(frames)
        
        self.keystates = keystates
        self.keystate_reasons = distances
        
        return keystates

    def preprocess_liv(self, images):
        images = np.array(images)
        # batch_size = images.shape[0]

        images = torch.stack([self.transform(images[i, ...]) for i in range(images.shape[0])])
        return images

    def preprocess_annotations(self, annotations):
        pass

    def extract_keyframes_uvd(self, frames: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        if self.model_name =="LIV":
            embeddings = self.get_embeddings(frames)
        else:
            images = self.model.preprocess_images(frames)
            images = self.model.preprocess(images)
            embeddings = self.model.get_embeddings(images).detach().cpu().numpy()
        # last frame as the last subgoal
        cur_goal_idx = embeddings.shape[0] - 1
        # saving (reversed) subgoal indices (timesteps)
        goal_indices = [cur_goal_idx]
        cur_emb = embeddings.copy()
        distances = []

        pb = tqdm(total=embeddings.shape[0])


        # L, d
        while cur_goal_idx > self.min_interval:
            # smoothed embedding distance curve (L,)
            d = norm(cur_emb - cur_emb[-1], axis=-1)
            d = self.smooth_fn(d)
            distances.append(d)
            # monotonicity breaks (e.g. maxima)
            extremas = argrelextrema(d, np.greater)[0]
            extremas = [
                e for e in extremas
                if cur_goal_idx - e > self.min_interval
            ]
            if extremas:
                # update subgoal by Eq.(3)
                cur_goal_idx = extremas[-1] - 1
                goal_indices.append(cur_goal_idx)
                cur_emb = embeddings[:cur_goal_idx + 1]
                pb.update(embeddings.shape[0] - cur_goal_idx)
            else:
                break

        new_dists = []

        for i in range(len(distances)):
            if i+1 < len(goal_indices):

                new_dists.append(distances[i][max(goal_indices[i+1]+1-5,0):])
            else:
                new_dists.append(distances[i])

        return goal_indices[::-1], new_dists[::-1]
        return embeddings[
            goal_indices[::-1]  # chronological
        ]

    def plot_uvd(self, goal_indices, distances, ground_truth_indices):


        fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(10,6))

        lens = [dist.shape[0] for dist in distances]
        x = np.array(range(goal_indices[-1]))

        x = np.split(x, (np.array(goal_indices))[:-1])

        #x = np.split(x,lens)
        x = [x_cur + 1 for x_cur in x]
        x = [np.concatenate([np.array(range(max(x_cur[0]-5,0), x_cur[0])),x_cur]) for x_cur in x]
        if goal_indices[0]!= 0:
            x = x[1:]

        #distances = np.split(distances, goal_indices)
        for i, goal_index in enumerate(goal_indices[:-1]):

            ax.plot(x[i], distances[i])
        ax.vlines(x=goal_indices,ymin=0.05, ymax=0.95,linestyles='dotted',colors="black")
        for i,goal_index in enumerate(goal_indices):
            plt.text(goal_index +1, plt.gca().get_ylim()[0], f"{i}", ha='center', va='bottom', color='black',
                     fontsize=10)
        ax.vlines(x= ground_truth_indices,ymin=0.05, ymax=0.95,linestyles='dashed', color=(0.7,0,0,0.5), label="Ground Truth Key States")
        ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
        plt.show()
