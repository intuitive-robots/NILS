import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from nils.annotator.keystate_predictors.keystate_predictor import (
    KeystatePredictor,
)
from nils.my_datasets.calvin_stream_dataset import (
    taco_objects_detection_map,
)


class GripperCamKeystatePredictor(KeystatePredictor):
    def predict(self, data):
        labeled_gripper_cam_data = data["labeled_gripper_cam_data"]

        bg_indices = [taco_objects_detection_map[obj] for obj in self.cfg.background_objects]

        gripper_cam_keystates_monotonicity, gripper_cam_keystates_monotonicity_objects = self.get_gripper_cam_keystates_monotonicity(
            labeled_gripper_cam_data, bg_indices)

        self.keystates = gripper_cam_keystates_monotonicity
        self.keystate_reasons = gripper_cam_keystates_monotonicity_objects

    def get_gripper_cam_keystates_monotonicity(self, labeled_gripper_cam_data, bg_indices):

        all_objects = np.unique(np.concatenate(
            [np.array(list(frame.keys())) for frame in labeled_gripper_cam_data if len(frame.keys()) > 0]))

        all_objects = np.array([obj for obj in all_objects if obj not in bg_indices])
        objects = {obj: {"masks": [], "scores": [], "areas": []} for obj in all_objects}

        for frame in labeled_gripper_cam_data:
            for obj in all_objects:
                if obj not in frame.keys():
                    objects[obj]["masks"].append(None)
                    objects[obj]["scores"].append(None)
                    objects[obj]["areas"].append(None)
                else:
                    objects[obj]["masks"].append(frame[obj]["mask"])
                    objects[obj]["scores"].append(frame[obj]["confidence"].item())
                    objects[obj]["areas"].append(frame[obj]["area"].item())

        rolling_window = 4
        min_periods = 1

        keystates = []
        keystate_objects = []

        for obj in objects.keys():
            obj_df = pd.DataFrame(objects[obj])
            obj_df.scores = obj_df.scores.interpolate(limit=2)
            obj_df.areas = obj_df.areas.interpolate(limit=2)

            nan_mask = obj_df.scores.isna()

            obj_df["scores_new"] = obj_df.scores.rolling(rolling_window, center=True, min_periods=min_periods).mean()
            obj_df["areas_new"] = obj_df.areas.rolling(rolling_window, center=True, min_periods=min_periods).mean()

            scores = np.array(obj_df.scores_new)
            areas = np.array(obj_df.areas_new)

            normalized_areas = minmax_scale(areas)
            normalized_scores = minmax_scale(scores)

            combined_scores = normalized_areas * 0.8 + normalized_scores * 0.2

            combined_scores[np.where(combined_scores < 0.01)] = None

            combined_scores = pd.DataFrame(combined_scores, columns=["score"])

            # Select non nan values
            combined_scores[nan_mask] = 0

            combined_scores['id'] = (combined_scores.score.diff() < -0.03).cumsum()
            combined_scores = combined_scores[~nan_mask]

            n = 8
            change_indices = np.array(combined_scores.groupby('id').apply(
                lambda group: group.index[-1] if len(group) >= n else None).dropna().astype(int))

            keystates.extend(change_indices + 2)

            keystate_objects.extend([obj] * len(change_indices))

        sorted_idx = np.argsort(keystates)
        keystates = np.array(keystates)[sorted_idx]
        keystate_objects = np.array(keystate_objects)[sorted_idx]

        return keystates, keystate_objects

    def get_gripper_cam_keystates(batch, labeled_gripper_cam_data, bg_indices, window_size=5, max_miss_count=2,
                                  visible_threshold=10):
        frames = labeled_gripper_cam_data

        keystates = []

        visibility = {}

        keystate_reasons = []

        for i in range(window_size, len(frames)):
            window_frames = frames[i - window_size:i]

            for obj_index in window_frames[0]:
                if obj_index in bg_indices:
                    continue

                visible_counts = sum(1 for frame in window_frames if obj_index in frame)

                states = [frame.get(obj_index, {}).get('state') for frame in window_frames]

                states = [state for state in states if state is not None
                          ]

                if visible_counts >= window_size - max_miss_count:
                    if obj_index not in visibility:
                        visibility[obj_index] = (i - window_size, i)
                    else:
                        visibility[obj_index] = (visibility[obj_index][0], i)

                elif obj_index in visibility:
                    # Object is not visible anymore
                    if visibility[obj_index][-1] - visibility[obj_index][0] >= visible_threshold:
                        # get last visible frame
                        keystates.append(visibility[obj_index][1])
                        keystate_reason = {"object": obj_index, "reason": "visibility", "state": None}
                        keystate_reasons.append(keystate_reason)

                    del visibility[obj_index]
                if len(states) > 2 and np.unique(states).shape[0] > 1 and np.all(
                        np.unique(states, return_counts=True)[1][0] >= 2):
                    # get last visible frame
                    state_change_frame = np.where(np.diff(states))[0][-1]
                    keystates.append(i - window_size + state_change_frame)
                    keystate_reason = {"object": obj_index, "reason": "state_change",
                                       "state": states[state_change_frame]}
                    keystate_reasons.append(keystate_reason)

                    del visibility[obj_index]

        return keystates, keystate_reasons
