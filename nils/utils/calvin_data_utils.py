import os

import numpy as np
from tqdm import tqdm


def get_tcp_center_screen(batch,env):
    tcp_pos_world, tcp_rot_world = batch["robot_obs"][:, :3], batch["robot_obs"][:, 3:6]

    tcp_pos_world_ones = np.c_[np.array(tcp_pos_world), np.ones(len(tcp_pos_world))]

    # tcp_center_gripper_screen = self.env.cameras[camera_id].project(np.append(tcp_pos_world_ones, 1))
    tcp_center_gripper_screen = env.cameras[0].project(tcp_pos_world_ones.T)

    tcp_center_gripper_screen = [np.array([x, y]) for x, y in
                                 zip(tcp_center_gripper_screen[0], tcp_center_gripper_screen[1])]
    tcp_center_gripper_screen = np.stack(tcp_center_gripper_screen)

    return tcp_center_gripper_screen


def filter_close_elements(arr, threshold):
    """Filter elements in the array that are close to each other based on a threshold.

    Parameters:
    - arr (np.ndarray): Input NumPy array.
    - threshold (float): Threshold value to determine closeness.

    Returns:
    - np.ndarray: Filtered array.
    """
    if len(arr) == 0:
        return arr, []
    filtered_indices = [0]  # Start with the first element
    for i in range(1, len(arr)):
        if np.abs(arr[i] - arr[filtered_indices[-1]]) > threshold:
            filtered_indices.append(i)
    return arr[filtered_indices], filtered_indices




def get_sample_data(calvin_root, problem_idx=0, custom_annotations=None, additional_annotations=None):
    annotations = np.load(
        os.path.join(calvin_root, "dataset/calvin_debug_dataset/validation/lang_annotations/auto_lang_ann.npy"),
        allow_pickle=True).item()

    sample_idx = 0
    save_video = False

    # Todo: Channging the descriptions alters the results.. Bug in scoring?
    min_idx = 9999999999999
    max_idx = 0
    for idx in annotations['info']['indx']:
        min_idx = min(min_idx, idx[0])
        max_idx = max(max_idx, idx[1])

    sample_annotation_idx = annotations['info']['indx'][problem_idx]
    # sample_annotation_idx = (min_idx,max_idx)
    annotation = annotations['language']['ann'][problem_idx]
    possible_annotations = annotations['language']['ann']

    # possible_annotations = ["lift up the red block from the table","lifting up the red block from the table","picking up the red block","turn on the lightbulb","turning on the lightbulb"]

    if custom_annotations != None:
        possible_annotations = ["lifting up the red block from the table"]
    elif additional_annotations != None:
        possible_annotations += additional_annotations

    # frames = [np.load(os.path.join(calvin_root, f"dataset/calvin_debug_dataset/validation/lang_annotations/0{idx}.npz"),allow_pickle=True).item() for idx in range(sample_annotation_idx[0],sample_annotation_idx[1])]
    frames = []
    for idx in tqdm(range(sample_annotation_idx[0], sample_annotation_idx[1] + 128)):
        try:
            frames.append(
                np.load(os.path.join(calvin_root, f"dataset/calvin_debug_dataset/validation/episode_0{idx}.npz"),
                        allow_pickle=True)['rgb_static'])
        except Exception:
            pass

    return frames, annotation, possible_annotations
