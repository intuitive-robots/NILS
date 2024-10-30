import logging
import math
import os

import numpy as np
import tensorflow as tf

tf.data.experimental.enable_debug_mode()

import tensorflow_datasets as tfds

import torch
from torch.utils.data import IterableDataset

short_horizon_datasets = [
    "fractal20220817_data,jaco_play,viola,berkeley_autolab_ur5,"
    "stanford_hydra_dataset_converted_externally_to_rlds,"
    "austin_buds_dataset_converted_externally_to_rlds,"
    "ucsd_kitchen_dataset_converted_externally_to_rlds,"
    "#austin_sailor_dataset_converted_externally_to_rlds,"
    "bc_z,"
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds,"
    "uiuc_d3field,cmu_play_fusion"]


short_horizon_datasets = ["fractal20220817_data"]


class OpenXDataset(IterableDataset):

    def __init__(self, dataset_dir="gs://gresearch/robotics/",
                 dataset="fractal20220817_data", num_frames=32, random_sampling=False,
                 sampling_interval=4, image_key="image", observation_cfg=None,task_number = 1,n_jobs = 1,
                 n_torch_workers = None,n_samples = 100, offset = 0):
        # _ = tfds.load(dataset, data_dir=dataset_dir)
        print(f"saving dataset to {dataset_dir}")

        concat_ds = None


        n_trajs = n_samples
        self.n_trajs = n_trajs
        per_worker = n_trajs / n_jobs
        self.n_trajs_local = per_worker
        worker_info = torch.utils.data.get_worker_info()
        self.n_jobs = n_jobs
        self.task_number = task_number


        self.offset = offset



        if n_jobs > 9999999:
            total_n_traj = n_trajs
            n_traj_per_job = total_n_traj // n_jobs
            start_idx = (task_number - 1) * n_traj_per_job
            end_idx = task_number * n_traj_per_job
            logging.info(f"Task {task_number} will process {start_idx} to {end_idx}")
        else:
            start_idx = 0
            end_idx = n_trajs

        self.ds = None
        path = os.path.join(dataset_dir, short_horizon_datasets[0], "0.1.0")


        #ds = tfds.builder_from_directory(path)

        #self.feature_dict = mod_features(ds.info.features)

        self.ds_path = path
        
        path = os.path.join(dataset_dir, short_horizon_datasets[0], "0.1.0")

        self.ds_path = path
        if n_torch_workers is None:

            for ds_name in short_horizon_datasets:
                path = os.path.join(dataset_dir, ds_name, "0.1.0")

                self.ds_path = path

                #ds = tfds.builder_from_directory(path)


                #if concat_ds is None:
                #    ds = ds.as_dataset(split=f"train[{start_idx}:{end_idx}]")
                #    concat_ds = ds
                #else:
                #    ds = ds.as_dataset(split=f"train[{start_idx}:{end_idx}]")
                #    concat_ds = concat_ds.concatenate(ds)




            # ds = tf.data.Dataset.from_tensor_slices(concat_ds)
            #
            # # 2. extract all elements from datasets and concat them into one dataset
            # concat_ds = ds.interleave(
            #     lambda x: x,
            #     cycle_length=1,
            #     num_parallel_calls=tf.data.AUTOTUNE,
            # )
            #concat_ds = concat_ds.map(
            #    self.episode2steps, num_parallel_calls=tf.data.AUTOTUNE)

            #self.ds = concat_ds



        self.observation_cfg = observation_cfg

        self.sampling_interval = sampling_interval

        #self.ds = ds.map(
        #    self.episode2steps, num_parallel_calls=tf.data.AUTOTUNE)

        self.observation_key = "observation"
        self.image_key = image_key
        self.language_key = "language_instruction"
        self.action_key = "action"
        self.depth_key = "depth"

        self.path = os.path.join("/home/DATA_SHARE/",short_horizon_datasets[0])
        self.name = "_".join(short_horizon_datasets)

    def step_map_fn(self, step):
        return {
            'observation': {
                'image': step['observation']['image'],
            },
            'annotation': step["langauge_instruction"]
        }

    def episode2steps(self, episode):
        return episode['steps']

    def __len__(self):
        return len(self.ds) if self.ds is not None else self.n_trajs_local

    def get_NILS_format_ds(self, ds):

        def get_NILS_format_trajectory(episode):
            indices = np.linspace(0, len(episode) - 1, len(episode) // self.sampling_interval, dtype=int)
            print(len(episode))
            rgb_static = []
            lang_ann = []
            robot_positions = []
            gripper_actions = []
            depth_static = []

            print(episode)

            for step_idx, step in enumerate(episode):
                if step_idx not in indices:
                    continue
                rgb_static.append(step["observation"]["image"].numpy())
                lang_ann.append(step["language_instruction"].numpy().decode("utf-8"))
                if self.observation_cfg.gripper_close_key is not None:
                    gripper_actions.append(step["observation"]["gripper_closed"].numpy())

            rgb_static = np.stack(rgb_static)
            if len(robot_positions) > 0:
                robot_positions = np.stack(robot_positions)
            if len(gripper_actions) > 0:
                gripper_actions = np.stack(gripper_actions)
            if len(depth_static) > 0:
                depth_static = np.stack(depth_static)
            return {"rgb_static": rgb_static, "lang_ann": lang_ann,
                    "gripper_actions": gripper_actions, "depth_static": depth_static}

        def filter_by_index(idx, val):
            return idx in val

        def episode_map_fn(episode):

            indices = np.linspace(0, len(episode) - 1, len(episode) // self.sampling_interval, dtype=int)

            # filter episode["steps"] to only include the steps at the indices

            episode["steps"] = episode["steps"].enumerate().filter(filter_by_index)

            episode["steps"] = episode["steps"].map(get_NILS_format_trajectory)

            return episode

        return ds.map(episode_map_fn)



    def init_ds(self, path, start_idx, end_idx):
        ds = tfds.builder_from_directory(path)

        self.feature_dict = mod_features(ds.info.features)

        ds = ds.as_dataset(split=f"train[{start_idx+ self.offset}:{end_idx}]",
                           shuffle_files=False)

        #ds = ds.map(
        #    self.episode2steps, num_parallel_calls=tf.data.AUTOTUNE).prefetch(32)

        self.ds = ds

        return ds


    def __iter__(self):

        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is not None:

        #     len_ds = self.n_trajs
        #     per_worker = int(math.ceil(len_ds / float(worker_info.num_workers)))
        #     self.n_trajs_local = per_worker
        #     worker_id = worker_info.id

        #     start_idx = worker_id * per_worker
        #     end_idx = min(start_idx + per_worker, len_ds)

        #     print(f"Worker {worker_id} will process {start_idx} to {end_idx}")
        #     ds = self.init_ds(self.ds_path, start_idx, end_idx)
        # else:
        len_ds = self.n_trajs
        per_worker = int(math.ceil(len_ds / float(self.n_jobs)))
        worker_id = self.task_number
        start_idx = worker_id * per_worker
        end_idx = min(start_idx + per_worker, len_ds)

        ds = self.init_ds(self.ds_path, start_idx, end_idx)

        try:
            for idx, cur_ep in enumerate(self.ds):
                if len(cur_ep) < 8:
                    indices = np.linspace(0, len(cur_ep["steps"]) - 1,8, dtype=int)
                else:

                    indices = np.linspace(0, len(cur_ep["steps"]) - 1, 16, dtype=int)

                rgb_static = []
                lang_ann = [] 
                robot_positions = []
                gripper_actions = []
                depth_static = []

                orig_dicts = [step for step in cur_ep["steps"]]

                yield stack_dicts(orig_dicts)


                # for step_idx, step in enumerate(cur_ep["steps"]):

                #     orig_dicts.append(step)

                # #     if step_idx not in indices:
                # #         continue
                # #     rgb_static.append(step[self.observation_key][self.image_key].numpy())
                # #     lang_ann.append(step[self.observation_key]["natural_language_instruction"].numpy().decode("utf-8"))
                # #     if "jaco" in self.name:
                # #         robot_positions.append(step[self.observation_key]["end_effector_cartesian_pos"].numpy())
                # #         gripper_actions.append(step["action"]["gripper_closedness_action"].numpy())
                # #     else:
                # #         robot_positions.append(step[self.observation_key]["base_pose_tool_reached"].numpy())
                # #         gripper_actions.append(step["observation"]["gripper_closed"].numpy())

                # # rgb_static = np.stack(rgb_static)
                # # robot_positions = np.stack(robot_positions)

                # # if "kuka" in self.name:
                # #     rgb_static =  F.center_crop(torch.tensor(rgb_static).permute(0,3,1,2), (int(rgb_static.shape[1]*0.7),int(rgb_static.shape[2]*0.6))).permute(0,2,3,1).numpy()

                # # if len(gripper_actions) > 0:
                # #     gripper_actions = np.stack(gripper_actions)
                # #     gripper_actions[gripper_actions > 0.5] = 1
                # #     gripper_actions[gripper_actions <= 0.5] = 0

                # yield {
                #     # "rgb_static": rgb_static, "lang_ann": lang_ann,
                #     #    "gripper_actions": gripper_actions, "keystates": [len(rgb_static)],
                #     #    "gripper_locations": robot_positions,
                #        "orig_dicts": orig_dicts}
        except Exception as e:
            print(e)
            yield None

    def get_single_problem(self, problem_idx):
        annotation = next(iter(self.ds[problem_idx]))[self.language_key].numpy().decode("utf-8")

        frames = torch.stack(
            [torch.tensor(step[self.observation_key][self.image_key].numpy(), dtype=torch.uint8) for step in
             self.ds[problem_idx]])

        return frames, annotation

    def get_possible_annotations(self):

        if "austin_buds_dataset_converted_externally_to_rlds" in self.dataset:
            return ["Take the lid off the pot", "Put the pot on the plate", "Push the pot to the front of the table"]
        annotations_dir = os.path.join(self.dataset_dir, self.dataset, "possible_annotations.npy")

        if os.path.exists(annotations_dir):
            return np.load(annotations_dir)
 

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    len_ds = len(dataset)
    per_worker = int(math.ceil(len_ds / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.ds = dataset.ds.skip(worker_id * per_worker).take(per_worker)

    return dataset



def mod_features(features):
    """Modifies feature dict."""



    obs_dict = dict(features["steps"]["observation"])
    obs_dict["language_instruction_NILS_0"] = tfds.features.Text()
    obs_dict["language_instruction_NILS_1"] = tfds.features.Text()
    obs_dict["language_instruction_NILS_2"] = tfds.features.Text()



    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            key: obs_dict[key] for key in obs_dict.keys()
                        }
                    ),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation",)
                    },
                }
            ),
            **{key: features[key] for key in features.keys() if key not in ("steps",)},
        }
    )

def episode2steps(episode):
    return episode['steps']



def stack_dicts(dict_list):
    def recurse_and_stack(key, values):
    # If values are dictionaries, recurse deeper
        if isinstance(values[0], dict):
            return {sub_key: recurse_and_stack(sub_key, [val[sub_key] for val in values]) 
                    for sub_key in values[0]}
        else:
            # Stack the values for the current key
            return np.stack(values)

    # Initialize the output dictionary
    stacked_dict = {}
    
    # Loop through each key in the first dictionary
    for key in dict_list[0]:
        # Collect all the values for the current key across all dictionaries
        values = [d[key] for d in dict_list]
        
        # Recurse into the dict if necessary and stack the results
        stacked_dict[key] = recurse_and_stack(key, values)

    return stacked_dict


def to_numpy(dict_in):
    """Recursively converts all tensors to numpy arrays."""
    for key in dict_in:
        if isinstance(dict_in[key], dict):
            dict_in[key] = to_numpy(dict_in[key])
        else:
            dict_in[key] = dict_in[key].numpy()
    return dict_in

def get_NILS_format_batch(batch_orig,n_frames = 16):

    n_steps = len(batch_orig["observation"]["image"])


    if n_steps < 8:
        indices = np.linspace(0, n_steps - 1,n_frames, dtype=int)
    else:

        indices = np.linspace(0, n_steps - 1, n_frames, dtype=int)





    rgb_static= batch_orig["observation"]["image"][indices]
    lang_ann = batch_orig["observation"]["natural_language_instruction"][0].decode("utf-8")
    robot_positions = batch_orig["observation"]["base_pose_tool_reached"][indices]
    gripper_actions = batch_orig["observation"]["gripper_closed"][indices]


    if len(gripper_actions) > 0:
        gripper_actions[gripper_actions > 0.5] = 1
        gripper_actions[gripper_actions <= 0.5] = 0


    return {"rgb_static": rgb_static, "lang_ann": lang_ann,
            "gripper_actions": gripper_actions, "gripper_locations": robot_positions,
            "keystates": [len(rgb_static)]}





def setup_ds(n_processes,task_id,n_trajs,ds_path,bsz=16,offset = 0):
    
    def episode2steps(episode):
        return episode['steps']

    import math
    import tensorflow_datasets as tfds
    import tensorflow as tf
    
    len_ds = n_trajs
    per_worker = int(math.ceil(len_ds / float(n_processes)))
    worker_id = task_id
    start_idx = worker_id * per_worker
    end_idx = min(start_idx + per_worker, len_ds)

    start_idx = start_idx + offset
    
    ds = tfds.builder_from_directory(ds_path)

    options = tf.data.Options()
    options.autotune.enabled = True
    options.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.OFF)
    options.experimental_optimization.inject_prefetch = False



    read_config = tfds.ReadConfig(try_autocache=False,
                                  override_buffer_size=1024)

    ds = ds.as_dataset(split=f"train[{start_idx}:{end_idx}]",
                        shuffle_files=False,read_config=read_config)
    

    ds = ds.with_options(options)
    ds = ds.map(episode2steps)

    return ds
