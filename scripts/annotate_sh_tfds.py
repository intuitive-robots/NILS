import logging
import os
import warnings
from multiprocessing import Pool, Queue
import hydra
import numpy as np

from nils.annotator.io_utils import save_annotated_frames

UNDEFINED_TASK = "Undefined"
warnings.filterwarnings('ignore')


queue = Queue()

GPU_IDS = [0,1,2,3]
PROC_PER_GPU = 2
BSZ = 16

NUM_GPUS = len(GPU_IDS)




def setup_logger(task_id):
    formatter = logging.Formatter(f'%(asctime)s - {task_id} - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    return logger

def to_numpy(dict_in):
    """Recursively converts all tensors to numpy arrays."""
    for key in dict_in:
        if isinstance(dict_in[key], dict):
            dict_in[key] = to_numpy(dict_in[key])
        else:
            dict_in[key] = dict_in[key].numpy()
    return dict_in


#@hydra.main(version_base=None, config_path="../conf/annotator", config_name="eval_fractal")
def annotate(cfg):

    task,gpu_id = queue.get()
    
    try:

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        #task = cfg.task
        #n_jobs = cfg.n_jobs

        n_jobs = NUM_GPUS * PROC_PER_GPU

        # task = 4

        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from nils.my_datasets.open_x_dataset import OpenXDataset
        from nils.specialist_models.llm.google_cloud_api import get_objects_from_annotation_parallel
        from nils.specialist_models.llm.google_cloud_api import get_object_task_dicts_parallel

        from nils.annotator.KeyStateAnnotatorRefactored import (
            KeyStateAnnotator,)
        from nils.my_datasets.open_x_dataset import get_NILS_format_batch
        from nils.specialist_models.llm.google_cloud_api import get_tasks_parallel
        from nils.my_datasets.open_x_dataset import stack_dicts
        import tensorflow as tf
        from scripts.experiments.label_fractal import setup_ds







        logging.info(f"Available GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

        logging.info(f"CUDA Device Count: {torch.cuda.device_count()}")

        task_id = task
        
        logging.info(f"Task ID: {task_id}")
        logger = setup_logger(task_id)


        num_workers = NUM_GPUS * PROC_PER_GPU
        n_jobs = num_workers

        bsz = BSZ

        offset = 5000
        n_trajs = 87211

        ds = OpenXDataset(n_samples = n_trajs,task_number=task_id, n_jobs=n_jobs,
                          offset = offset)

    
        ds_tf = setup_ds(n_jobs,task_id,ds_path=ds.ds_path,n_trajs=n_trajs,offset = offset)
        ds_len = len(ds_tf)

        annotator = KeyStateAnnotator(cfg, initialize_models=True, ds=ds,sh_detection=True)

        user_home = os.path.expanduser("~")

        task_list = None
        object_list = None

        if cfg.task_list is not None:
            with open (cfg.task_list,"r") as f:
                task_list = f.readlines()
        if cfg.object_list is not None:
            with open (cfg.object_list,"r") as f:
                object_list = f.readlines()
        
        out_dir = cfg.out_dir

        annotated_indices = os.listdir(out_dir)
        annotated_indices = [int(idx) for idx in annotated_indices]

        ds_iter = iter(ds_tf)
            
        
        for i in tqdm(range(ds_len), desc=f"Process: {task_id} - Labeling Demonstrations"):
            try:
                batch_large = next(ds_iter)
            except Exception as e:
                logging.error(f"Error in batch {i}: {e}")
                continue
       
            cur_actual_batch_idx = i + offset + (ds_len * (task_id))
            
            if cur_actual_batch_idx in annotated_indices:
                logging.info(f"Skipping batch {cur_actual_batch_idx}")
                continue
            
            orig_dicts = [step for step in batch_large]
            orig_dicts = stack_dicts(orig_dicts)
            
            cur_batch_lst.append(orig_dicts)
            
            if len(cur_batch_lst) < bsz:
                continue
            

            cur_batch_lst = []
            
            
            ########################################################
            # STAGE 1: Get objects and tasks
            ########################################################

            batch = cur_batch_lst


            if task_list is None and cfg.use_gt_tasks_for_objects :
                task_list = [b["observation"]["natural_language_instruction"][0].decode("utf-8") for b in batch]


            if object_list is not None:
                if cfg.use_gt_tasks_for_objects:
                    object_list = get_objects_from_annotation_parallel(task_list)
                    color_obj_dicts = None
                else:
                    all_objects = []
                    color_obj_dicts = []
                    for b in batch:
                        objects, synonyms = annotator.get_objects_in_image(b["rgb_static"],
                                                    predefined_objects=None,
                                                    use_som=False)
                        all_objects.append(objects)
                        color_obj_dicts.append(objects["color"])
                    object_list = all_objects


            cleaned_tasks = []
            cleaned_objects = []
            pred_indices = []
            cleaned_color_dicts = []
            for idx,(obj, task) in enumerate(zip(objects, task_list)):
                if obj is not None:
                    cleaned_tasks.append(task)
                    cleaned_objects.append(obj)
                    pred_indices.append(idx)
                    if color_obj_dicts is not None:
                        cleaned_color_dicts.append(color_obj_dicts[idx])


            cleaned_batch = [batch[idx] for idx in pred_indices]
            
            if cfg.use_gt_tasks_for_objects:
                obj_task_dicts = get_object_task_dicts_parallel(cleaned_objects, cleaned_tasks)
            else:
                obj_task_dicts = get_object_task_dicts_parallel(object_list, task_centric=False)






            all_prompts = []
            pred_batch_indices = []
            pred_local_indices = []
            pred_batches = []
            for idx, obj_task_dict in enumerate(tqdm(obj_task_dicts)):
                

                if obj_task_dict is None:
                    continue

                cur_actual_batch_idx = (i-len(obj_task_dicts)) + offset + idx + (ds_len * (task_id))
                cur_actual_batch_idx = str(cur_actual_batch_idx)

                hand_eval_dir = os.path.join(user_home, "hand_eval_iter_0", ds.name, str(cur_actual_batch_idx))
                os.makedirs(hand_eval_dir, exist_ok=True)

                vis_save_path = os.path.join(hand_eval_dir, "annotated_frames")

                cur_batch = cleaned_batch[idx]
                orig_cur_batch = cur_batch.copy()

                
                #get NILS format batch:

                cur_batch = get_NILS_format_batch(cur_batch,n_frames =2)

                if len(cleaned_color_dicts) > 0:
                    cur_color_dict = cleaned_color_dicts[idx]
                    for obj in obj_task_dict.keys():
                        if obj in cur_color_dict:
                            obj_task_dict[obj]["color"] = cur_color_dict[obj]
                        else:
                            obj_task_dict[obj]["color"] = ""
                else:
                    for obj in obj_task_dict.keys():
                        obj_task_dict[obj]["color"] = ""



                ########################################################
                # STAGE 2: Label frames
                ########################################################

                annotator.init_object_manager(obj_task_dict)
                try:
                    annotator.label_frames_fast(cur_batch["rgb_static"],cache_dir = cache_dir_folder, vis_save_path=vis_save_path)
                    annotator.get_obj_states(cur_batch["rgb_static"], None)


                    all_data_subset = {key: cur_batch[key] for key in cur_batch.keys() if
                                        key != "task_annotation" and key != "annotations" and "path" not in key and "keystate" not in key and "lang_ann" not in key}


                    annotator.object_manager.clean_objects(0,(obj_task_dict))
                    save_annotated_frames(annotator.object_manager, annotator.depth_predictions, all_data_subset,
                                          vis_save_path)

                    surface_det_images = cur_batch["rgb_static"][::2]
                    annotator.object_manager.robot.gripper_locations_3d = cur_batch["gripper_locations"][:, :3]


                    subset_frame_indices = np.array([0, len(cur_batch["rgb_static"]) - 1])

                    all_data_subset_sg_indices = {key: cur_batch[key][subset_frame_indices] for key in cur_batch.keys() if
                                                key != "task_annotation" and key != "annotations" and "path" not in key and "keystate" not in key and "lang_ann" not in key}


                    scene_graphs = annotator.create_scene_graphs(all_data_subset_sg_indices, vis_path=vis_save_path,
                                                                gripper_cam_labels=None,
                                                                subset_indices=None,
                                                                surface_det_images=surface_det_images,
                                                                area_weight=0.05)

                    gt_keystate_intervals = [(0, len(cur_batch["rgb_static"]))]

                    prompt_nl_reasons, reason_scores, objects = annotator.get_short_horizon_changes(
                        cur_batch["rgb_static"],
                        scene_graphs,
                        gt_keystate_intervals,
                        #gripper_action=cur_batch["gripper_actions"],
                        get_gripper_obj_proximity=False
                    )



                    if reason_scores[0] < 0.4:
                        continue

                    prompts = [annotator.llm.get_nl_prompt_keystate_reasons("", prompt_nl_reason, open_ended=cfg.open_ended) for
                            prompt_nl_reason in prompt_nl_reasons]


                    all_prompts.append(prompts[0])
                    pred_local_indices.append(idx)
                    pred_batch_indices.append(cur_actual_batch_idx)
                    pred_batches.append(orig_cur_batch)

                except Exception  as e:

                    logging.error(f"Error in labeling frames: {e}")
                    continue



            ########################################################
            # STAGE 3: Get tasks
            ########################################################


            llm_pred_tasks, candidates, scores = get_tasks_parallel(all_prompts)


            for prompt_idx, (task, candidate, score) in enumerate(zip(llm_pred_tasks, candidates, scores)):
                
                

                if task is None or len(task) == 0:
                    continue

                cur_local_idx = pred_local_indices[prompt_idx]
                cur_actual_batch_idx = str((i-len(obj_task_dicts)) + offset + cur_local_idx + (ds_len * (task_id)))
                cur_batch = cleaned_batch[cur_local_idx]

                hand_eval_dir = os.path.join(user_home, "hand_eval_iter_0", ds.name, cur_actual_batch_idx)
                
                #cur_out_dir = os.path.join(out_dir, str(pred_batch_indices[prompt_idx]))
                cur_out_dir = os.path.join(out_dir, cur_actual_batch_idx)

                #cur_batch = pred_batches[prompt_idx]
                prompt = all_prompts[prompt_idx]

                


                n_tasks = len(task)

                for t in range(n_tasks):
                    cur_batch["observation"]["language_instruction_NILS_" + str(t)] = task[t]

                n_tasks_to_predict = 3
                n_missing_tasks = n_tasks_to_predict - n_tasks

                for t in range(n_missing_tasks):
                    cur_batch["observation"]["language_instruction_NILS_" + str(t + n_tasks)] = ""

                os.makedirs(cur_out_dir, exist_ok=True)


                steps_flattened = flatten_dict(cur_batch)
                np.savez(os.path.join(cur_out_dir, "steps.npz"), **steps_flattened)

                with open (os.path.join(cur_out_dir,"prompt.txt"),"w") as f:
                    f.write(prompt)
                with open(os.path.join(cur_out_dir,"lang_NILS.txt"),"w") as f:
                    f.write("\n".join(task))
                with open(os.path.join(hand_eval_dir,"lang_NILS.txt"),"w") as f:
                    f.write("\n".join(task))


                




            
    except Exception as e:
        cur_actual_batch_idx = str((i-bsz)+offset+ cur_local_idx + (ds_len * (task_id)))
        cur_out_dir = os.path.join(out_dir, cur_actual_batch_idx)
        logging.error(f"Error in task {task_id} - batch {cur_actual_batch_idx}: {e}")
        os.makedirs(cur_out_dir, exist_ok=True)
        
    finally:
        queue.put((task, gpu_id))
                



    # with open(f"object_properties_{task_id}.pkl", "wb") as f:
    #     pickle.dump(obj_properties, f)


@hydra.main(config_path="../conf/annotator", config_name="eval_fractal")
def annotate_hydra_wrapper(cfg):
    



    cur_task = 0
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put((cur_task, GPU_IDS[gpu_ids]))
            cur_task += 1

    pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)

    repeated_cfg = [cfg] * (PROC_PER_GPU * NUM_GPUS)

    for _ in pool.imap_unordered(annotate, repeated_cfg):
        pass

    pool.close()
    pool.join()




def flatten_dict(d, parent_key='', sep='.'):
    """Flattens a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



if __name__ == "__main__":


    # NUM_GPUS = 4
    # PROC_PER_GPU = 3
    # queue = Queue()

    # cur_task = 0
    # for gpu_ids in range(NUM_GPUS):
    #     for _ in range(PROC_PER_GPU):
    #         queue.put((cur_task, gpu_ids))

    # pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)

    # for _ in pool.imap_unordered(annotate, range(PROC_PER_GPU * NUM_GPUS)):
    #     pass

    # pool.close()
    # pool.join()

    annotate_hydra_wrapper()