import gc
import logging
import os
import traceback
import warnings
from pathlib import Path
import hydra
import numpy as np
from multiprocessing import Pool, Queue


UNDEFINED_TASK = "Undefined"
warnings.filterwarnings('ignore')

queue_gpu = Queue()
queue = Queue()


def annotate(cfg):
    task = queue.get()

    gpu_id = queue_gpu.get()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from nils.annotator.KeyStateAnnotatorRefactored import (
        KeyStateAnnotator,
    )
    from nils.annotator.io_utils import save_annotated_frames
    from nils.utils.utils import setup_logging, split_batch, save_annotation, load_predefined_objects
    from nils.specialist_models.llm.openai_llm import get_tasks_parallel_gpt, get_simple_prompt_nl_reasons_gpt
    from nils.specialist_models.llm.google_cloud_api import get_tasks_parallel,get_prompt_simple
    from nils.specialist_models.vlm import gpt4v
    

    setup_logging()
    logging.info(f"Available GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logging.info(f"CUDA Device Count: {torch.cuda.device_count()}")

    ds = hydra.utils.instantiate(cfg["dataset"], n_jobs = cfg.n_splits,task_number=task)
    annotator = KeyStateAnnotator(cfg, initialize_models=True, ds=ds)
    loader = DataLoader(ds, batch_size=None)

    for i, batch_large in enumerate(tqdm(loader, desc=f"Process: {task} - Labeling Demonstrations")):
        if batch_large is None:
            continue
        if "keystates" in batch_large.keys():
            batch_split = split_batch(batch_large)
        else:
            batch_split = [batch_large]
        for batch in batch_split:
            scene_graphs = None
            try:
                if "actions" in batch.keys():
                    batch["actions"] = torch.concat(batch["actions"], dim=0)
                batch["frame_names"] = np.array(batch["frame_names"])
                annotator.owl.reset_cache()
                annotated_frames_dir = os.path.join(cfg.io.labeled_frames_dir, ds.name, Path(batch["path"][0]).parent,
                                                    "annotated_frames")
                os.makedirs(annotated_frames_dir, exist_ok=True)
                is_short_horizon = cfg.prior.use_gt_keystates

                logging.info("Retrieving objects in scene")
                predefined_objects = load_predefined_objects(cfg)
                

                if cfg.only_predefined_objects:
                    objects = predefined_objects
                    synonyms = {}
                else:
                    objects, synonyms = annotator.get_objects_in_image(batch["rgb_static"], predefined_objects = predefined_objects,
                                                                    use_som=False)
                
                    objects.extend(predefined_objects)

                if objects is None:
                    continue

                object_names = [obj["name"] for obj in objects]
                color_obj_dict = {obj["name"]: obj["color"] for obj in objects}
                objects = object_names
                objects = [obj for obj in objects if "robot" not in obj]

                logging.info(f"Detected the following objects in the scene: {objects}")
                if not cfg.prompt.open_ended:
                    logging.info("Prompting LLM to get possible actions")
                    if cfg.prior.use_predefined_task_list:
                        if not os.path.exists(cfg.prompt.task_list):
                            raise FileNotFoundError(f"Task list file not found: {cfg.prompt.task_list}")
                        with open(cfg.prompt.task_list, "r") as f:
                            possible_tasks = f.readlines()
                            possible_tasks = [task.strip() for task in possible_tasks]
                    else:
                        possible_tasks = annotator.get_possible_actions(objects)
                        logging.info(f"Possible tasks from LLM: {possible_tasks}")
                else:
                    possible_tasks = [UNDEFINED_TASK]

                ds.possible_actions = possible_tasks

                n_retries = 2
                successful_task_dict = False
                while n_retries > 0 and not successful_task_dict:
                    try:
                        object_task_dict = annotator.get_object_task_dict(possible_tasks, objects, False)
                        successful_task_dict = True
                    except Exception as e:
                        n_retries -= 1
                        logging.info("\033[91m" + f"Error Retrieving obj dict: {e}" + "\033[0m")
                        logging.info("Retrying to get object task dict")

                for obj in object_task_dict.keys():
                    if obj in color_obj_dict.keys():
                        object_task_dict[obj]["color"] = color_obj_dict[obj]
                    else:
                        object_task_dict[obj]["color"] = ""

                if object_task_dict is None:
                    continue

                interactable_objects = {name: properties for name, properties in list(object_task_dict.items())[1:]}
                surface_object = {list(object_task_dict.items())[0][0]: list(object_task_dict.items())[0][1]}

                annotator.init_object_manager(interactable_objects)
                annotator.object_manager.add_synonyms(synonyms)
                annotator.object_manager.add_surface_object(list(surface_object.keys())[0],
                                                            color=surface_object[list(surface_object.keys())[0]][
                                                                "color"])

                all_data_subset = {key: batch[key] for key in batch if key not in {"task_annotation",
                                                                                   "annotations"} and
                                   "path" not in key and "keystate" not in key}
                logging.info("Labeling frames")
                annotator.label_frames(all_data_subset["rgb_static"], vis_save_path=annotated_frames_dir,
                                       detection_threshold=cfg.detection_threshold,
                                       enable_temporal_aggregation=True,
                                       is_short_horizon=is_short_horizon)
                annotator.get_obj_states(all_data_subset["rgb_static"], None)

                obj_threshold = cfg.object_threshold
                annotator.object_manager.clean_objects(obj_threshold, predefined_objects)

                if cfg.io.save_frames:
                    save_annotated_frames(annotator.object_manager, annotator.depth_predictions, all_data_subset,
                                          annotated_frames_dir)
                annotator.owl.reset_cache()
 
                if cfg.prior.use_gt_keystates:
                    batch["keystates"] = np.array(batch["keystates"])
                    subset_frame_indices = batch["keystates"] - 1
                    subset_frame_indices = np.array([0] + list(subset_frame_indices))
                    is_short_horizon = True
                else:
                    subset_frame_indices = np.arange(0, len(batch["rgb_static"]), cfg.scene_graph.interval)
                all_data_subset_sg_indices = {
                    key: batch[key][subset_frame_indices] for key in batch.keys()
                    if key not in {"task_annotation", "annotations", "path", "keystate", "actions" ,"keystates"}
                }
                scene_graphs = annotator.create_scene_graphs(
                    all_data_subset_sg_indices, vis_path=annotated_frames_dir, gripper_cam_labels=None,
                    subset_indices=subset_frame_indices
                )

                
                if cfg.prior.use_gt_keystates:
                    gt_keystate_intervals = [
                        (0 if ks_idx == 0 else batch["keystates"][ks_idx - 1].item(), batch["keystates"][ks_idx].item())
                        for ks_idx in range(len(batch["keystates"]))
                    ]
                    keystate_reasons, keystate_scores, keystate_objects = annotator.get_short_horizon_changes(
                        all_data_subset["rgb_static"], scene_graphs, gt_keystate_intervals
                    )
                    keystates = batch["keystates"]-1
                else:
                    if is_short_horizon:
                        all_flows = None
                    else:
                        valid_flows = [sg.flow_raw for sg in scene_graphs if sg.flow_raw is not None]
                        if len(valid_flows) == 0:
                            all_flows = None
                        else:
                            all_flows = np.concatenate([sg.flow_raw for sg in scene_graphs if sg.flow_raw is not None], axis=0)
                    
                    object_movements = annotator.object_manager.get_object_movements(all_flows)
                    keystates, keystate_reasons, keystate_objects, keystate_scores = annotator.get_keystates(
                        all_data_subset, scene_graphs, object_movements, None,
                        cfg.scene_graph.interval)
                    #dict of lists to list of dicts
                    
                    valid_keystate_indices = keystate_scores > cfg.keystate_threshold
                    keystates = keystates[valid_keystate_indices]
                    keystate_reasons = {k: np.array(v)[valid_keystate_indices] for k, v in keystate_reasons.items()}
                    keystate_objects = keystate_objects[valid_keystate_indices]
                    keystate_scores = keystate_scores[valid_keystate_indices]
                    
                    prompts = annotator.create_prompts(keystates,batch,scene_graphs,keystate_reasons,keystate_objects)
                
                    keystate_reasons = np.array([{k: v[i] for k, v in keystate_reasons.items()} for i in range(len(keystate_reasons[list(keystate_reasons.keys())[0]]))])

                
                    
                
                annotator.depth_predictions = None

                color_obj_dict = {obj.name: obj.color for obj in annotator.object_manager.objects.values()}
                
                if "vertex" in cfg.llm._target_.lower():
                    prompts = [get_prompt_simple(reason) for reason in keystate_reasons]
                    llm_outputs, candidates, scores = get_tasks_parallel(prompts)
                elif "gpt" in cfg.llm._target_.lower():
                    prompts = [get_simple_prompt_nl_reasons_gpt(reason) for reason in keystate_reasons]
                    llm_outputs, candidates, scores = get_tasks_parallel_gpt(prompts)
                
                skipped_predictions_idx = []
                all_preds, all_scores, llm_scores = [], [], []

                ann_folder = "annotations"
                
                score_threshold = 1.4 if cfg.prior.use_gt_keystates else 0.1

                for idx, cur_pred_tasks in enumerate(llm_outputs):
                    cur_reasons = keystate_reasons[idx]
                    cur_score = [keystate_scores[idx]]
                    cur_llm_scores = scores[idx]
                    all_scores.append(keystate_scores)

                    if cur_score[0] <= score_threshold or not cur_pred_tasks:
                        skipped_predictions_idx.append(idx)
                        all_preds.append([])
                        llm_scores.append([-1, -1, -1])
                        continue

                    all_preds.append(cur_pred_tasks)
                    llm_scores.append(cur_llm_scores)
                    interacted_object = keystate_objects[idx]
                    interacted_object_synonyms = annotator.object_manager.objects[interacted_object].synonyms

                    data_save_path = str(Path(annotated_frames_dir).parent) + "/" +  ann_folder  + "_" + str(batch["frame_names"][keystates[idx]])
                    print(data_save_path)

                    save_annotation(color_obj_dict, cur_pred_tasks, data_save_path, interacted_object,
                                    interacted_object_synonyms, cur_reasons, cur_score)

                if cfg.prior.use_gt_keystates and cfg.use_vlm_fallback:
                    keystates_not_predicted = np.array(batch["keystates"])[skipped_predictions_idx]
                    predicted_actions, scores = gpt4v.get_grounding_predictions_baseline_vlm_batched(
                        keystates_not_predicted, batch, [ds.possible_actions], True)

                    for idx, pred in enumerate(predicted_actions):
                        global_idx = skipped_predictions_idx[idx]
                        all_preds[global_idx] = pred
                        data_save_path = str(Path(annotated_frames_dir).parent) + "/" +  ann_folder  + "_" + str(batch["frame_names"][keystates[idx]])
                        pred = [pr.strip() for pr in pred]
                        predicted_tasks_text = "\n".join(pred) + "\nconfidence: -1"
                        os.makedirs(data_save_path, exist_ok=True)
                        with open(data_save_path / "lang_NILS.txt", "w") as f:
                            f.write(predicted_tasks_text)

                # # memory cleanup
                # del scene_graphs
                # del annotator
                # gc.collect()
                # torch.cuda.empty_cache()
                # annotator = KeyStateAnnotator(cfg, initialize_models=True, ds=ds)
            except Exception as e:
                logging.info("\033[91m" + traceback.format_exc() + "\033[0m")
                logging.info(e)
                del scene_graphs
                torch.cuda.empty_cache()
                del annotator
                gc.collect()
                torch.cuda.empty_cache()
                annotator = KeyStateAnnotator(cfg, initialize_models=True, ds=ds)

                continue


cur_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(cur_dir)

@hydra.main(config_path=f"{parent_dir}/conf/annotator", config_name="base")
def annotate_hydra_wrapper(cfg):
    NUM_GPUS = len(cfg.GPU_IDS)
    GPU_IDS = cfg.GPU_IDS
    PROC_PER_GPU = cfg.PROC_PER_GPU 
    ds_n_splits = cfg.n_splits
    
    if NUM_GPUS*PROC_PER_GPU == 1:
        logging.info("Running in single process mode")
        for i in range(0, ds_n_splits):
            queue_gpu.put(GPU_IDS[0])
            queue.put(i)
            annotate(cfg)
        return

    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):  #
            queue_gpu.put(GPU_IDS[gpu_ids])                # # memory cleanup
                # del scene_graphs
                # del annotator
                # gc.collect()
                # torch.cuda.empty_cache()
                # annotator = KeyStateAnnotator(cfg, initialize_models=True, ds=ds)



if __name__ == "__main__":
    annotate_hydra_wrapper()