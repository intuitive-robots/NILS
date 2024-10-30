import copy
import gc
import logging
import os
import time
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import cv2
import hydra
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import zscore
from sklearn.linear_model import RANSACRegressor
from torchvision.ops import box_iou, masks_to_boxes, nms
from tqdm import tqdm

from nils.annotator.keystate_predictors.gripper_position_predictor import \
    GripperPosKeystatePredictor
from nils.annotator.keystate_predictors.keystate_combiner import (
    KeystateCombiner,
)
from nils.annotator.keystate_predictors.scene_graph_predictor import (
    get_scene_graph_changes,
)
from nils.annotator.object_manager import (
    NonMovableObject,
    ObjectManager,
)
from nils.annotator.prompts import (
    create_observation_prompt,
    get_nl_position_on_surface,
)
from nils.annotator.stage_1 import get_best_frame_objectness, create_som_annotated_image, \
    get_clip_class_scores, retrieve_surface_object_and_possible_surface_objects, get_temporal_coocurence, \
    filter_object_names_by_confidence, detect_high_overlap_objects
from nils.annotator.surface_detector import SurfaceTransformConfig, SurfaceTransformer
from nils.pointcloud.pointcloud import apply_mask_to_image
from nils.specialist_models.sam2.utils.amg import remove_small_regions
from nils.utils.utils import get_box_dense_agreement, get_intrinsics
from nils.specialist_models.llm.google_cloud_api import VertexAIAPI
from nils.specialist_models.llm.openai_llm import OpenAIAPI, get_simple_prompt_nl_reasons_gpt
from nils.scene_graph.canonicalization import get_surface_object_masks, \
    get_lines_from_surface_masks, cluster_lines, get_best_line_from_clusters, overlay_lines_on_mask, \
    plot_transformed_axes_on_image, get_best_surface_box
from nils.scene_graph.scene_graph import (
    create_scene_dist_graph,
    init_scene_dist_graph,)
from nils.scene_graph.scene_graph_utils import (
    average_scene_graphs,
    get_planar_transformation,
    transform_bbox, )
from nils.segmentator.deva_args import (
    add_common_eval_args,
    add_ext_eval_args,
    add_text_default_args,
    get_model_and_config,
)
from nils.segmentator.diva_resultsaver import ResultSaver
from nils.specialist_models.depth_predictor import DepthAnything
from nils.specialist_models.detectors.deva_custom_detection import (
    DEVAInferenceCoreCustom,
    process_frame_with_text, postprocess_deva_masks,
)
from nils.specialist_models.detectors.grounding_dino_hf import GroundingDINOHF
from nils.specialist_models.detectors.owlv2 import OWLv2
from nils.specialist_models.detectors.utils import postprocess_masks
from nils.specialist_models.flow_prediction import FlowPredictor
from nils.specialist_models.metrid_3d import Metric3D
from nils.specialist_models.openclip import get_occluded_frames
from nils.utils.plot import (
    crop_images_with_boxes,
)
from nils.utils.utils import filter_consecutive_values_with_outlier_fill, \
    list_of_dict_of_dict_to_dict_of_dict_of_list, compute_final_groups, \
    check_for_undetected_objects, get_best_detection_per_class, \
    incorporate_clip_scores_into_detections, incorporate_objectness_into_class_score, get_homography_transform, \
    get_mask_agreement, find_points_close_to_line, retry_on_exception, get_keystate_prediction_accuracy, \
    interleave_detections, cut_box_from_matrix
from nils.specialist_models.vlm.gemini import (
    get_object_states_batched,
    get_objects_in_image_batched,
)


class KeyStateAnnotator:
    def __init__(self, cfg, sam=None, initialize_models=True, ds=None, sh_detection=False):

        if initialize_models:
            self.detection_model = hydra.utils.instantiate(cfg.detector)

            self.segmentation_model = hydra.utils.instantiate(cfg.segmentation_model) if sam is None else sam

            if not sh_detection:
                self.flow_predictor = FlowPredictor()
                self.ov_segmentation_model = hydra.utils.instantiate(cfg.ov_segmentation_model)
                self.owl = OWLv2("google/owlv2-base-patch16-ensemble")
            self.state_detector = hydra.utils.instantiate(cfg.state_detector)

            # self.dust3r = Dust3r()
            self.m3d = Metric3D()
            self.depth_predictor = DepthAnything()

        self.cfg = cfg

        cuda_device = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(cuda_device)
        total_memory_bytes = device_properties.total_memory
        self.total_memory_gb = total_memory_bytes / (1024 ** 3)

        self.ds = ds

        if "bridge" in self.ds.name:
            self.intrinsic_parameters = {
                'width': 640,
                'height': 480,
                'fx': 500,
                'fy': 500,
                'cx': 320,
                'cy': 240,
            }
        else:
            self.intrinsic_parameters = None

        self.last_scene_graphs = []
        self.depth_predictions = None

        self.llm = hydra.utils.instantiate(cfg.llm)
        self.google_llm = VertexAIAPI()
        self.openai_api = OpenAIAPI()
        self.vlm = hydra.utils.instantiate(cfg.vlm)

        self.flow_mask_fwd = None
        self.flow_mask_bwd = None
        self.flow_predictions = None
        self.flow_predictions_bwd = None

    def init_object_manager(self, obj_dict, seg_prompts={}, det_prompts={}, state_prompts={}):
        if self.ds.name == "calvin":

            self.object_manager = ObjectManager(self.ds.object_dict, self.ds.segmentation_prompts,
                                                self.ds.detection_prompts, cfg=self.cfg)

            for obj, box in self.ds.gt_regions.items():
                mask = np.zeros((200, 200))
                box = box.astype(int)
                mask[box[1]:box[3], box[0]:box[2]] = 1
                self.object_manager.add_object_detections(
                    {obj: {"box": np.repeat(box[None], (self.cfg.seq_len // self.cfg.scene_graph_interval), axis=0),
                           "score": np.array([1.0] * (self.cfg.seq_len // self.cfg.scene_graph_interval)),
                           "mask": np.repeat(mask[None], self.cfg.seq_len // self.cfg.scene_graph_interval, axis=0)}})
        else:
            self.object_manager = ObjectManager(object_dict=obj_dict, segmentation_prompts=seg_prompts,
                                                detection_prompts=det_prompts, state_prompts=state_prompts,
                                                cfg=self.cfg)

    def get_objects_in_image(self, images, temporal_threshold=0.12, bsz=2, predefined_objects=[],
                             use_som=True, use_som_mask=False, simple_ablation=False, use_obj_det_clip_baseline=False):

        """
                STAGE 1: Get objects in image
        """

        use_first_frame = True
        if len(predefined_objects) == 0:
            predefined_objects = None

        n_images_for_detection = 16

        image_subset_indices = np.linspace(0, len(images) - 1, n_images_for_detection, dtype=int)

        image_subset = images[image_subset_indices]

        vocab_name = Path(self.cfg.vocab_file).stem

        owl_out_path = Path(self.cfg.vocab_file).parent / vocab_name / f"{self.owl.name}.pth"
        with open(self.cfg.vocab_file) as f:
            lvis_classes = f.read().splitlines()
            lvis_classes = [x[x.find(':') + 1:] for x in lvis_classes]

        lvis_classes = np.array(lvis_classes)

        best_frame = 0
        if not use_first_frame or use_som or use_obj_det_clip_baseline:
            owl_text_classifier = self.owl.get_text_classifier(lvis_classes, owl_out_path)
            n_boxes = 15
            best_frame = get_best_frame_objectness(image_subset, bsz, n_boxes, self.owl, owl_text_classifier)

        if self.cfg.use_vlm_predictions_only:
            vlm_objects = self.vlm.get_objects_in_image(image_subset[best_frame])
            logging.info(f"VLM objects: {vlm_objects}")

            if vlm_objects is not None:
                vlm_objects_list = [item.strip() for obj in vlm_objects.keys() for item in obj.split(",")]
                vlm_objects_list += [lvis_classes[0]]
                return vlm_objects_list
            else:
                return None

        get_vlm_names = True

        if use_obj_det_clip_baseline:
            get_vlm_names = False
        # get vlm names with SOM prompting

        if get_vlm_names:

            if simple_ablation:
                n_frames = 1
            else:
                n_frames = 8

            frame_indices = np.linspace(0, len(image_subset) - 1, n_frames, dtype=int)
            init_obj_det_frames = image_subset[frame_indices]

            annotated_images_np = []

            robot_detection = \
                self.detection_model.detect_objects(init_obj_det_frames, ["robot gripper"], threshold=0.1)
            for cur_frame_idx, init_obj_det_frame in enumerate(init_obj_det_frames):

                annotated_frame = np.array(init_obj_det_frame)

                if use_som:
                    # resize frame
                    img_annotated = create_som_annotated_image(annotated_frame, bsz, cur_frame_idx, use_som_mask)
                    img_annotated_np = img_annotated.get_image()

                else:
                    img_annotated_np = np.array(init_obj_det_frame)

                annotated_images_np.append(img_annotated_np)

            if "prismatic" in self.vlm.__class__.__name__.lower():
                results = [self.vlm.get_objects_in_image(annotated_images_np[i]) for i in
                           range(len(annotated_images_np))]
            else:
                results = Parallel(n_jobs=n_frames, backend="threading")(
                    delayed(get_objects_in_image_batched)(annotated_images_np[i]) for i in
                    range(len(annotated_images_np)))

            if simple_ablation:
                return results[0], None

            if any([result is None for result in results]):
                return None

            vlm_object_names = []
            obj_set_detections = []
            results_cleaned = []
            surface_obj_scores = []
            surface_objects = []

            possible_surface_objects_in_det = []
            possible_surface_objects_scores = []

            n_boxes = 15
            region_proposals, objectness, grounding_scores, class_embeds = self.owl.get_region_proposals(
                init_obj_det_frames,
                bsz,
                n_boxes=n_boxes,
                text_classifier=None
            )

            for idx, result in enumerate(results):
                ov_prompts = [f"{obj['color']} {obj['name']}" for obj in result]
                ov_prompts = [name.replace(".", "") for name in ov_prompts]
                detections = self.detection_model.detect_objects(np.array(init_obj_det_frames), ov_prompts, bsz=8,
                                                                 )
                all_clip_class_scores = get_clip_class_scores(self.state_detector, detections, init_obj_det_frames,
                                                              ov_prompts)

                all_frame_detections = []

                for frame_idx, detection in enumerate(detections):
                    image_area = init_obj_det_frames[frame_idx].shape[0] * init_obj_det_frames[frame_idx].shape[1]

                    surface_obj, surface_obj_score, possible_surface_obj, possible_surface_obj_score = retrieve_surface_object_and_possible_surface_objects(
                        detection, image_area, result)

                    surface_objects.append(surface_obj)
                    surface_obj_scores.append(surface_obj_score)
                    possible_surface_objects_in_det.append(possible_surface_obj)
                    possible_surface_objects_scores.append(possible_surface_obj_score)

                    # remove surface object
                    surface_obj_idx = detection.class_id == 0
                    detection = detection[~surface_obj_idx]

                    cur_frame_clip_scores = all_clip_class_scores[frame_idx]
                    cur_frame_clip_scores = cur_frame_clip_scores[~surface_obj_idx]

                    detections_filtered = detection[detection.confidence > 0.2]

                    cur_frame_clip_scores = cur_frame_clip_scores[detection.confidence > 0.2]
                    detections_clip_scores = incorporate_clip_scores_into_detections(detections_filtered,
                                                                                     cur_frame_clip_scores)

                    detections_filtered = get_best_detection_per_class(detections_clip_scores)
                    detections_voted = incorporate_objectness_into_class_score(detections_filtered,
                                                                               region_proposals[frame_idx],
                                                                               objectness[frame_idx])

                    all_frame_detections.append(detections_voted)

                obj_set_detections.append(all_frame_detections)
                results_cleaned.append(result[1:])

                vlm_object_names += result

            best_surface_obj_idx = np.argmax(surface_obj_scores)
            best_surface_obj = [surface_objects[best_surface_obj_idx]]
            if len(possible_surface_objects_in_det) > 0:
                best_det_obj_idx = np.argmax(possible_surface_objects_scores)

                best_det_obj_score = possible_surface_objects_scores[best_det_obj_idx]

                if best_det_obj_score > np.max(surface_obj_scores):
                    best_surface_obj = [possible_surface_objects_in_det[best_det_obj_idx]]

            results = results_cleaned

            all_class_scores, all_frame_detections_concat, all_nl_classes_to_compare, matching_classes_coocurrences, n_class_appearances = get_temporal_coocurence(
                init_obj_det_frames, obj_set_detections, results)

            final_cooccurrences = compute_final_groups(matching_classes_coocurrences, n_class_appearances, 0.2)
            matching_classes = final_cooccurrences

            all_class_scores_avg = {k: np.mean(np.stack(v), axis=0) for k, v in all_class_scores.items()}

            synonym_classes = {}
            nl_classes_to_compare = []
            global_class_indices = []
            global_class_scores = []
            top_3_scores_lst = []
            for group in matching_classes:
                scores_to_compare = []
                for idx in group:
                    scores_to_compare.append(all_class_scores_avg[idx])

                scores_to_compare = np.stack(scores_to_compare).squeeze(1)
                best_obj = np.argmax(scores_to_compare)
                top_3_scores = np.sort(scores_to_compare)[::-1][:3]
                top_3_scores_indices = np.argsort(scores_to_compare)[::-1][:3]
                top_3_global_indices = np.array([group[idx] for idx in top_3_scores_indices])
                best_global_idx = group[best_obj]

                # print(top_3_scores)
                top_3_scores_lst.append(top_3_scores)

                if np.sum(top_3_scores) < 0.6 and np.max(top_3_scores) < 0.3:
                    continue

                if best_global_idx in global_class_indices:
                    prev_idx = global_class_indices.index(best_global_idx)
                    prev_top_3_scores = top_3_scores_lst[prev_idx].sum()
                    cur_top_3_scores = top_3_scores.sum()
                    if cur_top_3_scores > prev_top_3_scores:
                        global_class_indices[prev_idx] = best_global_idx
                        global_class_scores[prev_idx] = top_3_scores[0]
                        nl_classes_to_compare[prev_idx] = all_nl_classes_to_compare[best_global_idx - 1]
                        close_score_diffs = top_3_scores[0] - top_3_scores[1:] < 0.5

                    else:
                        continue
                else:
                    nl_classes_to_compare.append(all_nl_classes_to_compare[best_global_idx - 1])

                    global_class_indices.append(best_global_idx)
                    global_class_scores.append(top_3_scores[0])

                score_diffs = top_3_scores[0] - top_3_scores[1:]

                close_score_diffs = score_diffs < 0.5
                synonym_classes[all_nl_classes_to_compare[best_global_idx - 1]["name"]] = [
                    all_nl_classes_to_compare[idx - 1] for idx in top_3_global_indices[1:][close_score_diffs] if
                    all_nl_classes_to_compare[idx - 1]["name"] != all_nl_classes_to_compare[best_global_idx - 1][
                        "name"]]

            boxes_to_compare = []
            nl_classes_cleaned = []

            for frame_idx in range(len(init_obj_det_frames)):
                cur_frame_object_set_dets = all_frame_detections_concat[frame_idx]
                matching_class_indices = []
                filtered_nl_classes_to_compare = []
                for idx, class_idx in enumerate(cur_frame_object_set_dets.class_id):

                    if class_idx in global_class_indices:
                        matching_class_indices.append(idx)
                        filtered_nl_classes_to_compare.append(all_nl_classes_to_compare[class_idx - 1])

                cur_boxes_to_compare = cur_frame_object_set_dets[matching_class_indices].xyxy
                boxes_to_compare.append(cur_boxes_to_compare)

                high_conf_proposals_idx = objectness[frame_idx] > self.cfg.objectness_threshold
                high_conf_proposals = region_proposals[frame_idx][high_conf_proposals_idx]

                ious = box_iou(torch.tensor(high_conf_proposals), torch.tensor(cur_boxes_to_compare))
                ious = ious.numpy()
                ious = np.max(ious, axis=0)
                for nl_class in np.array(filtered_nl_classes_to_compare)[ious > 0.75]:
                    if nl_class not in nl_classes_cleaned:
                        nl_classes_cleaned.append(nl_class)

            vlm_object_names = nl_classes_cleaned

            if vlm_object_names is None:
                return None

            ov_detections, valid_classes = filter_object_names_by_confidence(self.detection_model, init_obj_det_frames,
                                                                             vlm_object_names)

            high_overlap_class_id_counter_robot = detect_high_overlap_objects(ov_detections, robot_detection)
            high_overlap_class_ids = [k for k, v in high_overlap_class_id_counter_robot.items() if
                                      v >= len(robot_detection) // 2]

            high_overlap_class_ids = [class_id for class_id in high_overlap_class_ids if class_id in valid_classes]

            # check if objects contain robot gripper or similar
            matched_overlap_class_ids = []
            for idx, obj in enumerate(vlm_object_names):
                if "robot" in obj["name"] or "gripper" in obj["name"] or "arm" in obj["name"]:
                    matched_overlap_class_ids.append(idx)

            high_overlap_class_ids = high_overlap_class_ids + matched_overlap_class_ids
            logging.info(f"High overlap classes: {np.array(vlm_object_names)[high_overlap_class_ids]}")

            final_high_overlap_class_ids = []
            if predefined_objects is not None:
                predefined_objects_prompts = [f"{obj['color']} {obj['name']}" for obj in predefined_objects]
                predefined_objects_detections = self.detection_model.detect_objects(
                    np.array(image_subset), predefined_objects_prompts, reduce_threshold=True)

                high_predefined_objects_overlap_counter = detect_high_overlap_objects(ov_detections,
                                                                                      predefined_objects_detections)

                for k, v in high_predefined_objects_overlap_counter.items():
                    if v >= len(predefined_objects_detections) // 2:
                        final_high_overlap_class_ids.append(k)
                logging.info(
                    f"High overlap with predefined objects: {np.array(vlm_object_names)[final_high_overlap_class_ids]}")

                high_overlap_class_ids = np.concatenate(
                    [high_overlap_class_ids, final_high_overlap_class_ids])

            vlm_object_names = [obj for idx, obj in enumerate(vlm_object_names) if
                                idx not in high_overlap_class_ids and idx in valid_classes]
            
            vlm_object_names = [obj for obj in vlm_object_names if obj["name"] not in best_surface_obj[0]["name"]]
            vlm_object_names = best_surface_obj + vlm_object_names
            vlm_object_names = [val for val in vlm_object_names if "robot" not in val["name"]]

            return vlm_object_names, synonym_classes

    def clean_prompt_and_manager(self, possible_actions):
        obj_to_delete = []
        new_actions = copy.deepcopy(list(possible_actions))
        for obj_name, obj in self.object_manager.objects.items():
            if np.isnan(obj.confidence).sum() > len(obj.confidence) / 2:
                obj_to_delete.append(obj)
                for action in new_actions:
                    if obj.name in action:
                        new_actions.remove(action)
        for obj in obj_to_delete:
            self.object_manager.objects.pop(obj.name)
        return new_actions

    def propagate_masks_with_deva(self, images, save_path):
        parser = ArgumentParser()

        add_common_eval_args(parser)
        add_ext_eval_args(parser)
        add_text_default_args(parser, self.cfg)

        network, config, args = get_model_and_config(parser)

        vid_length = len(images)

        config["amp"] = True if torch.cuda.is_available() else False

        if vid_length < 30:
            config['num_voting_frames'] = 2
            config["detection_every"] = 2

        if vid_length == 2:
            config["num_voting_frames"] = 1
            config["detection_every"] = 1

        config['enable_long_term_count_usage'] = (
                config['enable_long_term']
                and (vid_length / (config['max_mid_term_frames'] - config['min_mid_term_frames']) *
                     config['num_prototypes']) >= config['max_long_term_elements'])
        deva_inference_core = DEVAInferenceCoreCustom(network, config=config)
        deva_inference_core.next_voting_frame = config['num_voting_frames'] - 1
        deva_inference_core.object_manager.use_long_id = True

        out_path = os.path.join(save_path, "deva_annotations")

        result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva_inference_core.object_manager)
        class_mask_dicts = []

        # if too short, forward and backward pass for consistent detections
        do_reverse = True
        indices_check = []

       
        image_indices_fwd = np.arange(len(images))
            
        if do_reverse:
            images_reversed = np.array(images)[::-1]
            images_t_steps = image_indices_fwd
            images_reversed_t_steps = image_indices_fwd[::-1]

            images_fwd_bwd = np.concatenate([images, images_reversed], axis=0)
            image_indices_fwd_bwd = np.concatenate([images_t_steps, images_reversed_t_steps], axis=0).astype(int)
        else:
            images_fwd_bwd = images
            image_indices_fwd_bwd = np.arange(len(images))
            

            
            
        with torch.cuda.amp.autocast(enabled=config['amp']):
            for image_index, ti in enumerate(tqdm((image_indices_fwd_bwd))):

                cur_image = images_fwd_bwd[image_index]
                frame_name = str(image_index) + ".jpg"
                object_info = self.object_manager.get_object_stats_frame(ti)

                object_info["robot"] = self.object_manager.get_robot_stats_frame(ti)["robot"]

                classes = list(object_info.keys())
                masks = []
                scores = []
                boxes = []

                
                # if last_fwd_step:
                #     deva_inference_core.next_voting_frame = 999999
                    
                if deva_inference_core.next_voting_frame >= len(images):
                    deva_inference_core.next_voting_frame = len(images)-1
                        
                    
                if image_index >= len(images):
                    deva_inference_core.next_voting_frame = 999999
                    
            
                if image_index < len(images):
                    for obj, info in object_info.items():
                        classes.append(obj)
                        masks.append(info["mask"])
                        scores.append(info["score"])
                        boxes.append(info["box"])

                cur_dict = process_frame_with_text(deva_inference_core, frame_name, result_saver, image_index,
                                                   len(images_fwd_bwd) - 1, np.array(masks),
                                                   np.array(scores), np.array(boxes), np.array(classes),
                                                   np.array(cur_image), last_obj_mask_dict=class_mask_dicts[-1] if len(
                        class_mask_dicts) > 0 else None,
                                                   )

                class_mask_dicts.extend(cur_dict)
                indices_check.append(ti)

        if do_reverse:
            class_mask_dicts = class_mask_dicts[len(images):][::-1]

        result_dict = {}
        for obj_name in classes:
            cur_obj_masks = []
            cur_obj_boxes = []
            cur_obj_scores = []

            for tstep in range(len(images)):
                if obj_name in class_mask_dicts[tstep].keys():
                    if class_mask_dicts[tstep][obj_name]["mask"].sum() == 0:

                        cur_obj_masks.append(np.zeros((images.shape[1], images.shape[2])))
                        cur_obj_boxes.append(np.zeros((4,)))
                        cur_obj_scores.append(np.nan)
                    else:
                        cur_obj_masks.append(class_mask_dicts[tstep][obj_name]["mask"])
                        cur_obj_boxes.append(np.array(
                            masks_to_boxes(torch.tensor(class_mask_dicts[tstep][obj_name]["mask"][None])).squeeze(0)))
                        cur_obj_scores.append(class_mask_dicts[tstep][obj_name]["score"])
                else:
                    cur_obj_masks.append(np.zeros((images.shape[1], images.shape[2])))
                    cur_obj_boxes.append(np.zeros((4,)))
                    cur_obj_scores.append(np.nan)

            cur_obj_masks = np.stack(cur_obj_masks)
            cur_obj_boxes = np.stack(cur_obj_boxes)
            cur_obj_scores = np.array(cur_obj_scores)

            if obj_name != "robot" and isinstance(self.object_manager.objects[obj_name], NonMovableObject):
                boxes = self.object_manager.objects[obj_name].boxes
                score = self.object_manager.objects[obj_name].confidence
                result_dict[obj_name] = {"mask": cur_obj_masks.copy(), "box": boxes.copy(), "score": score.copy()}
            else:
                result_dict[obj_name] = {"mask": cur_obj_masks.copy(), "box": cur_obj_boxes.copy(),
                                         "score": cur_obj_scores.copy()}

        postprocess_deva_masks(result_dict)

        penalty_list_of_dicts = deva_inference_core.penalty_list
        # list of dicts to dict of lists

        penalty_dict = {}
        for pen_dic in penalty_list_of_dicts:
            for key, value in pen_dic.items():
                if key not in penalty_dict:
                    penalty_dict[key] = []
                penalty_dict[key].append(value)

        for key, value in penalty_dict.items():
            # flatten list
            flattened_val = [item for sublist in value for item in sublist]
            penalty_dict[key] = np.sum(flattened_val) / len(flattened_val)
            result_dict[key]["score"] = result_dict[key]["score"] - (penalty_dict[key] * 0.3)

        self.object_manager.add_object_detections(result_dict, is_voted_temporal=True)

        del deva_inference_core
        del result_dict
        del cur_obj_masks
        gc.collect()
        torch.cuda.empty_cache()

    def get_possible_actions(self, objects):
        objects = [obj.split(",")[0] for obj in objects]
        possible_actions = self.openai_api.get_possible_actions(objects)
        # possible_actions = self.google_llm.get_possible_actions(objects)

        return possible_actions

    @retry_on_exception(Exception, retries=3)
    def get_object_task_dict(self, tasks, objects, task_centric=True):
        task_list = self.google_llm.get_object_task_list(tasks, objects, task_centric=task_centric)
        for obj in task_list.keys():
            if obj in task_list.keys():
                task_list[obj]["color"] = task_list[obj]
            else:
                task_list[obj]["color"] = ""
        return task_list

    def get_obj_states_vlm(self, images, init_states=None, index=0, end_indices=None):

        images_cropped = []
        promtps = []

        prompt = "What is the state of the [OBJECT]?\n Select from: [STATES]."

        for obj_name, obj in self.object_manager.objects.items():
            if obj.states is not None:
                cropped_images = np.stack(
                    [crop_images_with_boxes(images[idx], obj.boxes[idx], resize=True) for idx in
                     range(len(images))])

                if ("open" in obj.states or "closed" in obj.states) and (
                        "drawer" not in obj_name and "pot" not in obj_name and "door" not in obj_name):
                    obj_name += " door"

                for state_idx in range(len(obj.states)):
                    promtps.append(prompt.replace("[OBJECT]", "drawer").replace("[STATES]", ",".join(obj.states)))
                images_cropped += [cropped_img for cropped_img in cropped_images]
        if len(images_cropped) > 0:
            states = get_object_states_batched(images_cropped, promtps)

            for idx, (obj_name, obj) in enumerate(self.object_manager.objects.items()):
                if obj.states is not None:
                    obj_states = []
                    for state_idx, state in enumerate(obj.states):
                        obj_states.append(states[idx][state_idx])
                    obj.states = obj_states

    def get_obj_states(self, images, init_states=None, index=0, end_indices=None):
        robot_masks = self.object_manager.robot.mask

        if init_states is None:
            for obj_name, obj in self.object_manager.objects.items():
                if obj.states is not None:
                    cropped_images = np.stack(
                        [crop_images_with_boxes(images[idx], obj.boxes[idx], resize=True) for idx in
                         range(len(images))])

                    if ("open" in obj.states or "closed" in obj.states) and (
                            "drawer" not in obj_name and "pot" not in obj_name and "door" not in obj_name):
                        obj_name += " door"

                    states = self.state_detector.predict(cropped_images, obj.states, obj_name, reduction="sig")
                    calibrate = True
                    max_per_state = np.max(states, axis=0)
                    overall_max = np.max(max_per_state)
                    diff_per_state = overall_max - max_per_state

                    state_mean = np.mean(states)

                    calibrate_fac = np.minimum(state_mean, diff_per_state)
                    if calibrate:
                        states = states + calibrate_fac
                    if len(cropped_images) > 4:
                        if self.cfg.enable_object_state_filtering:
                            cropped_robot_masks = np.stack(
                                [crop_images_with_boxes(robot_masks[idx], obj.boxes[idx], resize=True) for idx in
                                 range(len(images))])

                            occluded_mask = get_occluded_frames(cropped_images, cropped_robot_masks, threshold=0.4)

                            states[occluded_mask] = 0
                        else:
                            occluded_mask = np.ones(len(states)).astype(bool)

                        states_smoothed = pd.DataFrame(states).rolling(3, min_periods=1, center=True).mean().values

                        states_smoothed[occluded_mask] = 0

                        pred_states = np.argmax(states_smoothed, axis=1)
                        pred_states[occluded_mask] = -1
                    else:
                        states_smoothed = states
                        pred_states = np.argmax(states_smoothed, axis=1)

                    pred_states = pred_states.astype(int)
                    pred_states_nl = np.array(
                        [obj.states[state] if state != -1 else "occluded" for state in pred_states])

                    if len(cropped_images) > 4:
                        pred_states_nl[occluded_mask] = None
                        pred_states_nl_filtered = filter_consecutive_values_with_outlier_fill(pred_states_nl, 2, 2)
                    else:
                        pred_states_nl_filtered = pred_states_nl

                    if end_indices is not None:
                        pred_states_nl_filtered = pred_states_nl_filtered[end_indices]
                        states_smoothed = states_smoothed[end_indices]
                        cropped_images = cropped_images[end_indices]
                    obj.state_confidences = states_smoothed
                    obj.predicted_states = list(pred_states_nl_filtered)

                    obj.cropped_images = cropped_images

    def get_save_path(self):
        base_path = self.ds.path

        if not os.path.isdir(base_path):
            base_path = base_path.split(".")[0]

        print(base_path)

        llm_name = self.llm.__class__.__name__
        keystate_voters = "_".join(
            [predictor.replace("_predictor", "") for predictor in list(self.cfg.keystate_predictors_for_voting)])

        if self.cfg.eval_grounding:
            path = os.path.join(base_path, "keystate_annotations", llm_name, keystate_voters)
        else:
            path = os.path.join(base_path, "keystate_annotations", keystate_voters)

        if self.cfg.ablation:
            ablation_path = self.get_ablation_save_dir()

            path = os.path.join(path, "ablation", ablation_path)

        os.makedirs(path, exist_ok=True)

        return path

    def get_keystates(self, batch, scene_graphs=None, object_movements=None, labeled_gripper_cam_data=None,
                      scene_graphs_subset_interval=4, match_threshold=4):

        interacted_robot_objects = None
        keystates_for_voting = list(self.cfg.keystate_predictors_for_voting)
        keystate_combiner = KeystateCombiner(keystates_for_voting)

        predictors = [hydra.utils.instantiate(predictor, downsample_interval=scene_graphs_subset_interval, name=name)
                      for
                      name, predictor in self.cfg.keystate_predictor.items()]

        
        has_gripper_actions = "gripper_actions" in batch.keys()
        
        for predictor in predictors:
            if "GripperClosePredictor" in predictor.__class__.__name__ and not has_gripper_actions:
                logging.warning("Gripper actions not found, skipping GripperClosePredictor")
                continue
            keystate_combiner.add_predictor(predictor)

        if object_movements is None:
            all_flows = np.concatenate([sg.flow_raw for sg in scene_graphs if sg.flow_raw is not None], axis=0)

            object_movements = self.object_manager.get_object_movements(all_flows)

        batch_without_ann = {key: batch[key] for key in batch.keys() if
                             key != "annotations" and "annotation" not in key}
        data = {"batch": batch_without_ann, "scene_graphs": scene_graphs,
                "labeled_gripper_cam_data": labeled_gripper_cam_data,
                "object_movements": object_movements,
                "object_manager": self.object_manager, "interacted_robot_objects": interacted_robot_objects,
                "depth_map": self.depth_predictions,
                "vlm": self.vlm}

        keystate_combiner.init_keystates_by_object(self.object_manager.get_all_obj_names())
        
        predicted_keystates = keystate_combiner.predict_keystates(data)

        frames = batch["rgb_static"]
        keystate_preds = {}
        if "annotations" in batch.keys():
            batch["annotations"] = np.array(batch["annotations"])

        keystates = keystate_combiner.combine_keystates(match_threshold=match_threshold,
                                                        score_threshold=0.3, min_keystate_distance=3)
        if len(keystates) == 0:
            return [], [], [], []
        keystate_reasons = keystate_combiner.get_keystate_reasons()

        keystate_objects = keystate_combiner.keystate_objects

        scores = keystate_combiner.keystate_scores

        return keystates, keystate_reasons, keystate_objects, scores

    def compute_masks(self, images, bsz=2, box_prompts=None, pred_objects=None):

        region_objects = self.object_manager.objects

        if box_prompts is None or pred_objects is None:
            box_prompts = []
            pred_objects = []
            for obj, obj_states in region_objects.items():
                if obj_states.boxes is not None:
                    pred_objects.append(obj)
                    box_prompts.append(obj_states.boxes)

            if len(box_prompts) == 0:
                box_prompts = self.object_manager.robot.boxes[:, None, :]
            else:
                box_prompts = np.stack(box_prompts, axis=1)

                box_prompts = np.concatenate([box_prompts, self.object_manager.robot.boxes[:, None, :]], axis=1)
        mask, ious = self.segmentation_model.segment(images, box_prompts, bsz=bsz)

        mask = np.stack([[remove_small_regions(m, area_thresh=200, mode="islands")[0] for m in masks_frame]
                         for masks_frame in mask])

        return mask, pred_objects, ious

    def get_surface_object(self, images, surface_object_prompts=None, objects=None, detections=None):

        image_area = images[0].shape[1] * images[0].shape[0]

        if detections is not None:
            # check for possible surface objects in detections
            possible_surface_objects = []
            possible_surface_object_scores = []

            for det in detections:
                object_areas = [(det[2] - det[0]) * (det[3] - det[1]) for det in det.xyxy]

                possible_surface_objects_idx = np.array(object_areas) > 0.6 * image_area
                possible_surface_objects_det = det[possible_surface_objects_idx]

                if len(possible_surface_objects_det) > 0:
                    best_surface_obj_idx = possible_surface_objects_det.confidence.argmax()

                    best_surface_obj_score = possible_surface_objects_det.confidence[best_surface_obj_idx]
                possible_surface_objects.append(best_surface_obj_idx)
                possible_surface_object_scores.append(best_surface_obj_score)
            best_surface_obj_idx = np.argmax(possible_surface_object_scores)
            best_surface_obj_score = possible_surface_object_scores[best_surface_obj_idx]
            surface_obj_det = objects[best_surface_obj_idx]

        else:

            if surface_object_prompts is None:
                surface_object_prompts = ["table top", "floor", "kitchen counter", "stove top", "hotplates",
                                          "counter top",
                                          "counter"]

            surface_obj = None
            if objects is not None:
                for object in objects:
                    if object in surface_object_prompts:
                        surface_obj = object

            best_boxes, best_class = get_best_surface_box(self.detection_model, images, surface_object_prompts,
                                                          vlm_key=surface_obj,
                                                          area_weight=0.05)

            return best_boxes, best_class

    def detect_surface_object(self, images):

        n_frames = 8
        subset_indices = np.linspace(0, len(images) - 1, n_frames).astype(np.int32)
        subset_images = images[subset_indices]

        prompt = self.object_manager.surface_object.name

        detections = self.detection_model.detect_objects(subset_images, [prompt], threshold=0.1, bsz=4)

        best_boxes = []
        # get best box for each frame:
        all_box_areas = []
        for detection in detections:
            scores = detection.confidence
            boxes = detection.xyxy
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            all_box_areas.append(np.max(box_areas))

            box_area_normed = box_areas / np.max(box_areas)

            scores_with_area = scores + 0.5 * box_area_normed

            best_detection_idx = np.argmax(scores_with_area)

            best_boxes.append(boxes[best_detection_idx])
        averaged_box = np.mean(best_boxes, axis=0)
        mean_box_area = np.mean(all_box_areas)

        if mean_box_area < 0.5 * images[0].shape[1] * images[0].shape[0]:
            self.object_manager.surface_object = None
        else:

            self.object_manager.surface_object.boxes = averaged_box

    def label_frames_fast(self, images, detection_threshold=0.04, score_threshold=0.1, vis_save_path=None,
                          cache_dir=None):
        """
        Fast implementation for cases where we do not need temporal consistency for keystate detection.
        """

        # methods: owl embed, siglip embed (what about state changed objects?)  + region proposals

        # perform inital object detection:

        width = images[0].shape[1]
        height = images[0].shape[0]

        # get intrinsics or load from cache:
        intrinsic_parameters = get_intrinsics(images, cache_dir=cache_dir)

        self.intrinsic_parameters = intrinsic_parameters

        intrinsic_parameters_naive = {
            'width': width,
            'height': height,
            'fx': 0.6 * width,
            'fy': 0.6 * height,
            'cx': width / 2,
            'cy': height / 2,
        }

        detector_prompts = self.object_manager.get_object_detection_prompts()
        objects = list(detector_prompts.values())

        if isinstance(self.detection_model, GroundingDINOHF):
            embodiment_prompt = self.cfg.embodiment_prompt_grounding_dino
        else:
            embodiment_prompt = self.cfg.embodiment_prompt_owl

        prompts = objects + [embodiment_prompt]
        prompts = [prompt.strip() for prompt in prompts]
        object_detector_detections = self.detection_model.detect_objects(images,
                                                                         prompts,
                                                                         detection_threshold,
                                                                         bsz=32)

        # region_proposals, objectness, grounding_scores, class_embeds = self.owl.get_region_proposals(
        #     images, 16,
        #     n_boxes=-1,
        #     text_classifier=None,
        #     return_class_embeds=True,
        #     save_cache=True)
        #
        #
        #
        best_detections = [get_best_detection_per_class(detections) for detections in object_detector_detections]
        #
        # owl_embeds_per_class = []
        # for class_idx in range(len(prompts)):
        #     if class_idx == 0:
        #         for frame_idx in range(len(images)):
        #             cur_frame_class_detections = object_detector_detections[frame_idx][object_detector_detections[frame_idx].class_id == class_idx]
        #             if len(cur_frame_class_detections) > 0:
        #                 ious = box_iou(torch.tensor(cur_frame_class_detections.xyxy), torch.tensor(region_proposals[frame_idx]))
        #                 max_iou_idx = ious.argmax()

        # get box prompts:
        box_prompts = []
        all_scores = []
        pred_objects = []
        for i, detection in enumerate(best_detections):
            cur_frame_box_prompts = []
            cur_frame_scores = []
            for idx, obj in enumerate(prompts):
                if idx not in detection.class_id:
                    box = np.array([0, 0, 0, 0])
                    score = 0.0
                else:
                    box = detection.xyxy[detection.class_id == idx][0]
                    score = detection.confidence[detection.class_id == idx][0]
                cur_frame_box_prompts.append(box)
                cur_frame_scores.append(score)
                pred_objects.append(obj)
            box_prompts.append(cur_frame_box_prompts)
            all_scores.append(cur_frame_scores)

        box_prompts = np.stack(box_prompts, axis=0)
        all_scores = np.stack(all_scores, axis=0)

        surface_prompts = ["table top", "kitchen counter", "stove top", "hotplates", "counter top", "counter"]
        orig_objects = list(detector_prompts.keys())

        try:
            surface_boxes, surface_obj = self.get_surface_object(images, surface_object_prompts=surface_prompts,
                                                                 objects=objects)
            detected_surface = True
        except Exception as e:
            detected_surface = False

        if detected_surface:
            surface_box_idx = 0
            if surface_obj in detector_prompts.keys():
                surface_box_idx = np.where(np.array(list(detector_prompts.keys())) == surface_obj)[0][0]
                box_prompts[:, surface_box_idx] = surface_boxes
                orig_objects = np.delete(orig_objects, surface_box_idx)
                del self.object_manager.objects[surface_obj]
            else:
                box_prompts = np.concatenate([surface_boxes[:, None, :], box_prompts], axis=1)
                all_scores = np.concatenate([np.ones((len(images), 1)) * 0.5, all_scores], axis=1)
                prompts = [surface_obj] + prompts

        mask, pred_objects, ious = self.compute_masks(images, 16, box_prompts, pred_objects)

        if detected_surface:
            # postprocess surface masks

            max_size = mask.shape[2] * mask.shape[3] * 0.1
            # surface_masks = postprocess_masks(mask[:,[surface_box_idx]], max_size=max_size)
            surface_masks = mask[:, surface_box_idx]
            surface_masks_processed = postprocess_masks(mask[:, [surface_box_idx]], max_size=max_size)

            intrinsics_m3d = np.array(
                [intrinsic_parameters["fx"], intrinsic_parameters["fy"], intrinsic_parameters["cx"],
                 intrinsic_parameters["cy"]])
            depth, normals, conf = self.m3d.predict(np.array(images), intrinsics_m3d)
            up_normals = np.array(normals[:, :, :, 2] > 0.7)

            normal_agreement_mask = get_mask_agreement(up_normals, threshold=0.45)
            combined_surface_masks = get_mask_agreement(surface_masks, threshold=0.45)
            combined_surface_masks = \
                remove_small_regions(combined_surface_masks, area_thresh=max_size, mode="holes")[0]
            total_agreement_mask = np.logical_and(normal_agreement_mask, combined_surface_masks)
            total_agreement_mask = \
                remove_small_regions(total_agreement_mask, area_thresh=max_size, mode="islands")[0]

            box = np.array(masks_to_boxes(torch.tensor(total_agreement_mask)[None])).astype(int)

            self.object_manager.add_surface_object(surface_obj, "")
            self.object_manager.surface_object.boxes = box[0]
            self.object_manager.surface_object.mask = surface_masks_processed.squeeze(1)
            self.object_manager.surface_object.combined_mask = total_agreement_mask
            self.object_manager.surface_object.normals = up_normals

            box_prompts = np.delete(box_prompts, surface_box_idx, axis=1)
            all_scores = np.delete(all_scores, surface_box_idx, axis=1)
            prompts = np.delete(prompts, surface_box_idx)
            mask = np.delete(mask, surface_box_idx, axis=1)
        else:
            surface_obj = None
            box = None

        # remove surface obj boxes from box prompts

        self.compute_depth(images)

        class_mask_dict = {}
        for i, obj in enumerate(prompts):
            if i == (len(prompts) - 1):
                class_mask_dict["robot"] = {"box": box_prompts[:, i], "mask": mask[:, i], "score": all_scores[:, i]}
            else:
                class_mask_dict[orig_objects[i]] = {"box": box_prompts[:, i], "mask": mask[:, i],
                                                    "score": all_scores[:, i]}

        self.object_manager.add_object_detections(class_mask_dict, is_voted_temporal=False)

        self.propagate_masks_with_deva(images, vis_save_path)

        return class_mask_dict, surface_obj, box

    def label_frames(self, images, detection_threshold=0.04, score_threshold=0.1, vis_save_path=None,
                     enable_temporal_aggregation=True, is_short_horizon=False):

        if self.total_memory_gb > 20:
            bsz = 2
            obj_det_bsz = 4
            mask_bsz = 16
            region_proposal_bsz = 4
        else:
            bsz = 2
            obj_det_bsz = 2
            mask_bsz = 8
            region_proposal_bsz = 4


        n_boxes = 15
        
        #get current file dir as chache dir
        cache_dir = os.path.dirname(os.path.realpath(__file__))

        intrinsic_parameters = get_intrinsics(images, cache_dir=cache_dir)

        self.intrinsic_parameters = intrinsic_parameters

        if enable_temporal_aggregation:
            det_mask_subset_frames = []
            for i in range(0, len(images), self.cfg.deva_detection_every):
                for s in range(0, self.cfg.deva_n_voting_frames):
                    if i + s < len(images):
                        det_mask_subset_frames.append(i + s)

        else:
            det_mask_subset_frames = range(len(images))

        # Optical Flow Calculation:

        if is_short_horizon or not self.cfg.use_flow_for_movement:
            pass
            # self.flow_mask_fwd = np.zeros_like(images[:, :2])
            # self.flow_mask_bwd = np.zeros_like(images[:, :2])
            # self.flow_predictions = np.zeros((images.shape[0], images.shape[1], 2))
            # self.flow_predictions_bwd = np.zeros((images.shape[0], images.shape[1], 2))
        else:
            logging.info(f"Calculating optical flow for {len(images)} frames")
            self.flow_mask_fwd, self.flow_mask_bwd, self.flow_predictions, self.flow_predictions_bwd = self.flow_predictor.calc_optical_flow(
                images, 1, bsz)

            with torch.no_grad():
                self.flow_predictor.flow_model.cpu()

                torch.cuda.empty_cache()

        # State detection

        # Object detection

        detector_prompts = self.object_manager.get_object_detection_prompts()

        objects = list(detector_prompts.values())

        orig_class_names = list(detector_prompts.keys())

        ov_segmentation_prompts_dict = self.object_manager.get_object_segmentation_prompts()
        ov_orig_class_names = list(ov_segmentation_prompts_dict.keys())

        assert orig_class_names == ov_orig_class_names, "Object detection and segmentation classes do not align"

        if isinstance(self.detection_model, GroundingDINOHF):
            embodiment_prompt = self.cfg.embodiment_prompt_grounding_dino
        else:
            embodiment_prompt = self.cfg.embodiment_prompt_owl

        images_subset_temp = images[det_mask_subset_frames]

        self.image_det_frames = images_subset_temp

        object_detector_detections = self.detection_model.detect_objects(images_subset_temp,
                                                                         objects + [embodiment_prompt],
                                                                         threshold=detection_threshold,
                                                                         bsz=obj_det_bsz)

        region_proposals, objectness, grounding_scores, class_embeds = self.owl.get_region_proposals(
            images_subset_temp, region_proposal_bsz,
            n_boxes=n_boxes,
            text_classifier=None,
            return_class_embeds=True,
            save_cache=True)

        self.objectness = objectness
        self.region_proposals = region_proposals
        self.class_embeds = class_embeds

        # Check for undetected objects

        with torch.no_grad():
            self.detection_model.to_cpu()

            torch.cuda.empty_cache()

        ov_segmentation_prompts = list(ov_segmentation_prompts_dict.values())
        ov_segmentation_prompts += [self.cfg.embodiment_prompt_clipseg]
        self.ov_segmentation_model.precompute_text_embeddings(ov_segmentation_prompts)
        ov_segmentation_affinites = self.ov_segmentation_model.get_affinities(images_subset_temp, bsz=16)
        ov_segmentation_affinites = torch.sigmoid(ov_segmentation_affinites)

        robot_boxes = []
        robot_scores = []
        robot_affinities = ov_segmentation_affinites[:, -1]

        for det_idx, detection in enumerate(object_detector_detections):
            cur_robot_affinities = robot_affinities[det_idx]
            cur_valid_detections = detection[detection.class_id == len(objects)]
            areas = cur_valid_detections.area
            cur_valid_detections = cur_valid_detections[areas < 0.5 * images_subset_temp[det_idx].shape[1] *
                                                        images_subset_temp[det_idx].shape[0]]

            scores = cur_valid_detections.confidence
            boxes = cur_valid_detections.xyxy

            if len(scores) == 0:
                robot_boxes.append(np.array([0, 0, 0, 0]))
                robot_scores.append(np.array(0.))
                print(det_idx)
                continue
            affinities = []
            for box in boxes:
                cur_robot_affinities_filtered = cur_robot_affinities
                cur_robot_affinities_filtered[cur_robot_affinities_filtered < 0.25] = 0
                cur_aff = cut_box_from_matrix(np.array(box), cur_robot_affinities_filtered).sum()

                # normalize by area
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                cur_aff = cur_aff / box_area

                affinities.append(cur_aff)

            areas = cur_valid_detections.area

            normalized_areas = areas / np.max(areas)
            scores_with_area = 0.4 * cur_valid_detections.confidence + (0.1 * normalized_areas) + 0.4 * (
                np.array(affinities))
            best_detection_idx = np.argsort(scores_with_area)[-1]
            best_detection = cur_valid_detections.xyxy[best_detection_idx]
            robot_boxes.append(best_detection)
            robot_scores.append(cur_valid_detections.confidence[best_detection_idx])

        robot_boxes = np.stack(robot_boxes)
        robot_scores = np.stack(robot_scores)
        n_pred_robot_boxes = (robot_scores > 0).sum()
        enough_robot_detections = n_pred_robot_boxes > len(images_subset_temp) * 0.4

        if not enough_robot_detections:
            max_val = \
                ov_segmentation_affinites[:, -1].reshape(ov_segmentation_affinites[:, -1].shape[0], -1).max(dim=1)[0]
            threshold = np.minimum(max_val - 0.05, 0.4)
            ov_seg_robot_boxes = masks_to_boxes(ov_segmentation_affinites[:, -1] > threshold[:, None, None])
            scores = np.array([torch.mean(cur_aff[-1][cur_aff[-1] > 0.3]) for cur_aff in ov_segmentation_affinites])

            invalid_robot_boxes = robot_scores == 0
            robot_boxes[invalid_robot_boxes] = np.array(ov_seg_robot_boxes[invalid_robot_boxes]).astype(int)
            robot_scores[invalid_robot_boxes] = scores[invalid_robot_boxes]

        # remove robot
        ov_segmentation_affinites = ov_segmentation_affinites[:, :-1]

        # Normalize to 0,1

        ov_segmentation_affinites_normalized = (ov_segmentation_affinites -
                                                ov_segmentation_affinites.amin(axis=(-2, -1))[..., None, None]) / (
                                                       ov_segmentation_affinites.amax(axis=(-2, -1))[..., None, None] -
                                                       ov_segmentation_affinites.amin(axis=(-2, -1))[
                                                           ..., None, None])

        bsz, n_classes = ov_segmentation_affinites.shape[:2]

        # Ensembling
        bb_scores = []
        all_masks = []

        if self.object_manager.surface_object is not None and self.object_manager.surface_object.name in orig_class_names:
            surface_obj_idx = orig_class_names.index(self.object_manager.surface_object.name)
        else:
            surface_obj_idx = -1

        class_mask_dicts = []
        init_mask_dict = {
            obj: {"box": np.array([0, 0, 0, 0]), "score": 0., "mask": np.zeros_like(ov_segmentation_affinites[0, 0])}
            for obj in orig_class_names}

        for i in range(bsz):
            class_sums = []
            best_masks = []
            cur_class_mask_dict = init_mask_dict.copy()

            cur_frame_best_indices = []
            for j in range(n_classes):
                if (object_detector_detections[i].class_id == j).sum() == 0:
                    class_sums.append(None)
                    best_masks.append(None)
                    cur_frame_best_indices.append(None)
                    continue

                cur_dense_logits = ov_segmentation_affinites[i, j]

                if j == surface_obj_idx:
                    area_threshold = (cur_dense_logits.shape[-1] * cur_dense_logits.shape[-2])
                else:
                    area_threshold = (cur_dense_logits.shape[-1] * cur_dense_logits.shape[-2]) * 0.5
                cur_class_detections = object_detector_detections[i].xyxy[object_detector_detections[i].class_id == j]

                # plot_boxes_np(images[i], cur_class_detections, [orig_class_names[j]], return_image=True)

                cur_areas = object_detector_detections[i].area[object_detector_detections[i].class_id == j]
                cur_class_detections = cur_class_detections[cur_areas < area_threshold]
                cur_class_scores = object_detector_detections[i].confidence[object_detector_detections[i].class_id == j]
                cur_class_scores = cur_class_scores[cur_areas < area_threshold]

                masks = ov_segmentation_affinites_normalized[i, j]

                if len(cur_class_detections) == 0:
                    class_sums.append(None)
                    best_masks.append(None)
                    cur_frame_best_indices.append(None)
                    continue

                if self.cfg.enable_detection_ensembling:
                    best_box_idx, box_scores, mask_filters = get_box_dense_agreement(self.detection_model,
                                                                                     cur_class_detections,
                                                                                     cur_class_scores,
                                                                                     cur_dense_logits)
                    best_mask_filter = mask_filters[best_box_idx]
                else:

                    best_box_idx = np.argmax(cur_class_scores)
                    box_scores = cur_class_scores
                    best_mask_filter = ov_segmentation_affinites_normalized[i, j] > 0.5

                cur_frame_best_indices.append(best_box_idx)
                best_box = cur_class_detections[best_box_idx]
                best_score = box_scores[best_box_idx]

                best_mask = masks * best_mask_filter
                best_mask = best_mask > 0.5

                # if best_mask.sum() > 80 and best_mask.sum() < 0.3 * cur_areas[best_box_idx]:
                #     best_box = np.array(masks_to_boxes(best_mask[None])[0], dtype=int)

                best_masks.append(best_mask)
                class_sums.append(best_box)

                if best_score < score_threshold:
                    continue
                cur_dict = {"mask": np.array(best_mask), "score": best_score, "box": np.array(best_box)}

                cur_class_mask_dict[orig_class_names[j]] = cur_dict

                # class_sums.append(np.sum(area))

            # perform nms:
            all_boxes = np.array(
                [cur_class_mask_dict[obj]["box"] for obj in orig_class_names if obj in cur_class_mask_dict.keys()])
            all_scores = np.array(
                [cur_class_mask_dict[obj]["score"] for obj in orig_class_names if obj in cur_class_mask_dict.keys()])

            nms_indices = nms(torch.tensor(all_boxes, dtype=torch.float32),
                              torch.tensor(all_scores, dtype=torch.float32), 0.5)

            for class_idx in range(len(orig_class_names)):
                if class_idx not in nms_indices:
                    cur_class_mask_dict[orig_class_names[class_idx]] = {"box": np.array([0, 0, 0, 0]), "score": 0.,
                                                                        "mask": np.zeros_like(
                                                                            ov_segmentation_affinites[0, 0])}

            bb_scores.append(class_sums)
            all_masks.append(best_masks)
            class_mask_dicts.append(cur_class_mask_dict)

        class_mask_dict_transformed = list_of_dict_of_dict_to_dict_of_dict_of_list(class_mask_dicts)

        robot_masks = np.zeros_like(ov_segmentation_affinites[0, 0])
        robot_masks = np.repeat(robot_masks[None], len(images_subset_temp), axis=0)
        class_mask_dict_transformed["robot"] = {"box": robot_boxes, "score": robot_scores, "mask": robot_masks}

        box_prompts = []
        pred_objects = []
        for obj, obj_states in class_mask_dict_transformed.items():
            box_prompts.append(obj_states["box"])
            pred_objects.append(obj)

        box_prompts = np.stack(box_prompts, axis=1)

        if self.object_manager.surface_object is not None:
            self.detect_surface_object(images_subset_temp)

        obj_scores = np.array([class_mask_dict_transformed[obj]["score"] for obj in pred_objects]).T
        noise_scores_obj_det = zscore(obj_scores, axis=0) < -1

        # Check for undetected objects
        if self.cfg.check_for_undetected_objects:
            undetected_objects_mem, undetected_object_indices = check_for_undetected_objects(region_proposals,
                                                                                             objectness, class_embeds,
                                                                                             box_prompts,
                                                                                             images_subset_temp,
                                                                                             )
        else:
            undetected_objects_mem = None
        new_vlm_objects = []
        if undetected_objects_mem is not None:
            vocab_name = Path(self.cfg.vocab_file).stem

            owl_out_path = Path(self.cfg.vocab_file).parent / vocab_name / f"{self.owl.name}.pth"
            with open(self.cfg.vocab_file) as f:
                lvis_classes = f.read().splitlines()
                lvis_classes = [x[x.find(':') + 1:] for x in lvis_classes]
            owl_text_classifier = self.owl.get_text_classifier(lvis_classes, owl_out_path)

            undetected_objects_mem_for_concat = np.repeat(undetected_objects_mem[:, None], owl_text_classifier.shape[1],
                                                          axis=1)
            text_classifier_concat = np.concatenate([undetected_objects_mem_for_concat, owl_text_classifier], axis=0)
            all_boxes, all_objectnesses, all_scores, all_class_embeds = self.owl.get_region_proposals(
                self.image_det_frames, region_proposal_bsz, n_boxes=-1,
                text_classifier=torch.tensor(text_classifier_concat),
                reduction=None)

            all_scores_predefined_classes = all_scores[:, :, len(undetected_objects_mem):]
            all_scores = all_scores[:, :, :len(undetected_objects_mem)]

            best_boxes_indices = np.argmax(all_scores, axis=1)
            best_boxes = np.take_along_axis(all_boxes, best_boxes_indices[..., None], axis=1)

            best_boxes_owl_scores = np.take_along_axis(all_scores_predefined_classes, best_boxes_indices[..., None],
                                                       axis=1)
            best_boxes_owl_scores_summed = torch.sum(torch.nn.functional.sigmoid(torch.tensor(best_boxes_owl_scores)),
                                                     dim=0)
            max_predefined_classes_idx = np.argmax(best_boxes_owl_scores_summed, axis=-1)
            scores_predefined_classes = np.max(np.array(best_boxes_owl_scores_summed) / len(all_scores), axis=-1)

            classes = [lvis_classes[cls_idx].split(",")[0] for cls_idx in max_predefined_classes_idx]

            objectnesses_best_boxes = np.take_along_axis(all_objectnesses, best_boxes_indices, axis=1)

            best_scores = np.max(all_scores, axis=1)

            zscores = zscore(best_scores, axis=0)
            noise_scores = zscores < -1.4

            best_scores[noise_scores] = np.nan
            best_boxes[noise_scores] = np.zeros((len(best_boxes[noise_scores]), 4))
            objectnesses_best_boxes[noise_scores] = np.zeros((len(objectnesses_best_boxes[noise_scores]),))

            ious = []
            matched_classes = []

            if self.object_manager.surface_object is not None:
                pred_objects_with_surface = [self.object_manager.surface_object.name] + pred_objects
            else:
                pred_objects_with_surface = pred_objects

            for idx in range(len(best_boxes)):
                cur_boxes = best_boxes[idx]
                # plot_boxes_np(self.image_det_frames[idx], cur_boxes)
                cur_detected_boxes = box_prompts[idx]
                cur_detected_boxes[noise_scores_obj_det[idx, :]] = [0, 0, 0, 0]

                if self.object_manager.surface_object is not None:
                    cur_detected_boxes_with_surface = np.concatenate(
                        [self.object_manager.surface_object.boxes[None], cur_detected_boxes], axis=0)
                else:
                    cur_detected_boxes_with_surface = cur_detected_boxes

                iou = box_iou(torch.tensor(cur_boxes), torch.tensor(cur_detected_boxes_with_surface))
                max_iou = np.max(np.array(iou), axis=1)
                is_nan = np.isnan(max_iou)
                max_iou_idx = np.argmax(np.array(iou), axis=1)
                cur_matched_classes = np.array(pred_objects_with_surface)[max_iou_idx]
                cur_matched_classes[is_nan] = "None"
                matched_classes.append(cur_matched_classes)
                ious.append(max_iou)
            ious = np.array(ious)

            matched_classes = np.array(matched_classes)
            matched_classes[ious < 0.35] = None

            undet_objects_idx = []
            unsure_obj_idx = []
            for obj_idx in range(best_boxes.shape[1]):
                class_counts = dict(Counter(matched_classes[:, obj_idx]))
                if "robot" in class_counts.keys():
                    if class_counts["robot"] > 0.15 * len(best_boxes):
                        continue
                if "None" in class_counts.keys():
                    none_count = class_counts["None"]
                    if none_count == len(best_boxes):
                        max_non_none_count = 0
                    else:
                        max_non_none_count = np.max([class_counts[key] for key in class_counts.keys() if key != "None"])
                    if none_count > 0.5 * len(
                            best_boxes) and max_non_none_count < 0.5 * none_count and objectnesses_best_boxes[:,
                                                                                      obj_idx].mean() > 0.25:
                        undet_objects_idx.append(obj_idx)
                        continue

                    max_name = max(class_counts, key=class_counts.get)
                    max_count = class_counts[max_name]
                    if max_count < 0.9 * (len(best_boxes) - none_count):
                        unsure_obj_idx.append(obj_idx)

            for obj_idx in undet_objects_idx:
                new_vlm_objects.append(classes[obj_idx])

        if len(new_vlm_objects) > 0:
            for undet_obj_idx, obj_name in zip(undet_objects_idx, new_vlm_objects):
                pred_objects.append(obj_name)

                obj_properties = {"name": obj_name, "movable": True, "container": False, "interactable": True,
                                  "states": [], "color": ""}

                self.object_manager.add_object(obj_name, obj_properties)

                box_prompts = np.concatenate([box_prompts, best_boxes[:, undet_obj_idx][:, None]], axis=1)

                class_mask_dict_transformed[obj_name] = {"box": best_boxes[:, undet_obj_idx],
                                                         "score": scores_predefined_classes[obj_idx] / 2, "mask": None}

        surface_prompts = ["table", "floor", "kitchen counter", "stove top", "hotplates", "counter top", "counter"]

        surface_boxes, surface_obj = self.get_surface_object(images_subset_temp, surface_object_prompts=surface_prompts,
                                                             objects=objects)

        surface_box_idx = 0
        if surface_obj in class_mask_dict_transformed.keys():
            surface_box_idx = np.where(np.array(pred_objects) == surface_obj)[0][0]
            del self.object_manager.objects[surface_obj]
        else:
            box_prompts = np.concatenate([surface_boxes[:, None, :], box_prompts], axis=1)
            pred_objects = [surface_obj] + pred_objects

        max_size = images.shape[1] * images.shape[2] * 0.2

        mask, pred_objects, ious = self.compute_masks(images_subset_temp, mask_bsz, box_prompts, pred_objects)

        # remove surface obj from masks and pred objects
        pred_objects = list(np.delete(pred_objects, surface_box_idx))

        surface_masks = mask[:, surface_box_idx]
        surface_masks_processed = postprocess_masks(mask[:, [surface_box_idx]], max_size=max_size)

        mask = np.delete(mask, surface_box_idx, axis=1)

        # remove
        for obj in pred_objects:
            class_mask_dict_transformed[obj]["mask"] = mask[:, pred_objects.index(obj)]

        if len(det_mask_subset_frames) != len(images):
            class_mask_dict_transformed = interleave_detections(class_mask_dict_transformed,
                                                                det_mask_subset_frames, len(images))

        max_size = mask.shape[2] * mask.shape[3] * 0.03

        intrinsics_m3d = np.array(
            [intrinsic_parameters["fx"], intrinsic_parameters["fy"], intrinsic_parameters["cx"],
             intrinsic_parameters["cy"]])
        depth, normals, conf = self.m3d.predict(np.array(images), intrinsics_m3d)
        up_normals = np.array(normals[:, :, :, 2] > 0.7)

        normal_agreement_mask = get_mask_agreement(up_normals, threshold=0.45)
        combined_surface_masks = get_mask_agreement(surface_masks, threshold=0.2)
        combined_surface_masks = \
            remove_small_regions(combined_surface_masks, area_thresh=1000, mode="holes")[0]
        total_agreement_mask = np.logical_and(normal_agreement_mask, combined_surface_masks)
        total_agreement_mask = \
            remove_small_regions(total_agreement_mask, area_thresh=max_size, mode="islands")[0]

        box = np.array(masks_to_boxes(torch.tensor(total_agreement_mask)[None])).astype(int)

        self.object_manager.add_surface_object(surface_obj, "")
        self.object_manager.surface_object.boxes = box[0]
        self.object_manager.surface_object.mask = surface_masks_processed.squeeze(1)
        self.object_manager.surface_object.combined_mask = total_agreement_mask
        self.object_manager.surface_object.normals = up_normals

        # remove surface obj boxes from class_mask_dict_transformed
        if surface_obj in class_mask_dict_transformed.keys():
            class_mask_dict_transformed.pop(surface_obj)

        self.object_manager.add_object_detections(class_mask_dict_transformed)

        if enable_temporal_aggregation:
            self.propagate_masks_with_deva(images, vis_save_path)

        self.compute_depth(images)

        gripper_locations = self.get_gripper_location(images)

        self.object_manager.robot.gripper_locations = gripper_locations

        # del self.detection_model
        # del self.ov_segmentation_model

        # self.models_to_cpu()

        torch.cuda.empty_cache()

        return class_mask_dicts

    def models_to_cpu(self):

        self.detection_model.to_cpu()
        self.ov_segmentation_model.to_cpu()
        self.flow_predictor.to_cpu()
        self.depth_predictor.to_cpu()
        self.owl.to_cpu()
        torch.cuda.empty_cache()

    def compute_depth(self, images):

        start = time.time()
        depth_predictions = self.depth_predictor.predict_depth(images)
        end = time.time()
        print(f"Depth prediction took {end - start} seconds")

        self.depth_predictions = depth_predictions

    def get_gripper_location(self, batch, robot_fixed_point="bottom_right", use_flow=False):
        # Resize to square

        robot_masks = self.object_manager.robot.mask
        image_dims = robot_masks.shape[1:]

        def sq(h, w):
            return np.concatenate(
                [(np.arange(w * h).reshape(h, w) % w)[:, :, None], (np.arange(w * h).reshape(h, w) // w)[:, :, None]],
                axis=-1
            )

        pos = sq(*image_dims)
        weight = pos[:, :, 0] + pos[:, :, 1]

        def mask_to_pos_naive(mask):

            min_pos = np.argmax((weight * mask).flatten())

            return min_pos % image_dims[0] - (image_dims[0] / 16), min_pos // image_dims[0] - (image_dims[0] / 24)

        if not use_flow:
            gripper_positions = [mask_to_pos_naive(mask) for mask in robot_masks]
            return np.stack(gripper_positions)

        flow_interval = 1
        flow_mask_fwd, flow_fwd = self.flow_mask_fwd, self.flow_predictions

        if flow_mask_fwd is not None and self.cfg.use_flow_for_movement:

            robot_masks_filtered = flow_mask_fwd * robot_masks[::flow_interval][:-1]
            robot_masks_filtered = robot_masks_filtered.astype(bool)
        else:
            robot_masks_filtered = robot_masks[::flow_interval][:-1].astype(bool)
        # get max movement in flow mask:

        reverse_mask = self.cfg.robot_location_camera_frame == "front"

        if reverse_mask:
            fac = -1
        else:
            fac = 1

        if self.depth_predictions is None:
            depth_predictions = self.depth_predictor.predict_depth(batch["rgb_static"])
            self.depth_predictions = depth_predictions
        if self.depth_predictions is not None:
            eroded_robot_masks = np.array(
                [cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2) for mask in
                 robot_masks_filtered])
            depth_masked = apply_mask_to_image(fac * self.depth_predictions[:-1], eroded_robot_masks.astype(bool))
            # max_depth = np.max(depth_masked, axis=(1, 2))

            max_pixel_indices = np.zeros((depth_masked.shape[0], 2), dtype=int)

            for i in range(depth_masked.shape[0]):
                max_pixel_indices[i] = np.unravel_index(np.argmax(depth_masked[i]), depth_masked[i].shape)

        flow_fwd = np.concatenate([flow_fwd, np.zeros_like(flow_fwd[-1])[None, ...]], axis=0)

        if self.cfg.use_flow_for_movement:
            movement = np.linalg.norm(flow_fwd * robot_masks[..., None], axis=-1)

            movement_all = np.linalg.norm(flow_fwd, axis=-1)
        else:
            movement = np.zeros_like(flow_fwd)
            movement_all = np.zeros_like(flow_fwd)

        movement_weighted = create_weight_mask(movement[0], mode="bottom_left", horizontal_weight=1, vertical_weight=1)[
                                None] * movement

        depth_masked = np.concatenate([depth_masked, np.ones_like(depth_masked[-1])[None, ...]], axis=0)

        movement_weighted = movement_weighted * depth_masked

        movement_weighted = depth_masked * \
                            create_weight_mask(movement[0], mode=self.cfg.robot_endeffector_location_weight,
                                               horizontal_weight=2, vertical_weight=2)[
                                None]

        max_movement_idx = np.column_stack(
            np.unravel_index(np.argmax(movement_weighted.reshape(movement_weighted.shape[0], -1), axis=1),
                             movement_weighted[0].shape))

        # y,x to x,y 
        max_movement_idx = max_movement_idx[:, ::-1]

        # Apply heavy smoothing:
        # smoothed_movement = pd.DataFrame(max_movement_idx).ewm(alpha=0.5, min_periods=1).mean().shift(-4).values
        smoothed_movement = max_movement_idx

        return smoothed_movement
    
    
    
    def create_prompts(self, keystates, batch, scene_graphs, keystate_reasons, keystate_objects,
                       possible_actions=None, open_ended=True, n_keystates_to_aggregate=1,
                       aggregate_keystates=1):

        last_keystate_idx = 0

        if possible_actions is None:
            possible_actions = self.ds.possible_actions

        objects = list(self.object_manager.objects.keys())
        task_nlp_map = {task: task for task in possible_actions}

        # Get indices where scene graph changed:
        has_scene_graph_change = []

        for i in range(len(scene_graphs) - 1):
            if scene_graphs[i] != scene_graphs[i + 1]:
                has_scene_graph_change.append(i + 1)
        has_scene_graph_change = np.array(has_scene_graph_change)

        last_pred_index = 0
        last_prompt_nl = ""

        prompts = []

        # small_granularity_step_size = np.mean(np.diff(keystates)) // 4
        # n_samples_small_gran = int(len(batch["rgb_static"]) // small_granularity_step_size)
        # small_granularity_keystates = np.linspace(0, len(batch["rgb_static"]) - 1, n_samples_small_gran).astype(
        #     np.int32)
        # small_granularity_keystates = np.array(small_granularity_keystates)[1:]
        # robot_movements = []

        # for ks_idx,ks in enumerate(small_granularity_keystates):
        #     if ks_idx == 0:
        #         start_idx = 0
        #     else:
        #         start_idx = small_granularity_keystates[ks_idx - 1]
        #     end_idx = ks
        #     #robot_movement = self.get_robot_movement(start_idx, end_idx, batch["rgb_static"])
        # 
        #     gripper_start_location, gripper_start_location_index = self.object_manager.robot.get_last_known_gripper_location(
        #         start_idx, range=2)
        #     gripper_end_location, gripper_end_location_index = self.object_manager.robot.get_last_known_gripper_location(
        #         end_idx, range=2)
        # 
        #     gripper_locations = np.array([gripper_start_location, gripper_end_location])
        # 
        #     #gripper_movement = get_object_movement_nl(gripper_locations)
        #     #robot_movements.append(gripper_movement)

        # robot_movement = self.get_robot_movement(start_idx, end_idx, batch["rgb_static"])

        all_nl_reasons = []
        for keystate_idx, ks in enumerate(tqdm(keystates)):
            try:

                # ks = ks + 2

                ks = min(ks, len(batch["rgb_static"]) - 1)

                if keystate_idx == 0:
                    last_keystate_idx = 0
                if ks < self.cfg.prompt_interval:
                    last_keystate_idx = ks
                    continue
                length = ks - last_keystate_idx
                length = max(self.cfg.prompt_interval * 2, length)
                length = min(20, length)
                start_idx = max(0, ks - length)

                ks = int(ks)

                last_keystate_idx = ks

                cur_traj_scene_graph_end_idx = min((ks // self.cfg.scene_graph.interval) + 1, len(scene_graphs) - 1)

                # Check if there is a scene graph change in near future, if yes set end index to that index
                dists = np.abs(has_scene_graph_change - cur_traj_scene_graph_end_idx)
                if np.any(dists <= 2):
                    new_index = has_scene_graph_change[np.argmin(dists)] + 1
                    if new_index > cur_traj_scene_graph_end_idx:
                        cur_traj_scene_graph_end_idx = new_index

                cur_traj_scene_graph_end_idx = min(cur_traj_scene_graph_end_idx, len(scene_graphs) - 1)
                cur_traj_scene_graph_start_idx = min(start_idx // self.cfg.scene_graph.interval,
                                                     cur_traj_scene_graph_end_idx - self.cfg.prompt_interval)

                scene_graphs_subset_keystate = scene_graphs[
                                               cur_traj_scene_graph_start_idx:cur_traj_scene_graph_end_idx]
                scene_graphs_subset_prompt_idx = np.linspace(0, len(scene_graphs_subset_keystate) - 1,
                                                             self.cfg.prompt_interval).astype(np.int32)

                all_nl_actions = [task_nlp_map[action] for action in possible_actions]
                filtered_actions = all_nl_actions


                filtered_actions += ["None"]
                task_list_nl = ["-" + task for task in filtered_actions]

                object_movements = self.object_manager.get_object_movements_in_interval(cur_traj_scene_graph_start_idx,
                                                                                        cur_traj_scene_graph_end_idx,
                                                                                        keystate_objects[keystate_idx])

                object_movements = [object_movements[obj] for obj in object_movements if len(object_movements[obj]) > 0]

                scene_graphs_start_average_index = max(0, cur_traj_scene_graph_start_idx - 4)
                start_avg_indices = (scene_graphs_start_average_index, cur_traj_scene_graph_start_idx)
                if scene_graphs_start_average_index == 0:
                    scene_graphs_start_average_index = cur_traj_scene_graph_start_idx + 3
                    start_avg_indices = (cur_traj_scene_graph_start_idx, scene_graphs_start_average_index)

                # averaged_sg_movement = get_averaged_movement_between_sg(
                #     scene_graphs[start_avg_indices[0]:start_avg_indices[1]],
                #     scene_graphs[cur_traj_scene_graph_end_idx:min(cur_traj_scene_graph_end_idx + 3, len(scene_graphs))])

                movements_nl_2d = ",".join(object_movements)

                # state_changes = self.object_manager.get_state_changes_in_interval(cur_traj_scene_graph_start_idx,
                #                                                                   cur_traj_scene_graph_end_idx)

                averaged_init_sg = average_scene_graphs(scene_graphs[start_avg_indices[0]:start_avg_indices[1]])

                averaged_final_sg = average_scene_graphs(scene_graphs[cur_traj_scene_graph_end_idx - 2: min(
                    cur_traj_scene_graph_end_idx + 3, len(scene_graphs))])

                max_conf_obj = keystate_objects[keystate_idx]
                end_idx = ks

                # robot_movement = self.get_robot_movement(start_idx, end_idx, batch["rgb_static"])

                start_box = self.object_manager.objects[max_conf_obj].get_last_known_box(start_idx)

                end_box = self.object_manager.objects[max_conf_obj].get_last_known_box(end_idx)

                surface_box = self.object_manager.surface_object.boxes
                if start_box is None or end_box is None:
                    pos_on_surface_prompt = None
                else:
                    if self.surface_transform_matrix is not None:
                        img_shape = batch["rgb_static"].shape[1:3]

                        # transform boxes to surface frame
                        start_box_t = transform_bbox(start_box, self.surface_transform_matrix)
                        end_box_t = transform_bbox(end_box, self.surface_transform_matrix)
                        surface_box_t = transform_bbox(surface_box, self.surface_transform_matrix)
                        warped_image = cv2.warpPerspective(np.array(batch["rgb_static"][start_idx]),
                                                           self.surface_transform_matrix,
                                                           (img_shape[1], img_shape[0]))

                        surface_box_t = np.array([0, 0, img_shape[1], img_shape[0]])

                        start_box = start_box_t
                        end_box = end_box_t
                        surface_box = surface_box_t

                    init_pos_on_surface = get_nl_position_on_surface(start_box, surface_box).strip()
                    end_pos_on_surface = get_nl_position_on_surface(end_box, surface_box).strip()

                    pos_on_surface_prompt = None
                    if init_pos_on_surface != end_pos_on_surface:
                        pos_on_surface_prompt = f"Global position changes: {max_conf_obj} moved from {init_pos_on_surface} of {self.object_manager.surface_object.name} to {end_pos_on_surface} of {self.object_manager.surface_object.name}"

                if self.cfg.enable_object_centric_relations:
                    init_sg_obj_edges = {key: edge for key, edge in averaged_init_sg.edges.items() if
                                         edge.start.name in keystate_objects[keystate_idx]}
                    averaged_init_sg.edges = init_sg_obj_edges
                    final_sg_obj_edges = {key: edge for key, edge in averaged_final_sg.edges.items() if
                                          edge.start.name in keystate_objects[keystate_idx]}
                    averaged_final_sg.edges = final_sg_obj_edges

                cur_reasons = {key: reason[keystate_idx] for key, reason in keystate_reasons.items()}

                cur_reasons["pos_on_surface"] = pos_on_surface_prompt

                prompt_nl_reasons = create_observation_prompt(averaged_final_sg, averaged_init_sg, cur_reasons,
                                                              movements_nl_2d)

                all_nl_reasons.append(prompt_nl_reasons)
                
                all_nl_reasons = "\n".join(all_nl_reasons)

                task_list_nl = "\n".join(task_list_nl)
                task_list_nl = task_list_nl.replace(",", "")

                prompt_nl = self.llm.get_nl_prompt_keystate_reasons(task_list_nl, prompt_nl_reasons,
                                                                    open_ended=open_ended)
                
                prompt_nl = get_simple_prompt_nl_reasons_gpt(all_nl_reasons) 

                prompts.append(prompt_nl)

            except FileNotFoundError as e:
                print(e)
                prompts.append(None)
                continue
        all_nl_reasons = np.array(all_nl_reasons)
        prompts_aggregated = []
        if aggregate_keystates != 1:
            prompts_aggregated = []
            for i in range(0, len(all_nl_reasons) - aggregate_keystates):
                cur_reasons = all_nl_reasons[i:i + aggregate_keystates]
                concat_reasons_nl = ""
                for task_idx, reason in enumerate(cur_reasons):
                    concat_reasons_nl += f"Task {task_idx + 1}: \n {reason}\n"
                prompt_nl = self.llm.get_nl_prompt_keystate_reasons(task_list_nl, concat_reasons_nl,
                                                                    open_ended=open_ended, aggregated=True)
                prompts_aggregated.append(prompt_nl)

        return prompts, prompts_aggregated
    
    def get_short_horizon_changes(self, all_data, scene_graphs, keystate_indices, gripper_action=None,
                                  get_gripper_obj_proximity=True):

        keystate_reasons = []
        scores = []
        objects = []

        img_shape = all_data[0].shape
        for i, idx in enumerate(keystate_indices):
            sg_start_idx = i
            sg_end_idx = i + 1
            start_idx = idx[0]
            end_idx = idx[1]

            start_sg = scene_graphs[sg_start_idx]
            end_sg = scene_graphs[sg_end_idx]

            score_dict = {}

            object_movements = self.object_manager.get_object_dict_in_interval(start_idx,
                                                                               end_idx - 1)

            object_movements_nl = self.object_manager.get_object_movements_in_interval(start_idx, end_idx - 1,
                                                                                       keystate_idx=i)

            all_movements = list(object_movements.values())
            all_movement_lengths = np.linalg.norm(np.array(all_movements), axis=1)
            max_movement_length = np.max(all_movement_lengths)
            scaled_movements = all_movement_lengths / np.max(all_movement_lengths)

            all_movement_lengths[all_movement_lengths < 20] = 0

            if all_movement_lengths.sum() == 0:
                for obj, movement in object_movements.items():
                    score_dict[obj] = 0
            else:
                for obj, movement in object_movements.items():
                    logging.info(f"Movement of {obj} is {movement}")
                    score_dict[obj] = np.linalg.norm(movement) / np.max(all_movement_lengths)

            if get_gripper_obj_proximity:
                gripper_proximity_nl = {}
                gripper_prox_predictor = GripperPosKeystatePredictor("gripper_pos")
                data = {"object_manager": self.object_manager}
                gripper_object_proximity = gripper_prox_predictor.get_in_contact(data)

                n_prox_frames = np.sum(gripper_object_proximity, axis=0)

                # create weighting of len frames, where later proximit is weighted more:
                weight = np.linspace(0.2, 1.8, len(gripper_object_proximity))

                scores_gripper_proximity = gripper_object_proximity * weight[:, None]

                normed_scores = np.sum(scores_gripper_proximity / weight.sum(), axis=0)

                normed_scores += 0.5 * n_prox_frames / len(gripper_object_proximity)

                keys = list(self.object_manager.objects.keys())
                avg_sizes = [self.object_manager.objects[keys[key_idx]].mask.sum(axis=(1, 2)).mean() for key_idx in
                             range(len(keys))]
                avg_sizes_normed = [max(0.3, 1 - (avg_size / np.max(avg_sizes))) for avg_size in avg_sizes]
                for key_idx in range(len(keys)):

                    if n_prox_frames[key_idx] > 0.3 * len(gripper_object_proximity):
                        gripper_proximity_nl[keys[
                            key_idx]] = f"Gripper in contact with {keys[key_idx]} for {n_prox_frames[key_idx]} frames"
                    score_dict[keys[key_idx]] += (normed_scores[key_idx] * avg_sizes_normed[key_idx])

            relation_changes, change_scores = get_scene_graph_changes(start_sg, end_sg, start_idx)

            for obj, change in relation_changes.items():
                score_dict[obj] += 0.4

            state_changes = self.object_manager.get_state_changes_in_interval(start_idx, end_idx)

            for obj, change in state_changes.items():
                score_dict[obj] += 0.8

            max_conf_obj = max(score_dict, key=score_dict.get)
            max_score = score_dict[max_conf_obj]
            scores.append(max_score)

            if max_score == 0:
                keystate_reasons.append("None")
                objects.append(None)
                continue

            start_box = self.object_manager.objects[max_conf_obj].get_last_known_box(start_idx)

            end_box = self.object_manager.objects[max_conf_obj].get_last_known_box(end_idx)

            if self.object_manager.surface_object is not None:
                surface_box = self.object_manager.surface_object.boxes
            else:
                surface_box = None

            pos_on_surface_prompt = None
            if self.surface_transform_matrix is not None and start_box is not None and end_box is not None and surface_box is not None:
                # transform boxes to surface frame
                start_box_t = transform_bbox(start_box, self.surface_transform_matrix)
                end_box_t = transform_bbox(end_box, self.surface_transform_matrix)
                surface_box_t = np.array([0, 0, img_shape[1], img_shape[0]])

                start_box = start_box_t
                end_box = end_box_t
                surface_box = surface_box_t

                init_pos_on_surface = get_nl_position_on_surface(start_box, surface_box).strip()
                end_pos_on_surface = get_nl_position_on_surface(end_box, surface_box).strip()

                if init_pos_on_surface != end_pos_on_surface:
                    pos_on_surface_prompt = f"Global position changes: {max_conf_obj} moved from {init_pos_on_surface} of {self.object_manager.surface_object.name} to {end_pos_on_surface} of {self.object_manager.surface_object.name}"

            prompt_nl_relation = None
            if max_conf_obj in relation_changes.keys():
                obj_relation_changes = relation_changes[max_conf_obj][0]
                final_rel = obj_relation_changes.split("Final obj")[-1]
                if "inside" in final_rel:
                    pos_on_surface_prompt = None
                prompt_nl_relation = obj_relation_changes

            if max_movement_length > 6:
                object_movement_nl = object_movements_nl[max_conf_obj]
                movements_nl_2d = "Object movement:" + object_movement_nl
            else:
                movements_nl_2d = None

            objects.append(max_conf_obj)

            state_changes_nl = None
            if max_conf_obj in state_changes.keys():
                before_state = state_changes[obj][0]
                after_state = state_changes[obj][1]
                state_changes_nl = (f"State changes: {obj} changed from {before_state} to {after_state}\n")

            if get_gripper_obj_proximity and max_conf_obj in gripper_proximity_nl.keys():
                gripper_proximity_nl = gripper_proximity_nl[max_conf_obj]
            else:
                gripper_proximity_nl = None

            reasons = {"movement": movements_nl_2d, "global_pos_changes": pos_on_surface_prompt,
                       "scene_graph": prompt_nl_relation, "state": state_changes_nl,
                       "gripper_proximity": gripper_proximity_nl}

            if gripper_action is not None:
                init_gripper_closed = "closed" if gripper_action[0] == 0 else "open"
                final_gripper_closed = "closed" if gripper_action[-1] == 0 else "open"

                gripper_action_nl = f"Initiallly, the gripper was {init_gripper_closed}. At the end, the gripper was {final_gripper_closed}"
                reasons["gripper"] = gripper_action_nl

            prompt_nl_reasons = ""
            for method, keystate_reason in reasons.items():
                cur_reason = keystate_reason
                if cur_reason is not None:
                    prompt_nl_reasons += cur_reason + '\n'

            keystate_reasons.append(prompt_nl_reasons)

        return keystate_reasons, scores, objects

    def create_scene_graphs(self, batch, vis_path, gripper_cam_labels=None, subset_indices=None,
                            surface_det_images=None,
                            ):

        scene_graphs = []

        tcp_center_gripper_screen = None

        object_mask_areas = self.object_manager.get_object_mask_areas()
        init_scene_graph = init_scene_dist_graph(list(self.object_manager.objects.keys()), batch["rgb_static"][0],
                                                 object_mask_areas)
        init_scene_graph.flow_raw = np.zeros((1, batch["rgb_static"][0].shape[0], batch["rgb_static"][0].shape[1], 2))
        flow_predictions  = self.flow_predictions

        if flow_predictions is not None:
            flow_predictions = np.concatenate([flow_predictions, np.zeros_like(flow_predictions[-1:])], axis=0)
        else:
            flow_predictions = [None] * self.object_manager.robot.mask.shape[0]
        # else:
        #     flow_predictions = np.zeros(
        #         (batch["rgb_static"].shape[0], batch["rgb_static"][0].shape[0], batch["rgb_static"][0].shape[1], 2))

        if self.cfg.use_depth and self.depth_predictions is None:
            raise ValueError("Depth predictions are None")

        depth_predictions = self.depth_predictions
        intrinsic_parameters = self.intrinsic_parameters
        # res = cv2.solvePnPRansac(np.array(gripper_pos), np.array(gripper_pos_2d), camera_mat, None)

        
        
        
        n_average = min(16, len(batch["rgb_static"]))

        surface_object_prompts = ["table", "kitchen counter", "stove top", "hotplates", "counter top", "counter"]
        if self.object_manager.surface_object is not None:
            surface_object_prompts += [
                self.object_manager.surface_object.color + " " + self.object_manager.surface_object.name]

        image_indices = np.linspace(0, len(batch["rgb_static"]) - 1, n_average).astype(int)
        average_images = batch["rgb_static"][image_indices]
        surface_object_prompts = ["table", "floor", "kitchen counter", "stove top", "hotplates", "counter top"]
        
        
        
        
        surface_transform_config = SurfaceTransformConfig(
            use_depth=True,
            canonicalize=self.cfg.canonicalize,
        )
        
        if self.cfg.use_depth:
            transformer = SurfaceTransformer(
                config=surface_transform_config,
                detection_model=self.detection_model,
                segmentation_model=self.segmentation_model,
                m3d=self.m3d,
                object_manager=self.object_manager,
                intrinsic_parameters=self.intrinsic_parameters
            )
            
            final_transform, homography_transform = transformer.compute_transform(
                average_images=average_images,
                surface_object_prompts=surface_object_prompts,
                depth_predictions=depth_predictions,
                image_indices=image_indices,
                vis_path=vis_path
            )
        
            final_inverse_transform = np.linalg.inv(final_transform)
            
            self.surface_transform_matrix = homography_transform
            
        else:
            all_masks = []
            combined_masks = np.logical_or.reduce(all_masks).astype(np.uint8)
            homography_transform = get_homography_transform(combined_masks)
            self.surface_transform_matrix = homography_transform

        all_obj_centers = {}

        for obj in self.object_manager.objects:
            all_obj_centers[obj] = []

        for i in tqdm(range(len(batch["rgb_static"])), position=1, desc="Creating Scene Graphs"):

            if subset_indices is not None:
                global_idx = subset_indices[i]
            else:
                global_idx = i

            offset = 0 if global_idx == len(self.object_manager.robot.mask) - 1 or subset_indices is None else 1

            obj_seg_dict = self.object_manager.get_object_stats_frame(global_idx + offset)

            cur_tcp_center = None

            # if some key is missing, take the last known location:
            if i >= 1:
                last_frame_obj_seg_dict = self.object_manager.get_object_stats_frame(global_idx - 1)
                if set(obj_seg_dict.keys()) != set(last_frame_obj_seg_dict.keys()):
                    for key in last_frame_obj_seg_dict.keys():
                        if key not in obj_seg_dict.keys():
                            obj_seg_dict[key] = last_frame_obj_seg_dict[key]

            obj_seg_dict["robot"] = self.object_manager.get_robot_stats_frame(i)["robot"]

            if self.object_manager.robot.gripper_locations is not None:
                gripper_pos = self.object_manager.robot.gripper_locations[i]
                if gripper_pos.sum() == 0:
                    gripper_pos = None
            else:
                gripper_pos = None

            scene_graph_step, obj_centers = create_scene_dist_graph(obj_seg_dict, object_mask_areas,
                                                                    batch["rgb_static"][i],
                                                                    flow_predictions[global_idx], self.cfg,
                                                                    gripper_pos=gripper_pos,
                                                                    tcp_center_screen=cur_tcp_center,
                                                                    gt_depth=depth_predictions[
                                                                        i] if self.cfg.use_depth else None,
                                                                    transformation=final_inverse_transform,
                                                                    intrinsic_parameters=intrinsic_parameters,
                                                                    last_scene_graph=scene_graphs[
                                                                        i - 1] if i > 0 else init_scene_graph)

            for obj in all_obj_centers.keys():
                if obj not in obj_centers.keys():
                    all_obj_centers[obj].append(np.array([None, None, None]))
                else:
                    all_obj_centers[obj].append(obj_centers[obj])

            if scene_graph_step is False:
                return False

            if gripper_cam_labels is not None:
                scene_graph_step.gripper_cam_labels = gripper_cam_labels[i]

            scene_graphs.append(scene_graph_step)

        self.last_scene_graphs.append(scene_graphs)

        for obj, centers in all_obj_centers.items():
            all_obj_centers[obj] = np.array(centers)
            if obj == "robot":
                continue
            self.object_manager.objects[obj].pos_3d = all_obj_centers[obj]

        return scene_graphs
