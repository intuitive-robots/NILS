import copy
import warnings
from os import path
from typing import Dict, Iterable, List, Literal, Optional

import cv2
import numpy as np
import torch
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.inference.frame_utils import FrameInfo
from deva.inference.image_feature_store import ImageFeatureStore
from deva.inference.memory_manager import MemoryManager
from deva.inference.object_utils import *
from deva.inference.result_utils import ResultSaver
from deva.model.network import DEVA
from deva.utils.tensor_utils import pad_divide_by, unpad

# try:
#     from groundingdino.util.inference import Model as GroundingDINOModel
# except ImportError:
#     # not sure why this happens sometimes
#     from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import SamPredictor
from torchvision.ops import masks_to_boxes

from nils.segmentator.deva_object_manager import ObjectManager
from nils.segmentator.deva_utils import ObjectInfo
from nils.utils.utils import (
    clean_outlier_labels,
)

import warnings

from nils.segmentator.segment_merging import match_and_merge
from nils.specialist_models.detectors.utils import compute_iou
from nils.utils.utils import get_return_masks_dict

warnings.filterwarnings('ignore')



class DEVAInferenceCoreCustom:
    def __init__(self,
                 network: DEVA,
                 config: Dict,
                 *,
                 image_feature_store: ImageFeatureStore = None):
        self.network = network
        self.mem_every = config['mem_every']
        self.enable_long_term = config['enable_long_term']
        self.chunk_size = config['chunk_size']
        self.max_missed_detection_count = config.get('max_missed_detection_count')
        self.max_num_objects = config.get('max_num_objects')
        self.config = config

        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory = MemoryManager(config=config)
        self.object_manager = ObjectManager()

        self.penalty_list = []

        self.id_to_tmp_id_hist = []
        self.class_to_obj_id_hist = []
        self.mask_bool_hist = []

        if image_feature_store is None:
            self.image_feature_store = ImageFeatureStore(self.network)
        else:
            self.image_feature_store = image_feature_store

        self.last_mask = None

        self.last_mask_bool = None

        # for online/semi-online processing
        self.frame_buffer = []

        self.masks = []
        self.obj_to_tmp_ids = []

    def enabled_long_id(self) -> None:
        # short id is the default, 1~255, converted to a grayscale mask with a palette
        # long id is usually used by panoptic segmnetation, 255~255**3, converted to a RGB mask
        self.object_manager.use_long_id = True

    @property
    def use_long_id(self):
        return self.object_manager.use_long_id

    def _add_memory(self,
                    image: torch.Tensor,
                    ms_features: Iterable[torch.Tensor],
                    prob: torch.Tensor,
                    key: torch.Tensor,
                    shrinkage: torch.Tensor,
                    selection: torch.Tensor,
                    *,
                    is_deep_update: bool = True) -> None:
        # image: 1*3*H*W
        # ms_features: from the key encoder
        # prob: 1*num_objects*H*W, 0~1
        if prob.shape[1] == 0:
            # nothing to add
            warnings.warn('Empty object mask!', RuntimeWarning)
            return

        self.memory.initialize_sensory_if_needed(key, self.object_manager.all_obj_ids)
        value, sensory = self.network.encode_mask(image,
                                                  ms_features,
                                                  self.memory.get_sensory(
                                                      self.object_manager.all_obj_ids),
                                                  prob,
                                                  is_deep_update=is_deep_update,
                                                  chunk_size=self.chunk_size)
        self.memory.add_memory(key,
                               shrinkage,
                               value,
                               self.object_manager.all_obj_ids,
                               selection=selection)
        self.last_mem_ti = self.curr_ti
        if is_deep_update:
            self.memory.update_sensory(sensory, self.object_manager.all_obj_ids)

    def _segment(self,
                 key: torch.Tensor,
                 selection: torch.Tensor,
                 ms_features: Iterable[torch.Tensor],
                 update_sensory: bool = True) -> torch.Tensor:
        if not self.memory.engaged:
            warnings.warn('Trying to segment without any memory!', RuntimeWarning)
            return torch.zeros((1, key.shape[-2] * 16, key.shape[-1] * 16),
                               device=key.device,
                               dtype=key.dtype)
        memory_readout = self.memory.match_memory(key, selection)
        memory_readout = self.object_manager.realize_dict(memory_readout)
        memory_readout = memory_readout.unsqueeze(0)
        sensory, _, pred_prob_with_bg = self.network.segment(ms_features,
                                                             memory_readout,
                                                             self.memory.get_sensory(
                                                                 self.object_manager.all_obj_ids),
                                                             self.last_mask,
                                                             chunk_size=self.chunk_size,
                                                             update_sensory=update_sensory)
        # remove batch dim
        pred_prob_with_bg = pred_prob_with_bg[0]
        if update_sensory:
            self.memory.update_sensory(sensory, self.object_manager.all_obj_ids)
        return pred_prob_with_bg

    def add_to_temporary_buffer(self, frame_info: FrameInfo) -> None:
        self.frame_buffer.append(frame_info)

    def vote_in_temporary_buffer(
            self,
            keyframe_selection: Literal['last', 'middle', 'score', 'first'] = 'first'
    ) -> (int, torch.Tensor, List[ObjectInfo]):
        projected_ti, projected_mask, projected_info = find_consensus_auto_association(
            self.frame_buffer,
            network=self.network,
            store=self.image_feature_store,
            config=self.config,
            keyframe_selection=keyframe_selection)

        return projected_ti, projected_mask, projected_info

    def clear_buffer(self) -> None:
        # clear buffer
        for f in self.frame_buffer:
            self.image_feature_store.delete(f.ti)
        self.frame_buffer = []

    def incorporate_detection(self,
                              image: torch.Tensor,
                              new_mask: torch.Tensor,
                              segments_info: List[ObjectInfo],
                              *,
                              image_ti_override: bool = None,
                              forward_mask: torch.Tensor = None) -> torch.Tensor:
        # this is used for merging detections from an image-based model
        # it is not used for VOS inference
        self.curr_ti += 1

        if image_ti_override is not None:
            image_ti = image_ti_override
        else:
            image_ti = self.curr_ti

        image, self.pad = pad_divide_by(image, 16)
        new_mask, _ = pad_divide_by(new_mask, 16)
        image = image.unsqueeze(0)  # add the batch dimension

        ms_features = self.image_feature_store.get_ms_features(image_ti, image)
        key, shrinkage, selection = self.image_feature_store.get_key(image_ti, image)

        if forward_mask is None:
            if self.memory.engaged:
                # forward prediction
                prob = self._segment(key, selection, ms_features)
                forward_mask = torch.argmax(prob, dim=0)
            else:
                # initialization
                forward_mask = torch.zeros_like(new_mask)

        # merge masks (Section 3.2.2)
        merged_mask, class_id_match_dict = match_and_merge(forward_mask,
                                                           new_mask,
                                                           self.object_manager,
                                                           segments_info,
                                                           max_num_objects=self.max_num_objects,
                                                           incremental_mode=(forward_mask is not None))

        # find inactive objects that we need to delete
        purge_activated, tmp_keep_idx, obj_keep_idx = self.object_manager.purge_inactive_objects(
            self.max_missed_detection_count)

        if purge_activated:
            # purge memory
            self.memory.purge_except(obj_keep_idx)
            # purge the merged mask, no background
            new_list = [i - 1 for i in tmp_keep_idx]
            merged_mask = merged_mask[new_list]

        # add mask to memory
        self.last_mask_bool = merged_mask

        self.last_mask = merged_mask.unsqueeze(0).type_as(key)
        self._add_memory(image, ms_features, self.last_mask, key, shrinkage, selection)
        pred_prob_with_bg = self.network.aggregate(self.last_mask[0], dim=0)

        self.image_feature_store.delete(image_ti)

        return unpad(pred_prob_with_bg, self.pad), class_id_match_dict

    def step(self,
             image: torch.Tensor,
             mask: torch.Tensor = None,
             objects: Optional[List[int]] = None,
             *,
             hard_mask: bool = True,
             end: bool = False,
             image_ti_override: bool = None,
             delete_buffer: bool = True) -> torch.Tensor:
        """image: 3*H*W
        mask: H*W or len(objects)*H*W (if hard) or None
        objects: list of object id, in corresponding order as the mask
                    Ignored if the mask is None.
                    If None, hard_mask must be False.
                        Since we consider each channel in the soft mask to be an object.
        end: if we are at the end of the sequence, we do not need to update memory
            if unsure just set it to False always
        """
        if objects is None and mask is not None:
            assert not hard_mask
            objects = list(range(1, mask.shape[0] + 1))

        self.curr_ti += 1

        if image_ti_override is not None:
            image_ti = image_ti_override
        else:
            image_ti = self.curr_ti

        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension

        is_mem_frame = ((self.curr_ti - self.last_mem_ti >= self.mem_every) or
                        (mask is not None)) and (not end)
        # segment when there is no input mask or when the input mask is incomplete
        need_segment = (mask is None) or (not self.object_manager.has_all(objects)
                                          and self.object_manager.num_obj > 0)

        ms_features = self.image_feature_store.get_ms_features(image_ti, image)
        key, shrinkage, selection = self.image_feature_store.get_key(image_ti, image)

        if need_segment:
            pred_prob_with_bg = self._segment(key, selection, ms_features, update_sensory=not end)

        # use the input mask if provided
        if mask is not None:
            # inform the manager of the new objects, get list of temporary id
            corresponding_tmp_ids, _ = self.object_manager.add_new_objects(objects)

            mask, _ = pad_divide_by(mask, 16)
            if need_segment:
                # merge predicted mask with the incomplete input mask
                pred_prob_no_bg = pred_prob_with_bg[1:]
                # use the mutual exclusivity of segmentation
                if hard_mask:
                    pred_prob_no_bg[:, mask > 0] = 0
                else:
                    pred_prob_no_bg[:, mask.max(0) > 0.5] = 0

                new_masks = []
                for mask_id, tmp_id in enumerate(corresponding_tmp_ids):
                    if hard_mask:
                        this_mask = (mask == objects[mask_id]).type_as(pred_prob_no_bg)
                    else:
                        this_mask = mask[tmp_id]
                    if tmp_id >= pred_prob_no_bg.shape[0]:
                        new_masks.append(this_mask.unsqueeze(0))
                    else:
                        # +1 because no background
                        pred_prob_no_bg[tmp_id + 1] = this_mask
                # new_masks are always in the order of tmp_id
                mask = torch.cat([pred_prob_no_bg, *new_masks], dim=0)
            elif hard_mask:
                # simply convert cls to one-hot representation
                mask = torch.stack(
                    [mask == objects[mask_id] for mask_id, _ in enumerate(corresponding_tmp_ids)],
                    dim=0)
            pred_prob_with_bg = self.network.aggregate(mask, dim=0)
            pred_prob_with_bg = torch.softmax(pred_prob_with_bg, dim=0)

        # #merge same class masks that are in contact:
        # for cls,objects in self.object_manager.class_to_obj_id.items():
        #     if len(objects) > 1:
        #
        #         mask = torch.zeros_like(pred_prob_with_bg[0])
        #         tmp_ids = [self.object_manager.obj_to_tmp_id[obj] for obj in objects]
        #
        #         for obj in objects:
        #             tmp_id = self.object_manager.obj_to_tmp_id[obj]
        #             mask += pred_prob_with_bg[tmp_id]
        #
        #         mask = mask > 0.5
        #         mask_dilated = cv2.dilate(mask.cpu().numpy().astype(np.uint8), np.ones((5,5), np.uint8), iterations=2)
        #         n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dilated, connectivity=8)
        #         if n_components > 2:
        #             continue
        #         else:
        #             #store in the first object
        #             pred_prob_with_bg[tmp_ids[0]] = mask

        self.last_mask = pred_prob_with_bg[1:].unsqueeze(0)

        # save as memory if needed
        if is_mem_frame:
            self._add_memory(image, ms_features, self.last_mask, key, shrinkage, selection)

        if delete_buffer:
            self.image_feature_store.delete(image_ti)

        return unpad(pred_prob_with_bg, self.pad)


def get_deva_format_segmentations(boxes, masks, confidences, classes, prev_class_mask_dict=None):
    segments_info = []
    if len(masks) == 0:
        mask = None
        return mask, segments_info
    output_mask = torch.zeros_like(masks[0], dtype=torch.int64)

    curr_id = 1  # with bg

    mask_areas = [(mask > 0.5).sum() for mask in masks]

    # sort by mask area
    sorted_idx = np.argsort(mask_areas)[::-1]

    # for class_idx, (box, mask, confidence, class_id) in enumerate(zip(boxes, masks, confidences, classes)):
    for idx in sorted_idx:
        class_idx = idx
        box = boxes[idx]
        mask = masks[idx]
        confidence = confidences[idx]
        class_id = classes[idx]
        curr_id = int(class_idx + 1)
        if not confidence > 0:
            continue
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        if box_area < 1:
            continue
        if class_id == "robot" and prev_class_mask_dict is not None and "robot" in prev_class_mask_dict:
            robot_masks = np.array(prev_class_mask_dict["robot"]["mask"])

            # resize to current mask size
            robot_masks = cv2.resize(robot_masks, (mask.shape[1], mask.shape[0]))
            iou = (robot_masks * np.array(mask)).sum() / (robot_masks + np.array(mask)).sum()
            if iou < 0.15 and mask.sum() > mask.shape[1] * mask.shape[0] * 0.1:
                continue
        output_mask[mask > 0.5] = curr_id
        segments_info.append(ObjectInfo(id=curr_id, category_id=class_idx, score=confidence, class_scores=[confidence]))
        curr_id += 1

    return output_mask, segments_info


def get_mask_from_prob(prob, need_resize, shape):
    if need_resize:
        prob = torch.nn.functional.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,
               0]
    # Probability mask -> index mask
    mask = torch.argmax(prob, dim=0)

    return mask


@torch.inference_mode()
def process_frame_with_text(deva: DEVAInferenceCoreCustom,

                            frame_path: str,
                            result_saver: ResultSaver,
                            ti: int,
                            total_steps,
                            masks=None,
                            confidences=None,
                            boxes=None,
                            classes=None,
                            image_np: np.ndarray = None,
                            last_obj_mask_dict=None,
                            flow=None,
                            dense_logits=None
                            ) -> None:
    # image_np, if given, should be in RGB
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = deva.config

    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)

    masks = torch.tensor(masks).to(torch.uint8)
    if need_resize:
        h, w = image_np.shape[:2]
        scale = new_min_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        if len(masks > 0):
            masks_resized = \
                torch.nn.functional.interpolate(masks.unsqueeze(0), (new_h, new_w), mode='bilinear',
                                                align_corners=False)[0]
            masks = masks_resized

    frame_name = path.basename(frame_path)

    # get current masks and classes

    class_mask_dicts = []

    if cfg['temporal_setting'] == 'semionline':

        # if there is a new class which is not tracked, incorporate detection
        # assigned_classes = list(deva.object_manager.class_to_obj_id.keys())
        # valid_segementation_classes = [obj.category_ids[0] for obj in segments_info if obj.scores[0] > 0]
        #
        # unassigned_classes = [c for c in valid_segementation_classes if c not in assigned_classes]

        # if len(unassigned_classes) > 0 and ti >= cfg['num_voting_frames']:
        #     # incorporate new detections
        #     #s = 1
        #     #unassigned_classes_seg_info = [obj for obj in segments_info if obj.category_ids[0] in unassigned_classes]
        #     #prob = deva.incorporate_detection(image, mask.cuda(), unassigned_classes_seg_info)
        #     deva.next_voting_frame = ti + cfg['num_voting_frames'] - 1

        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:

            mask, segments_info = get_deva_format_segmentations(boxes, masks, confidences, classes, last_obj_mask_dict)

            frame_info = FrameInfo(image, mask.cuda(), segments_info, ti, {
                'frame': [frame_name],
                'shape': [h, w],
            })

            # mask, segments_info = make_segmentation_with_text(cfg, image_np, gd_model, sam_model,
            #                                                   prompts, new_min_side)

            # frame_info.mask = mask
            # frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            # wait for more frames before proceeding
            deva.add_to_temporary_buffer(frame_info)

            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np

                if len(deva.frame_buffer) == 1:
                    mask, new_segments_info = mask, segments_info
                    mask = mask.cuda()
                else:
                    _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                        keyframe_selection='first')

                # if class is missing, add it to mask

                for info in new_segments_info:
                    info.class_scores = info.scores.copy()

                prob, class_id_match_dict = deva.incorporate_detection(this_image, mask, new_segments_info)

                out_mask = torch.argmax(prob, dim=0)
                out_mask = deva.object_manager.tmp_to_obj_cls(out_mask)

                deva.next_voting_frame += cfg['detection_every']

                if deva.next_voting_frame > total_steps:
                    deva.next_voting_frame = deva.next_voting_frame + cfg['num_voting_frames']

                deva.object_manager.unfiltered_obj_to_tmp_id = copy.deepcopy(deva.object_manager.obj_id_to_obj)

                clean_outlier_labels(deva.object_manager, deva.last_mask_bool)

                class_mask_dict = get_return_masks_dict(prob, this_image_np, classes,
                                                        deva.object_manager.obj_to_tmp_id,
                                                        deva.object_manager.class_to_obj_id, need_resize, (h, w),
                                                        )

                class_id_match_dict_lang = {classes[k]: v for k, v in class_id_match_dict.items()}

                deva.penalty_list.append(class_id_match_dict_lang)

                # result_saver.save_mask(prob,
                #                        this_frame_name,
                #                        need_resize=need_resize,
                #                        shape=(h, w),
                #                        image_np=this_image_np,
                #                        prompts=classes)

                mask = get_mask_from_prob(prob, need_resize, (h, w))
                cur_mapping = copy.deepcopy(deva.object_manager.obj_to_tmp_id)

                deva.masks.append(mask.cpu())
                deva.obj_to_tmp_ids.append(cur_mapping)
                # class_mask_dict = get_return_masks_dict(prob, image_np, classes,
                #                                         deva.object_manager.obj_to_tmp_id,
                #                                         deva.object_manager.class_to_obj_id, need_resize, (h, w))
                #
                class_mask_dicts.append(class_mask_dict)

                deva.id_to_tmp_id_hist.append(copy.deepcopy(deva.object_manager.obj_to_tmp_id))
                deva.class_to_obj_id_hist.append(copy.deepcopy(deva.object_manager.class_to_obj_id))

                mask_bool = torch.argmax(prob, dim=0)
                # split masks and remove background
                split_masks = []
                for i in range(0, len(deva.object_manager.tmp_id_to_obj) + 1):
                    split_masks.append(mask_bool == i)
                split_masks = torch.stack(split_masks)[1:].to(bool)
                deva.mask_bool_hist.append(split_masks)

                for frame_info in deva.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = deva.step(this_image, None, None)
                    deva.object_manager.unfiltered_obj_to_tmp_id = copy.deepcopy(deva.object_manager.obj_id_to_obj)

                    clean_outlier_labels(deva.object_manager, deva.last_mask_bool)
                    class_mask_dict = get_return_masks_dict(prob, image_np, classes,
                                                            deva.object_manager.obj_to_tmp_id,
                                                            deva.object_manager.class_to_obj_id, need_resize, (h, w))
                    # result_saver.save_mask(prob,
                    #                        this_frame_name,
                    #                        need_resize,
                    #                        shape=(h, w),
                    #                        image_np=this_image_np,
                    #                        prompts=classes)
                    mask = get_mask_from_prob(prob, need_resize, (h, w))
                    cur_mapping = copy.deepcopy(deva.object_manager.obj_to_tmp_id)

                    deva.masks.append(mask.cpu())
                    deva.obj_to_tmp_ids.append(cur_mapping)
                    # class_mask_dict = get_return_masks_dict(prob, image_np, classes,
                    #                                         deva.object_manager.obj_to_tmp_id,
                    #                                         deva.object_manager.class_to_obj_id, need_resize, (h, w))
                    #
                    class_mask_dicts.append(class_mask_dict)

                    deva.id_to_tmp_id_hist.append(copy.deepcopy(deva.object_manager.obj_to_tmp_id))
                    deva.class_to_obj_id_hist.append(copy.deepcopy(deva.object_manager.class_to_obj_id))

                    mask_bool = torch.argmax(prob, dim=0)
                    # split masks and remove background
                    split_masks = []
                    for i in range(0, len(deva.object_manager.tmp_id_to_obj) + 1):
                        split_masks.append(mask_bool == i)
                    split_masks = torch.stack(split_masks)[1:].to(bool)
                    deva.mask_bool_hist.append(split_masks)

                deva.clear_buffer()
        else:
            # standard propagation
            # if flow is not None:
            #     #propagate mask with flow
            #     prob_masks = []
            #     for obj_mask in deva.last_mask_bool:
            #         obj_mask_orig_shape = cv2.resize(obj_mask.cpu().numpy().astype(np.uint8), (flow.shape[1], flow.shape[0])).astype(bool)
            #         prop_mask = propagate_seg_mask_flow(obj_mask_orig_shape, flow)
            #         prob_masks.append(torch.from_numpy(prop_mask).to(bool))
            #     prob_masks = torch.stack(prob_masks)
            #     
            #     
            # 
            #     
            prob = deva.step(image, None, None)
            # prob_masks = torch.nn.functional.interpolate(prob_masks.unsqueeze(0).float(), prob.shape[-2:], mode='bilinear', align_corners=False)[0].to(bool)
            # 
            # prob_masks_with_bg = torch.zeros_like(prob)
            # prob_masks_with_bg[1:] = prob_masks

            # prob = prob * prob_masks

            deva.object_manager.unfiltered_obj_to_tmp_id = copy.deepcopy(deva.object_manager.obj_id_to_obj)

            clean_outlier_labels(deva.object_manager, deva.last_mask_bool)

            class_mask_dict = get_return_masks_dict(prob, image_np, classes,
                                                    deva.object_manager.obj_to_tmp_id,
                                                    deva.object_manager.class_to_obj_id, need_resize, (h, w))

            # result_saver.save_mask(prob,
            #                        frame_name,
            #                        need_resize=need_resize,
            #                        shape=(h, w),
            #                        image_np=image_np,
            #                        prompts=classes)
            mask = get_mask_from_prob(prob, need_resize, (h, w))
            cur_mapping = copy.deepcopy(deva.object_manager.obj_to_tmp_id)

            deva.masks.append(mask.cpu())
            deva.obj_to_tmp_ids.append(cur_mapping)
            # class_mask_dict = get_return_masks_dict(prob, image_np, classes,

            #                                         deva.object_manager.obj_to_tmp_id,
            #                                         deva.object_manager.class_to_obj_id, need_resize, (h, w))
            #
            class_mask_dicts.append(class_mask_dict)

            deva.id_to_tmp_id_hist.append(copy.deepcopy(deva.object_manager.obj_to_tmp_id))
            deva.class_to_obj_id_hist.append(copy.deepcopy(deva.object_manager.class_to_obj_id))
            mask_bool = torch.argmax(prob, dim=0)
            # split masks and remove background
            split_masks = []
            for i in range(0, len(deva.object_manager.tmp_id_to_obj) + 1):
                split_masks.append(mask_bool == i)
            split_masks = torch.stack(split_masks)[1:].to(bool)
            deva.mask_bool_hist.append(split_masks)


    elif cfg['temporal_setting'] == 'online':
        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            # mask, segments_info = make_segmentation_with_text(cfg, image_np, gd_model, sam_model,
            #                                                   prompts, new_min_side)
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, mask, segments_info)
        else:
            # Run the model on this frame
            prob = deva.step(image, None, None)
        # result_saver.save_mask(prob,
        #                        frame_name,
        #                        need_resize=need_resize,
        #                        shape=(h, w),
        #                        image_np=image_np,
        #                        prompts=classes)

    return class_mask_dicts


from typing import List, Literal, Dict
from collections import defaultdict
import torch

from deva.model.memory_utils import *
from deva.model.network import DEVA
from deva.inference.image_feature_store import ImageFeatureStore
from deva.inference.consensus_associated import spatial_alignment
from deva.utils.tensor_utils import pad_divide_by, unpad

import numpy as np

import pulp

try:
    from gurobipy import GRB
    import gurobipy as gp

    use_gurobi = True
except ImportError:
    use_gurobi = False


def solve_with_gurobi(pairwise_iou: np.ndarray, pairwise_iou_indicator: np.ndarray,
                      total_segments: int) -> List[bool]:
    # All experiments in the paper are conducted with gurobi.
    m = gp.Model("solver")
    m.Params.LogToConsole = 0

    # indicator variable
    x = m.addMVar(shape=(total_segments, 1), vtype=GRB.BINARY, name="x")

    # maximize this
    m.setObjective(
        # high support, *2 to compensate because we only computed the upper triangle
        (pairwise_iou @ x).sum() * 2
        # few segments -- the paper says *0.5 but it's later found
        # that the experiments were done with alpha=1 -- should not have a major impact
        - x.sum(),
        GRB.MAXIMIZE)

    # no two selected segments should have >0.5 iou
    m.addConstr((pairwise_iou_indicator * (x @ x.transpose())).sum() == 0, "iou")

    m.optimize()

    results = (x.X > 0.5)[:, 0].tolist()
    return results


def solve_with_pulp(pairwise_iou: np.ndarray, pairwise_iou_indicator: np.ndarray,
                    total_segments: int) -> List[bool]:
    # pulp is a fallback solver; no guarantee that it works the same
    m = pulp.LpProblem('prob', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', range(total_segments), cat=pulp.LpBinary)

    support_objective = pulp.LpAffineExpression([(x[i], pairwise_iou[:, i].sum() * 2)
                                                 for i in range(total_segments)])
    penal_objective = pulp.LpAffineExpression([(x[i], -1) for i in range(total_segments)])
    m += support_objective + penal_objective

    for i in range(total_segments):
        for j in range(i + 1, total_segments):
            if pairwise_iou_indicator[i, j] == 1:
                constraint = pulp.LpConstraint(pulp.LpAffineExpression([(x[i], 1), (x[j], 1)]),
                                               pulp.LpConstraintLE, f'{i}-{j}', 1)
                m += constraint

    # you can change the solver if you have others installed
    m.solve(pulp.PULP_CBC_CMD(msg=0))

    results = [None for _ in range(total_segments)]
    for v in m.variables():
        results[int(v.name[2:])] = v.varValue
    return results


def find_consensus_auto_association(frames: List[FrameInfo],
                                    keyframe_selection: Literal['last', 'middle', 'score',
                                    'first'] = 'last',
                                    *,
                                    network: DEVA,
                                    store: ImageFeatureStore,
                                    config: Dict) -> (int, torch.Tensor, List[ObjectInfo]):
    global use_gurobi

    time_indices = [f.ti for f in frames]
    images = []
    masks = []
    for f in frames:
        image, pads = pad_divide_by(f.image, 16)
        # masks here have dtype Long and is index-based, i.e., not one-hot
        mask, _ = pad_divide_by(f.mask, 16)
        images.append(image)
        masks.append(mask)

    segments_info = [f.segments_info for f in frames]
    channel_to_id_mappings = []
    internal_id_bookkeeper = 0
    all_new_segments_info = {}
    frame_index_to_seg_info = defaultdict(list)

    # convert all object indices such that indices from different frames do not overlap
    # also convert the masks into one-hot format for propagation
    for i, this_seg_info in enumerate(segments_info):
        new_one_hot_mask = []
        this_channel_mapping = {}
        for si, seg_info in enumerate(this_seg_info):
            old_id = seg_info.id
            internal_id_bookkeeper += 1
            new_id = internal_id_bookkeeper

            # create new object info
            new_seg_info = ObjectInfo(new_id)
            new_seg_info.copy_meta_info(seg_info)
            all_new_segments_info[new_id] = new_seg_info

            # make that into the mask
            new_one_hot_mask.append(masks[i] == old_id)
            this_channel_mapping[si] = new_id
            frame_index_to_seg_info[i].append(new_seg_info)

        if len(new_one_hot_mask) == 0:
            masks[i] = None  # no detected mask
        else:
            masks[i] = torch.stack(new_one_hot_mask, dim=0).float()
        channel_to_id_mappings.append(this_channel_mapping)

    # find a keyframe
    if keyframe_selection == 'last':
        keyframe_i = len(time_indices) - 1
    elif keyframe_selection == 'first':
        keyframe_i = 0
    elif keyframe_selection == 'middle':
        keyframe_i = (len(time_indices) + 1) // 2
    elif keyframe_selection == 'score':
        keyframe_i = None
        raise NotImplementedError
    else:
        raise NotImplementedError

    keyframe_ti = time_indices[keyframe_i]
    keyframe_image = images[keyframe_i]
    keyframe_mask = masks[keyframe_i]

    # project all frames onto the keyframe
    projected_masks = []
    segment_id_to_areas = {}
    segment_id_to_mask = {}
    for ti, image, mask, mapping in zip(time_indices, images, masks, channel_to_id_mappings):
        if mask is None:
            # no detection -> no projection
            projected_masks.append(None)
            continue

        if ti == keyframe_ti:
            # no need to project the keyframe
            projected_mask = torch.cat([torch.ones_like(keyframe_mask[0:1]) * 0.5, keyframe_mask],
                                       dim=0)
        else:
            projected_mask = spatial_alignment(ti, image, mask, keyframe_ti, keyframe_image,
                                               network, store, config)[0]
        projected_mask = unpad(projected_mask, pads)
        # maps the projected mask back into the class index format
        projected_mask = torch.argmax(projected_mask, dim=0)
        remapped_mask = torch.zeros_like(projected_mask)
        for channel_id, object_id in mapping.items():
            # +1 because of background
            this_mask = projected_mask == (channel_id + 1)
            remapped_mask[this_mask] = object_id
            segment_id_to_areas[object_id] = this_mask.sum().item()
            segment_id_to_mask[object_id] = this_mask

        projected_masks.append(remapped_mask.long())

    # compute pairwise iou
    image_area = keyframe_image.shape[-1] * keyframe_image.shape[-2]
    total_segments = internal_id_bookkeeper
    SCALING = 4096
    assert total_segments < SCALING
    # we are filling the upper triangle; diagonal-blocks remain zero
    matching_table = defaultdict(list)
    pairwise_iou = np.zeros((total_segments, total_segments), dtype=np.float32)
    # pairwise_intersection = np.zeros((total_segments, total_segments), dtype=np.float32)
    segments_area = np.zeros((total_segments, 1), dtype=np.float32)
    segments_area[:, 0] = np.array(list(segment_id_to_areas.values()))

    # empty masks in all frames
    if total_segments == 0:
        output_mask = torch.zeros_like(frames[0].mask)
        output_info = []
        return keyframe_ti, output_mask, output_info

    for i in range(len(time_indices)):
        if projected_masks[i] is None:
            continue
        mask1_scaled = projected_masks[i] * SCALING
        for j in range(i + 1, len(time_indices)):
            if projected_masks[j] is None:
                continue
            mask2 = projected_masks[j]
            # vectorized intersection check
            combined = mask1_scaled + mask2

            match_isthing = [None, False, True]  # for isthing
            for isthing_status in match_isthing:
                matched_mask2_id = set()
                for obj1 in frame_index_to_seg_info[i]:
                    mask1_id = obj1.id
                    if obj1.isthing != isthing_status:
                        continue
                    for obj2 in frame_index_to_seg_info[j]:
                        mask2_id = obj2.id
                        # skip if already matched, since we only care IoU>0.5 which is unique
                        if (obj2.isthing != isthing_status) or (mask2_id in matched_mask2_id):
                            continue

                        target_label = mask1_id * SCALING + mask2_id
                        intersection = (combined == target_label).sum().item()
                        if intersection == 0:
                            continue
                        union = segment_id_to_areas[mask1_id] + \
                                segment_id_to_areas[mask2_id] - intersection
                        iou = intersection / union
                        if iou > 0.5:
                            matching_table[mask1_id].append(mask2_id)
                            matching_table[mask2_id].append(mask1_id)
                            matched_mask2_id.add(mask2_id)
                            pairwise_iou[mask1_id - 1, mask2_id - 1] = iou
                            break

    # make symmetric
    pairwise_iou = pairwise_iou + pairwise_iou.T
    # same as >0.5 as we excluded IoU<=0.5
    # 0.49 is used for numerical reasons (probably doesn't actually matter)
    pairwise_iou_indicator = (pairwise_iou > 0.49)
    # suppress low confidence estimation
    pairwise_iou = pairwise_iou * pairwise_iou_indicator
    segments_area /= image_area  # normalization

    if use_gurobi:
        try:
            results = solve_with_gurobi(pairwise_iou, pairwise_iou_indicator, total_segments)
        except gp.GurobiError:
            print('GurobiError, falling back to pulp')
            use_gurobi = False
    if not use_gurobi:
        results = solve_with_pulp(pairwise_iou, pairwise_iou_indicator, total_segments)

    output_mask = torch.zeros_like(frames[0].mask)
    output_info = []
    matched_object_id_to_area = {}
    for channel_id, selected in enumerate(results):
        if selected:
            object_id = channel_id + 1
            matched_object_id_to_area[object_id] = segment_id_to_areas[object_id]

            # merge object info
            new_object_info = all_new_segments_info[object_id]
            n_matching = len(matching_table[object_id])
            matching_score = n_matching / (len(frames) - 1)
            matching_score = max(0, (0.5 - matching_score) * 0.2)
            # print(matching_score)
            for other_object_id in matching_table[object_id]:
                new_object_info.merge(all_new_segments_info[other_object_id])

            # new_object_info.missed_concensous.append(matching_score)
            new_object_info.class_scores = [s - matching_score for s in new_object_info.class_scores]

            output_info.append(new_object_info)

    sorted_by_area = sorted(matched_object_id_to_area.items(), key=lambda x: x[1], reverse=True)
    for object_id, _ in sorted_by_area:
        output_mask[segment_id_to_mask[object_id]] = object_id

    return keyframe_ti, output_mask, output_info


def postprocess_deva_masks(result_dict):
    robot_masks = result_dict["robot"]["mask"]
    for obj, stats in result_dict.items():
        masks = stats["mask"]

        for idx, m in enumerate(masks):
            if m.sum() == 0:
                continue
            ref_masks_idx = np.arange(len(masks))
            ref_masks_idx = ref_masks_idx[ref_masks_idx != idx]

            masks_dilated = cv2.dilate(m.astype(np.uint8), np.ones((12, 12), np.uint8),
                                       iterations=2)
            # masks_dilated = m

            robot_mask_dilated = cv2.dilate(robot_masks[idx].astype(np.uint8), np.ones((1, 1), np.uint8),
                                            iterations=1)
            # Check if robot occludes
            robot_obj_mask_combined = np.logical_or(masks_dilated, robot_mask_dilated)

            n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
                masks_dilated.astype(np.uint8))

            if n_components > 2:
                n_components_robot, labels_robot, _, _ = cv2.connectedComponentsWithStats(
                    robot_obj_mask_combined.astype(np.uint8))
                if n_components_robot <= 2:
                    continue

                component_scores = []

                for i in range(1, n_components):
                    # get iou with other masks:
                    cur_mask = labels == i
                    # iou = compute_iou(cur_mask,masks[ref_masks_idx]).squeeze()
                    iou = compute_iou(cur_mask, masks[idx]).squeeze()
                    score = iou.mean()
                    component_scores.append(score)
                component_scores = np.array(component_scores)
                best_component = np.argmax(component_scores) + 1

                best_mask = labels == best_component

                # get best orig mask by iou:

                orig_mask_ious = []
                n_comps_undilated, masks_undilated, _, _ = cv2.connectedComponentsWithStats(
                    m.astype(np.uint8))
                for i in range(1, n_comps_undilated):
                    cur_mask = masks_undilated == i
                    iou = compute_iou(cur_mask, best_mask).squeeze()
                    orig_mask_ious.append(iou)
                orig_mask_ious = np.array(orig_mask_ious)
                best_orig_mask_idx = np.argmax(orig_mask_ious) + 1

                result_dict[obj]["mask"][idx] = masks_undilated == best_orig_mask_idx

                # result_dict[obj]["mask"][idx] = labels == best_component
                result_dict[obj]["box"][idx] = masks_to_boxes(torch.tensor(labels == best_component)[None]).squeeze(
                    0)
