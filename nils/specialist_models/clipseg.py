import gc
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.ops import nms
from tqdm import tqdm
from transformers import CLIPSegConfig, CLIPSegForImageSegmentation, CLIPSegProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clipseg.modeling_clipseg import (
    CLIPSegDecoder,
    CLIPSegImageSegmentationOutput,
    CLIPSegModel,
    CLIPSegOutput,
    CLIPSegPreTrainedModel,
)

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import Compose, Normalize, ToTensor

from nils.specialist_models.detectors.utils import create_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_preprocess = Compose([
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


class CLipSeg:

    def __init__(self):

        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined",
                                                                 torch_dtype=torch.float16)

        # self.image_level_model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = self.model.to(device)

        self.text_embeddings = None
        self.prompts = None
        self.prompt_category_map = None
        self.id_state_map = None
        self.state_text_embeddings = {}

        self.obj_state_map = None

    def get_text_embeddings(self, texts):
        prompts_split = [prompt.split(",") for prompt in texts]

        prompts_split = [[prompt.strip() for prompt in prompts] for prompts in prompts_split]

        tokenized_categories = [self.processor(text=prompt, padding="max_length",
                                               return_tensors="pt") for prompt in prompts_split]

        return self.model.get_text_embeddings(tokenized_categories)

    def precompute_text_embeddings(self, objects, states=None, class_id_mapping=None, id_state_mapping={}):

        if class_id_mapping is None:
            class_id_mapping = list(range(0, len(objects)))
        self.prompts = objects
        self.prompt_category_map = class_id_mapping
        self.id_state_map = id_state_mapping

        self.text_embeddings = self.get_text_embeddings(objects)

        # all_states = np.concatenate([states[obj] for obj in states])

        # obj_state_mapping = {}
        if states is not None:
            # count = 0
            # for obj,states in states.items():
            #     
            #     obj_state_mapping[obj] = list(range(count,count+len(states)))
            #     count += len(states)
            #     

            self.state_text_embeddings = self.get_text_embeddings(states)

            # self.obj_state_map = obj_state_mapping

    def get_clip_sim(self, text_embeds, image_embeds):

        #        image_embeds = frame_model_outputs.vision_model_output.projected_pool_output
        #        text_embeds = frame_model_outputs.decoder_output.conditional_embeddings

        # image_embeds = vision_outputs[1]
        # image_embeds = self.visual_projection(image_embeds)

        # text_embeds = text_outputs[1]
        # text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.clip.logit_scale.exp().cpu()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        s = 1

        return logits_per_text

    def to_cpu(self):
        self.model.to("cpu")
        self.clip_model.to("cpu")
        torch.cuda.empty_cache()

    def get_affinities(self, images, text_embeddings=None, prompts=None, bsz=16, return_embeds=False):

        self.model.to("cuda")
        orig_images_shape = images.shape

        if self.text_embeddings is not None:
            prompts = self.prompts
            prompt_class_map = self.prompt_category_map

        if text_embeddings is None:
            text_embeddings = self.text_embeddings

        if images.ndim == 3:
            images = images[None, ...]

        images = F.interpolate(torch.tensor(np.array(images)).permute(0, 3, 1, 2), size=(224, 224),
                               mode="nearest-exact")

        images = images.permute(0, 2, 3, 1).numpy()

        if isinstance(images, np.ndarray):
            images = list(images)

        image_bsz = len(images)

        bsz = bsz
        n_prompts = len(prompts) if text_embeddings is None else text_embeddings.shape[0]
        img_batches = create_batches(bsz, np.array(images))

        output_lst = []
        image_embeds = []
        text_embeds = []
        for batch in tqdm(img_batches, disable=True):

            batch = list(batch)

            prompts_repeated = []

            for prompt in prompts:
                prompts_repeated.extend([prompt] * bsz)

            if self.text_embeddings is None:
                inputs = self.processor(text=prompts_repeated, images=batch * len(prompts), padding="max_length",
                                        return_tensors="pt")
            else:
                inputs = self.processor(images=batch, padding="max_length",
                                        return_tensors="pt")

            inputs = inputs.to(device)

            # predict
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.model(**inputs, conditional_embeddings=text_embeddings)

                    image_embeds.append(outputs.vision_model_output.projected_pool_output.cpu().to(torch.float32))
                    text_embeds.append(outputs.conditional_embeddings_unrepeated.cpu().to(torch.float32))

                    logits = outputs["logits"].cpu().to(torch.float32)
                    logits_reshaped = logits.reshape(n_prompts, len(batch), logits.shape[1], logits.shape[2]).permute(1,
                                                                                                                      0,
                                                                                                                      2,
                                                                                                                      3)

                    output_lst.append(logits_reshaped)

                    inputs = inputs.to("cpu")
                    del outputs
                    del inputs

        affinities = torch.concatenate(output_lst, axis=0)
        affinities = F.interpolate(affinities, size=(orig_images_shape[1], orig_images_shape[2]))

        self.model.to("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        if return_embeds:
            image_embeds = torch.cat(image_embeds, dim=0)
            text_embeds = torch.cat(text_embeds, dim=0)
            return affinities, image_embeds, text_embeds
        else:
            return affinities

    def get_best_component(self, mask, logits):
        logits = np.array(logits)
        mask = np.array(mask)

        components = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)

        labels = components[1]

        # get scores of each component
        # scores = np.array([scores[i] for i in cur_scores])
        component_scores = {
            component_id: np.sum(logits[labels == component_id]) / (np.sum(labels == component_id) * 0.5) for
            component_id
            in np.unique(labels)}

        best_component_id = max(component_scores, key=component_scores.get)
        best_score = component_scores[best_component_id]
        best_mask = (labels == best_component_id).astype(bool)

        return best_mask, best_score

    # Argmax does not work. Part of the object is recognized as another object sometimes.
    def filter_duplicate_masks(self, masks, scores):

        masks = np.array(masks)
        scores = np.array(scores)
        # Select highest score mask (based on component)
        # masks_dilated = np.stack(
        #     [cv2.dilate(mask.astype(np.uint8), np.ones((3, 3)), iterations=1) for mask in masks])
        # best_masks = []
        # best_scores = []
        # for i in range(len(masks_dilated)):
        #     
        #     cur_scores = scores[i] * masks[i]
        #     
        #     cur_scores  = scores[i]
        #     cur_mask = masks_dilated[i]
        #     
        #     best_mask, best_score = self.get_best_component(cur_mask,cur_scores)
        #     
        #     best_masks.append(best_mask)
        #     best_scores.append(best_score)

        boxes = masks_to_boxes(torch.tensor(masks))
        nms_indices = nms(boxes, torch.tensor(scores, dtype=boxes.dtype), 0.1)

        if len(nms_indices) != masks.shape[0]:
            s = 1
        return np.array(nms_indices)

    def temporal_voting(self, masks, scores, boxes):

        pass

    def predict_mask_simple(self, data_batch, prompts=None, subset="rgb_gripper", bsz=16, threshold=0.3):
        images = data_batch[subset]

        preds = self.get_affinities(images, bsz=bsz)

        preds_sigmoid = torch.sigmoid(preds)

        preds_normalized = (preds_sigmoid - preds_sigmoid.amin(axis=(-2, -1))[..., None, None]) / (
                    preds_sigmoid.amax(axis=(-2, -1))[..., None, None] - preds_sigmoid.amin(axis=(-2, -1))[
                ..., None, None])

        bmask = preds_normalized > threshold

        preds_normalized[~bmask] = 0

        return bmask

    def predict_static(self, data_batch, prompts=None, subset="rgb_gripper", bsz=32, bg_thresh_modifier=0.3,
                       conf_thresh=0.8):
        images = data_batch[subset]

        preds = self.get_affinities(images, bsz=bsz)

        if self.text_embeddings is not None:
            prompts = self.prompts
            prompt_class_map = self.prompt_category_map

        preds = F.interpolate(preds, size=(images[0].shape[0], images[0].shape[1]))

        # hardcoded for now
        if len(self.id_state_map) > 0:
            state_preds = preds[:, [9, 10], :]
            preds = preds[:, :10, :]
        else:
            state_preds = None
            preds = preds

        bg = torch.fill(torch.zeros_like(preds[:, 0, ...])[:, None], preds.mean() + 1.5 * preds.std())

        preds_sigmoid = torch.sigmoid(preds)
        per_class_bg = preds_sigmoid.mean(dim=(2, 3))

        preds_flattened = preds.reshape(preds.shape[0], preds.shape[1], -1)

        preds_sigmoid = torch.sigmoid(preds)
        bg_sig = torch.fill(torch.zeros_like(preds[:, 0, ...])[:, None], 0.25)
        filtered_preds_sigmoid = preds_sigmoid > 0.3
        preds_sigmoid_with_bg = torch.cat([bg_sig, preds_sigmoid], dim=1)
        max_sigmoid = torch.argmax(preds_sigmoid_with_bg, dim=1)
        # filter masks with sum 0 

        best_masks = []
        best_scores = []
        for i in range(len(filtered_preds_sigmoid)):

            cur_batch_best_scores = []
            cur_batch_best_masks = []
            for j in range(len(filtered_preds_sigmoid[i])):
                cur_mask = filtered_preds_sigmoid[i, j]
                cur_score = preds_sigmoid[i, j]
                best_mask, best_score = self.get_best_component(cur_mask, cur_score)
                cur_batch_best_scores.append(best_score)
                cur_batch_best_masks.append(best_mask)
            best_masks.append(cur_batch_best_masks)
            best_scores.append(cur_batch_best_scores)

        best_masks = np.array(best_masks)
        best_scores = np.array(best_scores)

    def predict(self, data_batch, prompts=None, subset="rgb_gripper", bsz=32, bg_thresh_modifier=0.3, conf_thresh=0.8):

        images = data_batch[subset]

        preds = self.get_affinities(images, bsz=bsz)

        preds_state, image_embeds, text_embeds = self.get_affinities(images, self.state_text_embeddings, bsz=bsz,
                                                                     return_embeds=True)

        # preds = torch.concatenate(output_lst, axis=0)

        if self.text_embeddings is not None:
            prompts = self.prompts
            prompt_class_map = self.prompt_category_map

        # text_embeds = text_embeds[0]
        # image_embeds = torch.cat(image_embeds, dim = 0)

        # probs = torch.softmax(preds.reshape(preds.shape[0], preds.shape[1] , -1), dim = -1).reshape(preds.shape)

        preds = F.interpolate(preds, size=(images[0].shape[0], images[0].shape[1]))
        preds_state = F.interpolate(preds_state, size=(images[0].shape[0], images[0].shape[1]))

        state_preds = preds_state

        # hardcoded for now
        # if len(self.id_state_map) > 0:
        #     state_preds = preds[:,[9,10],:]
        #     preds = preds[:,:10,:]
        # else:
        #     state_preds = None
        #     preds = preds

        # per_pixel_probs = torch.softmax(preds, dim = 1)
        bg = torch.fill(torch.zeros_like(preds[:, 0, ...])[:, None], preds.mean() + 1.5 * preds.std())
        # bg = torch.fill(torch.zeros_like(preds[:, 0, ...])[:, None], 0)

        preds_sigmoid = torch.sigmoid(preds)
        per_class_bg = preds_sigmoid.mean(dim=(2, 3))

        preds_flattened = preds.reshape(preds.shape[0], preds.shape[1], -1)
        preds_softmax = torch.softmax(preds_flattened, dim=-1).reshape(preds.shape)

        # bg_per_frame = preds.mean(dim = (1,2,3)) + 1 * preds.std(dim = (1,2,3))
        # bg_per_frame = bg_per_frame[:,None,None,None].repeat(1,1,preds.shape[2],preds.shape[3])

        preds_with_bg = torch.cat([bg, preds], dim=1)

        preds_sigmoid = torch.sigmoid(preds)
        bg_sig = torch.fill(torch.zeros_like(preds[:, 0, ...])[:, None], 0.25)
        filtered_preds_sigmoid = preds_sigmoid > 0.3
        preds_sigmoid_with_bg = torch.cat([bg_sig, preds_sigmoid], dim=1)
        max_sigmoid = torch.argmax(preds_sigmoid_with_bg, dim=1)
        # filter masks with sum 0

        best_masks = []
        best_scores = []
        for i in range(len(filtered_preds_sigmoid)):

            cur_batch_best_scores = []
            cur_batch_best_masks = []
            for j in range(len(filtered_preds_sigmoid[i])):
                cur_mask = filtered_preds_sigmoid[i, j]
                cur_score = preds_sigmoid[i, j]
                best_mask, best_score = self.get_best_component(cur_mask, cur_score)
                cur_batch_best_scores.append(best_score)
                cur_batch_best_masks.append(best_mask)
            best_masks.append(cur_batch_best_masks)
            best_scores.append(cur_batch_best_scores)

        best_masks = np.array(best_masks)
        best_scores = np.array(best_scores)

        scores = (filtered_preds_sigmoid * preds_sigmoid).sum(dim=(2, 3)) / (filtered_preds_sigmoid).sum(dim=(2, 3))
        scores = torch.nan_to_num(scores, nan=0)

        # nms_indices = [self.filter_duplicate_masks(filtered_preds_sigmoid[idx],preds_sigmoid[idx]) for idx in range(len(filtered_preds_sigmoid))]

        # nms_indices = [nms(boxes[i], scores[i], 0.8) for i in range(len(boxes))]

        per_pixel_thresh = torch.max(preds_with_bg, dim=1)[0] - bg_thresh_modifier

        max_class_per_pixel = torch.argmax(preds_with_bg, dim=1)

        object_masks = preds_with_bg > per_pixel_thresh[:, None]

        # preds = preds.reshape(image_bsz, n_prompts, preds.shape[1], preds.shape[2])

        detected_classes = []

        use_hard_max = False
        for frame_idx in range(len(images)):
            detected_dict = {}

            cur_max_confidence = preds[frame_idx].max()

            for class_idx in max_class_per_pixel[frame_idx].unique():

                if class_idx == 0:
                    continue
                if use_hard_max:
                    cur_mask = (max_class_per_pixel[frame_idx] == class_idx)
                else:
                    cur_mask = object_masks[frame_idx][class_idx]

                mask_area = cur_mask.sum()
                if mask_area > 200:
                    mask = cur_mask
                    confidence = preds[frame_idx][class_idx - 1][mask].mean()
                    if "robot" in self.prompts[class_idx - 1]:
                        thresh = 0.4
                    else:
                        thresh = conf_thresh
                    if confidence < min(thresh, cur_max_confidence * 3):
                        continue
                    cur_prompt_text = prompts[class_idx - 1]

                    # If object is stateful, detect state
                    state_idx = None
                    if cur_prompt_text in self.id_state_map:
                        state_indices = np.array(self.id_state_map[cur_prompt_text])
                        state_probs_mask = torch.softmax(state_preds[frame_idx][state_indices][:, mask].mean(dim=1),
                                                         dim=0)

                        state_probs_image = torch.softmax(
                            self.get_clip_sim(text_embeds[state_indices], image_embeds[frame_idx]).detach().cpu(),
                            dim=0)

                        mean_score = (state_probs_image * 0.3 + state_probs_mask * 0.7)
                        state_idx = torch.argmax(mean_score)

                    detected_dict[cur_prompt_text] = {"mask": mask.numpy(),
                                                      "area": mask_area.numpy(),
                                                      "confidence": preds[frame_idx][class_idx - 1][
                                                          mask].mean().numpy(),
                                                      "label": prompts[class_idx - 1], "label_idx": class_idx.item(),
                                                      "state": state_idx.item() if state_idx else None,
                                                      "frame_name": data_batch["frame_name"][frame_idx]}


            filtered_dict = detected_dict

            detected_classes.append(filtered_dict)

        return detected_classes


def get_text_embeddings(clip, tokenized_categories):
    attentienion

    with torch.no_grad():
        all_text_embeddings = []
        print("Building text embeddings...")

        for category in tqdm(tokenized_categories):
            texts = clip.tokenize(category)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = self.clip.encode_text(texts)  # embed with text encoder

            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

            text_embedding = text_embeddings.mean(dim=0)

            text_embedding /= text_embedding.norm()

            all_text_embeddings.append(text_embedding)

        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)

        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()
    return all_text_embeddings.cpu().numpy().T


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        if mask.sum() == 0:
            bounding_boxes[index, :] = 0
            continue

        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes


class CLIPSegCustomModel(CLIPSegModel):
    def __init__(self, config):
        super().__init__(config)

    def get_text_features(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            avg_mode="pre_projection"
    ) -> torch.FloatTensor:
        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        if avg_mode == "pre_projection":
            pooled_output = pooled_output.mean(dim=0)
        text_features = self.text_projection(pooled_output)

        return text_features


@dataclass
class ExtendedModelOutput(BaseModelOutputWithPooling):
    projected_pool_output: torch.FloatTensor = None


@dataclass
class CLIPSegImageSegmentationOutputExtended(CLIPSegImageSegmentationOutput):
    conditional_embeddings_unrepeated: torch.FloatTensor = None


class CLIPSegForImageSegmentation(CLIPSegPreTrainedModel):
    config_class = CLIPSegConfig

    def __init__(self, config: CLIPSegConfig):
        super().__init__(config)

        self.config = config

        self.clip = CLIPSegCustomModel(config)
        self.extract_layers = config.extract_layers

        self.decoder = CLIPSegDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_embeddings(self, tokenized_categories, avg_mode="pre_projection"):
        """Args:
            tokenized_categories:

        Returns:

        """
        with torch.no_grad():
            all_text_embeddings = []

            for category in tqdm(tokenized_categories, disable=True):

                input_ids = category["input_ids"]
                attention_mask = category["attention_mask"]
                if "position_ids" in category:

                    position_ids = category["position_ids"].to(self.clip.device)
                else:
                    position_ids = None

                conditional_embeddings = self.clip.get_text_features(
                    input_ids.to(self.clip.device), attention_mask=attention_mask.to(self.clip.device),
                    position_ids=position_ids
                )

                if avg_mode == "post_projection":
                    conditional_embeddings = conditional_embeddings.mean(dim=0)

                all_text_embeddings.append(conditional_embeddings)
            all_text_embeddings = torch.stack(all_text_embeddings, dim=0)

            all_text_embeddings = all_text_embeddings.cuda()
        return all_text_embeddings

    def get_conditional_embeddings(
            self,
            batch_size: int = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            conditional_pixel_values: Optional[torch.Tensor] = None,
    ):
        if input_ids is not None:
            # compute conditional embeddings from texts
            if len(input_ids) != batch_size:
                raise ValueError("Make sure to pass as many prompt texts as there are query images")
            with torch.no_grad():
                conditional_embeddings = self.clip.get_text_features(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids
                )
        elif conditional_pixel_values is not None:
            # compute conditional embeddings from images
            if len(conditional_pixel_values) != batch_size:
                raise ValueError("Make sure to pass as many prompt images as there are query images")
            with torch.no_grad():
                conditional_embeddings = self.clip.get_image_features(conditional_pixel_values)
        else:
            raise ValueError(
                "Invalid conditional, should be either provided as `input_ids` or `conditional_pixel_values`"
            )

        return conditional_embeddings

    def forward(
            self,
            input_ids: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            conditional_pixel_values: Optional[torch.FloatTensor] = None,
            conditional_embeddings: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPSegOutput]:
        r"""Labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoProcessor, CLIPSegForImageSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["a cat", "a remote", "a blanket"]
        >>> inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> print(logits.shape)
        torch.Size([3, 352, 352])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the query images through the frozen CLIP vision encoder
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=return_dict,
            )
            pooled_output = self.clip.visual_projection(vision_outputs[1])

            hidden_states = vision_outputs.hidden_states if return_dict else vision_outputs[2]
            # we add +1 here as the hidden states also include the initial embeddings
            activations = [hidden_states[i + 1] for i in self.extract_layers]

            # update vision_outputs
            if return_dict:
                vision_outputs = ExtendedModelOutput(
                    last_hidden_state=vision_outputs.last_hidden_state,
                    pooler_output=vision_outputs.pooler_output,
                    hidden_states=vision_outputs.hidden_states if output_hidden_states else None,
                    attentions=vision_outputs.attentions,
                    projected_pool_output=pooled_output
                )
            else:
                vision_outputs = (
                    vision_outputs[:2] + vision_outputs[3:] if not output_hidden_states else vision_outputs
                )

        # step 2: compute conditional embeddings, either from text, images or an own provided embedding
        if conditional_embeddings is None:
            conditional_embeddings = self.get_conditional_embeddings(
                batch_size=pixel_values.shape[0],
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                conditional_pixel_values=conditional_pixel_values,
            )
        else:

            n_prompts = conditional_embeddings.shape[0]

            conditional_embeddings_unrepeated = conditional_embeddings
            conditional_embeddings = conditional_embeddings.repeat_interleave(pixel_values.shape[0], dim=0)

            pixel_values = pixel_values.repeat(n_prompts, 1, 1, 1)

            activations = [act.repeat(n_prompts, 1, 1) for act in activations]

            if conditional_embeddings.shape[0] != pixel_values.shape[0]:
                raise ValueError(
                    "Make sure to pass as many conditional embeddings as there are query images in the batch"
                )
            if conditional_embeddings.shape[1] != self.config.projection_dim:
                raise ValueError(
                    "Make sure that the feature dimension of the conditional embeddings matches"
                    " `config.projection_dim`."
                )

        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        decoder_outputs = self.decoder(
            activations,
            conditional_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]

        loss = None
        if labels is not None:
            # move labels to the correct device to enable PP
            labels = labels.to(logits.device)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        if not return_dict:
            output = (logits, conditional_embeddings, pooled_output, vision_outputs, decoder_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPSegImageSegmentationOutputExtended(
            loss=loss,
            logits=logits,
            conditional_embeddings=conditional_embeddings,
            conditional_embeddings_unrepeated=conditional_embeddings_unrepeated,
            pooled_output=pooled_output,
            vision_model_output=vision_outputs,
            decoder_output=decoder_outputs,
        )


def filter_duplicate_class_assignments(detected_dict, frame_logits):
    filtered_dict = {}
    detected_classes = list(detected_dict.values())

    for class_idx, masks in detected_dict.items():
        if len(masks) == 1:
            filtered_dict[class_idx] = masks[0]
        else:
            merged_mask = torch.sum([mask["mask"].astype(bool) for mask in masks], dim=0)
            class_confidences = np.array([frame_logits[mask["label_idx"] - 1][merged_mask].sum() for mask in masks])
            max_confidence_idx = torch.argmax(torch.tensor(class_confidences))
            merged_entry = masks[max_confidence_idx].copy()
            merged_entry["mask"] = merged_mask
            merged_entry["confidence"] = frame_logits[masks[max_confidence_idx]["label_idx"] - 1][merged_mask].mean()

            filtered_dict[class_idx] = merged_entry

    if len(filtered_dict) <= 1:
        return filtered_dict
    scores = [frame_logits[val["label_idx"] - 1][val["mask"].astype(bool)].mean() for cur_idx, val in
              filtered_dict.items()]

    all_masks = np.stack([val["mask"] for val in list(filtered_dict.values())])
    boxes = masks_to_boxes(torch.tensor(all_masks))
    max_boxes_idx = nms(boxes, torch.tensor(scores), 0.6)
    filtered_dict_iou = {list(filtered_dict.keys())[idx]: list(filtered_dict.values())[idx] for idx in max_boxes_idx}
    if len(filtered_dict) != len(filtered_dict_iou):
        s = 1
    # ious = compute_iou(all_masks,all_masks)
    # s = 1

    return filtered_dict_iou
