import gc
import os

import cv2
import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou, masks_to_boxes
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers.image_transforms import center_to_corners_format

from nils.specialist_models.clip_utils import VILD_PROMPT, split_label
from nils.specialist_models.detectors.utils import (
    create_batches,
    plot_boxes_np,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OWLv2():
    def __init__(self, hf_path, **kwargs):
        self.detection_processor = Owlv2Processor.from_pretrained(hf_path)
        self.detection_model = Owlv2ForObjectDetection.from_pretrained(hf_path, torch_dtype=torch.float16

                                                                       ).to(device)
        
        
        self.name = hf_path.split("/")[-1]
        
        
        self.image_feat_cache = None

    def to_cpu(self):
        self.detection_model = self.detection_model.to('cpu')
        torch.cuda.empty_cache()

        
    def detect_objects_with_class_embeds(self, images,class_embeds, threshold = 0.2,bsz = 1):
        
        
        self.detection_model.to(device)
        bsz = min(images.shape[0], bsz)

        images_split = create_batches(bsz, images)
        
        class_embeds = torch.tensor(class_embeds).to(device)
        
        boxes = []
        scores = []
        labels = []

        for i in tqdm(range(len(images_split)), desc = "Detecting objects", disable = True):
            image = images_split[i]
            if image.ndim == 3:
                image = image[None, ...]
            inputs = self.detection_processor(images=image, return_tensors="pt").pixel_values
            inputs = {key: val.to(device) for key, val in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.detection_model(image_embeds = class_embeds, **inputs)
            # target_sizes = (torch.Tensor([image.squeeze().shape[0:2]])).repeat_interleave(self.config.detector.bsz,
            #                                                                               dim=0).to(device)
            target_sizes = (torch.Tensor([max(image.shape[1:3]),max(image.shape[1:3])])).repeat(image.shape[0], 1).to(device)

            results = self.detection_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                                             threshold=threshold)
            
            del outputs
            torch.cuda.empty_cache()
            for result in results:
                boxes.append(result["boxes"].detach().cpu().numpy())
                scores.append(result["scores"].detach().cpu().numpy())
                labels.append(result["labels"].detach().cpu().numpy())


        detections = [sv.Detections(xyxy=box, confidence=score, class_id=label) for box, score, label in
                      zip(boxes, scores, labels)]

        return detections

    def detect_objects(self, images, classes, threshold=0.2,bsz = 1):
        """Args:
            images (np.ndarray):(B, H, W, C)
            classes (List):

        Returns:

        """
        self.detection_model.to(device)
        bsz = min(images.shape[0], bsz)

        images_split = create_batches(bsz, images)

        texts = [classes]

        boxes = []
        scores = []
        labels = []

        for i in tqdm(range(len(images_split)), desc = "Detecting objects"):
            image = images_split[i]
            if image.ndim == 3:
                image = image[None, ...]
            inputs = self.detection_processor(texts * image.shape[0], images=image, return_tensors="pt", max_length=16, truncation=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.detection_model(**inputs)
            # target_sizes = (torch.Tensor([image.squeeze().shape[0:2]])).repeat_interleave(self.config.detector.bsz,
            #                                                                               dim=0).to(device)
            #target_sizes = (torch.Tensor([max(image.shape[1:3]),max(image.shape[1:3])])).repeat(image.shape[0], 1).to(device)
            target_sizes = (torch.Tensor([image.shape[1],image.shape[2]])).repeat(image.shape[0], 1).to(device)

            results = self.detection_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                                             threshold=threshold)
            
            del outputs
            torch.cuda.empty_cache()
            for result in results:
                boxes.append(result["boxes"].detach().cpu().numpy())
                scores.append(result["scores"].detach().cpu().numpy())
                labels.append(result["labels"].detach().cpu().numpy())


        detections = [sv.Detections(xyxy=box, confidence=score, class_id=label) for box, score, label in
                      zip(boxes, scores, labels)]

        return detections

    def resize(self, image, size):
        return self.detection_processor.image_processor.resize(image, size)
    
    
    def to_cpu(self):
        self.detection_model.to('cpu')

    def get_ref_image_features(self, ref_images, ref_masks,compare_based_on):
        ref_source = ref_images
        ref_source_processed = self.detection_processor(images=ref_source, return_tensors="pt").pixel_values

        ref_masks_processed = self.detection_processor.image_processor(ref_masks[..., None], do_normalize=False,
                                                                       return_tensors="pt").pixel_values.squeeze(1)

        batched_source = create_batches(4, ref_source_processed)

        objectnesses = []
        ref_boxes = []
        source_class_embeddings = []
        im_features = []
        for batch in batched_source:
            with torch.no_grad():
                ref_feature_maps = self.detection_model.image_embedder(batch.to(device))[0]
                print(ref_feature_maps.shape)

                batch_size, height, width, hidden_size = ref_feature_maps.shape
                image_features = ref_feature_maps.reshape(batch_size, height * width, hidden_size)
                im_features.append(image_features)
                objectness = self.detection_model.objectness_predictor(image_features)
                ref_box = self.detection_model.box_predictor(image_features, ref_feature_maps)
                source_class_embedding = self.detection_model.class_predictor(image_features)[1]
                objectnesses.append(objectness.cpu().to(torch.float32))
                ref_boxes.append(ref_box.cpu().to(torch.float32))
                source_class_embeddings.append(source_class_embedding.cpu().to(torch.float32))
        objectnesses = torch.cat(objectnesses)
        ref_boxes = torch.cat(ref_boxes)
        im_features = torch.cat(im_features)
        source_class_embeddings = torch.cat(source_class_embeddings)

        gt_bounding_boxes = masks_to_boxes(torch.tensor(ref_masks_processed))
        gt_bounding_boxes_normalized = gt_bounding_boxes / ref_masks_processed.shape[1]
        ref_embeddings = []
        final_boxes = []
        plot_debug= False

        for i in range(len(ref_masks)):
            iou = box_iou(ref_boxes[i], gt_bounding_boxes_normalized[i][None, ...])
            best_5_iou = torch.topk(iou, 5, dim=0).indices
            best_5_objectness = torch.topk(objectnesses[i], 2, dim=0).indices
            if plot_debug:
                plot_boxes_np(cv2.resize(ref_images[i], (ref_masks_processed.shape[1], ref_masks_processed.shape[1])),
                              [gt_bounding_boxes[i]])
                
                for best_iou in best_5_iou:
                    plot_boxes_np(cv2.resize(ref_images[i], (ref_masks_processed.shape[1], ref_masks_processed.shape[1])),
                                  [ref_boxes[i][best_iou][0].numpy() *  ref_masks_processed.shape[1], gt_bounding_boxes[i]])

                for best_objectnes in best_5_objectness:
                    plot_boxes_np(cv2.resize(ref_images[i], (ref_masks_processed.shape[1], ref_masks_processed.shape[1])),
                                  [ref_boxes[i][best_objectnes].numpy() *  ref_masks_processed.shape[1]])

            best_iou_idx = torch.argmax(iou)
            if compare_based_on == "hidden":
                ref_embeddings.append(im_features[i,best_iou_idx,:].cpu())
            else:
                ref_embeddings.append(source_class_embeddings[i][best_iou_idx])
            final_boxes.append(ref_boxes[i][best_iou_idx])

        ref_embeddings = torch.stack(ref_embeddings)
        final_boxes = torch.stack(final_boxes)

        return ref_embeddings, final_boxes
    
    
    
    def get_text_embeddings(self,texts):

        #self.detection_model.to(device)
        
        processed_texts = self.detection_processor(texts, return_tensors="pt",truncation=True,max_length = 16)
        
        
        input_ids = processed_texts["input_ids"]

        max_text_queries = input_ids.shape[0] // len(texts)
        input_ids = input_ids.reshape( len(texts), max_text_queries, input_ids.shape[-1])
        query_mask = input_ids[..., 0] > 0
        

        with torch.no_grad():
            processed_texts = {key: val.to(device) for key, val in processed_texts.items()}
            
            text_outputs = self.detection_model.owlv2.text_model(**processed_texts)
            
            text_embeds = text_outputs[1]
            text_embeds = self.detection_model.owlv2.text_projection(text_embeds)
            
            text_embeds = text_embeds.cpu()
            
        return text_embeds
            
            
    def get_text_classifier(self,classes,cache_dir = None):
        
        if cache_dir is not None and os.path.exists(cache_dir):
            return torch.load(cache_dir)
        
        descriptions = []
        candidates = []
        for cls_name in classes:
            labels_per_cls = split_label(cls_name)
            candidates.append(len(labels_per_cls))
            for label in labels_per_cls:
                for template in VILD_PROMPT:
                    description = template.format(label)
                    descriptions.append(description)

        bsz = 256
        NUM_BATCH = len(descriptions) // bsz
        
        NUM_BATCH = max(1,NUM_BATCH)

        bs = len(descriptions)
        local_bs = bs // NUM_BATCH
        if bs % NUM_BATCH != 0:
            local_bs += 1
        feat_list = []
        for i in tqdm(range(NUM_BATCH), desc="Embedding texts", disable=True):
            local_descriptions = descriptions[i * local_bs: (i + 1) * local_bs]
            local_feat = self.get_text_embeddings(local_descriptions)
            feat_list.append(local_feat)
        features = torch.cat(feat_list)

        dim = features.shape[-1]
        candidate_tot = sum(candidates)
        candidate_max = max(candidates)
        features = features.reshape(candidate_tot, len(VILD_PROMPT), dim)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.mean(dim=1, keepdims=False)
        features = features / features.norm(dim=-1, keepdim=True)
        
        

        cur_pos = 0
        classifier = []
        for candidate in candidates:
            cur_feat = features[cur_pos:cur_pos + candidate]
            
            if candidate < candidate_max:
                cur_feat = torch.cat([cur_feat, cur_feat[0].repeat(candidate_max - candidate, 1)])
            classifier.append(cur_feat)
            cur_pos += candidate
        classifier = torch.stack(classifier)

        save_path = cache_dir
        classifier_to_save = classifier
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(classifier_to_save, save_path)
        
        return classifier
            
    
    
    
    def get_region_proposals(self,images,bsz, n_boxes = 15,text_classifier = None,return_class_embeds = False, reduction = "sig",
                             save_cache = False):

        #self.detection_model.to(device)

        bsz = min(images.shape[0], bsz)

        images_split = create_batches(bsz, images)

        
        
        
        if text_classifier is not None:
            text_classifier = torch.mean(text_classifier,dim = 1).to(device)
        
        
        
        all_class_embeds = []
        
        
        all_objectnesses = []
        all_boxes = []
        all_scores = []
        all_class_embeds = []
        
        if self.image_feat_cache is None:
            
            image_feat_cache = []
        for i in tqdm(range(len(images_split)),disable=True):
            image = images_split[i]
            if image.ndim == 3:
                image = image[None, ...]
            inputs = self.detection_processor(images=image, return_tensors="pt").pixel_values
            
            max_side = max(image.shape[1:3])
            offset = (max_side - np.array(image.shape[1:3])) // 2
            offset = offset[::-1]

            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    
                    if self.image_feat_cache is not None:
                        feature_map = self.image_feat_cache[i].cuda()
                        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
                        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

                    else:
                        feature_map = self.detection_model.image_embedder(inputs.to(device))[0]
                        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
                        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))
                    
                        image_feat_cache.append(feature_map.cpu())
                    
                    objectness_logits = self.detection_model.objectness_predictor(image_feats).to(torch.float32)
                    pred_boxes = self.detection_model.box_predictor(image_feats, feature_map).cpu().cpu().to(torch.float32)
                    
                   
                    objectnesses = torch.sigmoid(objectness_logits.cpu())

                    #objectness_threshold = np.partition(objectnesses, -n_boxes,axis = -1)[:,-n_boxes]

                    if n_boxes == -1:
                        objectness_indices = torch.argsort(objectnesses, descending=True)
                    else:
                        objectness_indices = torch.argsort(objectnesses, descending=True)[:, :n_boxes]

                    pred_boxes = pred_boxes[np.arange(pred_boxes.shape[0])[:,None], objectness_indices]
                    objectness_subset = objectnesses[np.arange(objectnesses.shape[0])[:,None], objectness_indices].cpu().numpy()
                    scores = None
                    if text_classifier is not None:

                        image_feats_subset = image_feats[np.arange(image_feats.shape[0])[:,None], objectness_indices]



                        pred_logits,class_embeds = self.detection_model.class_predictor(image_feats_subset, text_classifier)
                        if reduction == "sig":
                            scores = torch.sigmoid(pred_logits).to(torch.float32)
                        elif reduction is None:
                            scores = pred_logits.to(torch.float32)
                        del pred_logits,feature_map,objectness_logits
                        torch.cuda.empty_cache()

                        s =1
                    if return_class_embeds:
                        image_feats_subset = image_feats[np.arange(image_feats.shape[0])[:, None], objectness_indices]
                        class_embeds = self.detection_model.class_predictor(image_feats_subset)[1]
                        all_class_embeds.append(class_embeds.cpu().to(torch.float32).numpy())
                        
                    
            #rescale boxes (cx,cy,w,h) to (x1,y1,x2,y2)
            #boxes given as bsz x n_boxes x 4
            pred_boxes = center_to_corners_format(pred_boxes)

            img_h = image.shape[1]
            img_w = image.shape[2]



            pred_boxes[:, :, 0] *= max_side
            pred_boxes[:, :, 1] *= max_side
            pred_boxes[:, :, 2] *= max_side
            pred_boxes[:, :, 3] *= max_side



            #rescale to image size

            pred_boxes = np.array(pred_boxes).astype(np.int32)





            all_objectnesses.append(objectness_subset)
            all_boxes.append(pred_boxes)
            if scores is not None:
                all_scores.append(scores.cpu().numpy())

        all_boxes = np.concatenate(all_boxes)
        all_objectnesses = np.concatenate(all_objectnesses)
        
        if self.image_feat_cache is None and save_cache:
            self.image_feat_cache = image_feat_cache
            
        
        if return_class_embeds:
            all_class_embeds = np.concatenate(all_class_embeds)
        if scores is not None:
            all_scores = np.concatenate(all_scores)
        else:
            all_scores = None

        #self.detection_model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

        
                
        return all_boxes, all_objectnesses, all_scores, all_class_embeds
    
    
    def reset_cache(self):
        self.image_feat_cache = None
    
    def get_region_text_alignment(self,image_feats,texts):
        pass
        
                    


    def detect_mask_conditioned(self, images, ref_images, ref_masks, compare_based_on = "hidden", threshold=0.12, is_embedding=False):
        bsz = min(images.shape[0], 4)

        images_split = create_batches(bsz, images)

        boxes = []
        scores = []
        labels = []

        query_embeddings,ref_areas = self.get_ref_image_features(ref_images, ref_masks,compare_based_on = compare_based_on)


        for i in (range(len(images_split))):
            image = images_split[i]
            if image.ndim == 3:
                image = image[None, ...]
            target_pixel_values = self.detection_processor(images=image, return_tensors="pt").pixel_values


           
            with torch.inference_mode():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    feature_map = self.detection_model.image_embedder(target_pixel_values.to(device))[0]
                    b, h, w, d = feature_map.shape
                    target_boxes = self.detection_model.box_predictor(
                        feature_map.reshape(b, h * w, d), feature_map=feature_map
                    )
                    if compare_based_on == "class_token":
                        target_class_predictions = self.detection_model.class_predictor(
                            feature_map.reshape(b, h * w, d),
                            torch.tensor(query_embeddings[None, ...].to(device)),  # [batch, queries, d]
                        )[0]



            target_boxes = np.array(target_boxes.cpu().detach().to(torch.float32))
            if compare_based_on == "hidden":
                target_emebds = feature_map.reshape(b, h * w, d).cpu()
                target_emebds_norm = F.normalize(target_emebds, dim=-1)

                ref_emebds_norm = F.normalize(query_embeddings, dim=-1)

                sim = torch.einsum('bnd,md->bnm',target_emebds_norm.to(torch.float32), ref_emebds_norm.to(torch.float32))

                target_logits = sim
            else:
                target_logits = np.array(target_class_predictions.cpu().detach().to(torch.float32))



            target_areas = (target_boxes[:,:, 2] - target_boxes[:,:, 0]) * (target_boxes[:,:, 3] - target_boxes[:,:, 1])
            ref_areas = (ref_areas[:, 2] - ref_areas[:, 0]) * (ref_areas[:, 3] - ref_areas[:, 1])


            area_diff = np.abs(target_areas[...,None] - ref_areas[None,None,...].numpy())
            target_logits_softmax = np.array(torch.softmax(torch.tensor(target_logits),dim = 1))

            top_ind_weighted = np.argmax(target_logits_softmax - area_diff, axis=1)
            top_ind = np.argmax(target_logits, axis=1)




            for ref_idx in range(query_embeddings.shape[0]):

                cur_top_ind = top_ind[0,ref_idx]
                cur_target_boxes = target_boxes[0,cur_top_ind,:]
                plot_boxes_np(cv2.resize(image[0], (ref_masks.shape[1], ref_masks.shape[1])),
                              [cur_target_boxes *  ref_masks.shape[1]])

                #score = sigmoid(target_logits[top_ind, 0])


            # target_sizes = (torch.Tensor([image.squeeze().shape[0:2]])).repeat_interleave(self.config.detector.bsz,
            #                                                                               dim=0).to(device)
            target_sizes = (torch.Tensor([image.shape[1:3]])).repeat(image.shape[0], 1).to(device)



            results = self.detection_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                                             threshold=threshold)

            for result in results:
                boxes.append(result["boxes"].detach().cpu().numpy())
                scores.append(result["scores"].detach().cpu().numpy())
                labels.append(result["labels"].detach().cpu().numpy())

        # boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        # xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        # confidence = logits.numpy()

        detections = [sv.Detections(xyxy=box, confidence=score, class_id=label) for box, score, label in
                      zip(boxes, scores, labels)]
        # detections = sv.Detections(xyxy=np.array(torch.cat(boxes)), confidence=np.array(torch.cat(scores)),
        #                           class_id=np.array(torch.cat(labels)))
        # detections.class_id = text_labels
        
        
        


