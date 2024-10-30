
import cv2
import numpy as np
import PIL
import supervision as sv
import torch
import torchvision
from tqdm import tqdm
from transformers import (
    AutoModelForZeroShotObjectDetection,
    GroundingDinoProcessor,
)
from transformers.image_transforms import center_to_corners_format
from transformers.models.grounding_dino.processing_grounding_dino import (
    get_phrases_from_posmap,
)

from nils.specialist_models.detectors.utils import create_batches

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GroundingDINOHF():
    def __init__(self, model_id = "IDEA-Research/grounding-dino-base",box_threshold = 0.35,text_threshold = 0.25):
        # self.detection_processor = Owlv2Processor.from_pretrained(hf_path)
        #model_id = "IDEA-Research/grounding-dino-base"
       

        self.processor = CustomGroundingDinoProcessor.from_pretrained(model_id)
        self.detection_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        self.separator_id = 1012

    
    def to_cpu(self):
        self.detection_model = self.detection_model.to('cpu')
        torch.cuda.empty_cache()
        
    def detect_objects(self, images, classes,threshold = 0.24, bsz = 4,reduce_threshold = False):

        self.detection_model = self.detection_model.to(device)
      
        
        
        
        images_split = create_batches(bsz, images)
        texts = [classes]

        boxes = []
        scores = []
        labels = []

        queries = ""
        for query in classes:
            queries += f"{query}. "
        
        for i in tqdm(range(len(images_split)), disable=True):
            image = images_split[i]
            #images_transformed, _ = self.detection_processor(image,None)
            
            prompt = ". ".join(classes)
            prompt = [queries] * len(image)
            inputs_processed = self.processor(images= image, text = prompt,return_tensors="pt").to(device)
            
            #images_preprocessed = images_preprocessed.squeeze()
            with torch.inference_mode():
                #with torch.autocast(device_type=device):
                outputs = self.detection_model(**inputs_processed)
                
            
            

            sep_inidices = torch.where(inputs_processed.input_ids[0] == self.separator_id)[0].cpu()
            token_span_per_class = list(inputs_processed.input_ids[0].cpu().tensor_split(sep_inidices))
            token_span_lens = [len(span) -1 for span in token_span_per_class]
            token_span_lens = token_span_lens[:-1]
            # token_span_lens[0] = token_span_lens[0] - 1
            #token_span_lens[-1] = token_span_lens[-1] - 1
            
            target_sizes = [image.shape[1:]] * len(image)
            
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs_processed.input_ids,
                box_threshold=threshold,
                text_threshold=threshold,
                target_sizes=target_sizes,
                token_span_lens=token_span_lens,
                reduce_threshold = reduce_threshold
            )
            # del outputs
            # torch.cuda.empty_cache()


        
            #for box, logit, phrase in zip(boxes, logits, phrases):
            for result in results:
                boxes.append(result["boxes"].cpu().numpy())
                scores.append(result["scores"].cpu().numpy())
                labels.append(result["labels"])
            


        # boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        # xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        # confidence = logits.numpy()

        detections = [sv.Detections(xyxy=box, confidence=score, class_id=np.array(label)) for box, score, label in
                      zip(boxes, scores, labels)]
        # detections = sv.Detections(xyxy=np.array(torch.cat(boxes)), confidence=np.array(torch.cat(scores)),
        #                           class_id=np.array(torch.cat(labels)))
        # detections.class_id = text_labels
        
        # self.detection_model = self.detection_model.to('cpu')
        # torch.cuda.empty_cache()

       
        
        return detections

    def resize(self, image, size):
        return self.detection_processor.image_processor.resize(image, size)



class CustomGroundingDinoProcessor(GroundingDinoProcessor):
    
    def post_process_grounded_object_detection(
        self,
        outputs,
        input_ids,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        target_sizes = None,
        token_span_lens = None,
        reduce_threshold = False
    ):
        """Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text label.

        Args:
            outputs ([`GroundingDinoObjectDetectionOutput`]):
                Raw outputs of the model.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The token ids of the input text.
            box_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep object detection predictions.
            text_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep text detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        probs = torch.sigmoid(logits)  # (batch_size, num_queries, 256)
        scores = torch.max(probs, dim=-1)[0]  # (batch_size, num_queries)

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, list):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        
        if token_span_lens is not None:
            for idx, (s, b, p) in enumerate(zip(scores, boxes, probs)):
                last_start_idx = 1
                #results.append({"scores": [], "labels": [], "boxes": []})

                batch_scores, batch_boxes, batch_labels = get_results_from_token_scores(s, b, p, token_span_lens, box_threshold, last_start_idx)

                res_dict = {"scores": batch_scores, "labels": batch_labels, "boxes": batch_boxes}

                results.append(res_dict)

                # for class_id in range(len(token_span_lens)):
                #     start_token_idx = last_start_idx
                #     end_token_idx = last_start_idx + token_span_lens[class_id]
                #
                #     token_probs = p[:, start_token_idx:end_token_idx]
                #
                #     #token_probs_summed = torch.sum(token_probs, dim=-1)
                #     token_probs_mean = torch.mean(token_probs, dim=-1)
                #     max_box = torch.argmax(token_probs_mean)
                #
                #     class_box_indices = token_probs_mean > box_threshold
                #     class_boxes = b[class_box_indices]
                #     class_scores = s[class_box_indices]
                #
                #     #print(self.batch_decode(input_ids[idx][start_token_idx:end_token_idx]))
                #     #print(token_probs_mean.max())
                #
                #     for box, score in zip(class_boxes, class_scores):
                #         results[-1]["scores"].append(score)
                #         results[-1]["boxes"].append(box)
                #         results[-1]["labels"].append(class_id)
                #
                #
                #     last_start_idx = end_token_idx + 1

                if len(results[-1]["boxes"]) == 0 and reduce_threshold:
                    batch_scores, batch_boxes, batch_labels = get_results_from_token_scores(s, b, p,
                                                                                            token_span_lens,
                                                                                            box_threshold/2,
                                                                                            last_start_idx)
                    res_dict = {"scores": batch_scores, "labels": batch_labels, "boxes": batch_boxes}
                    results[-1] = res_dict

                if len(results[-1]["boxes"]) > 0:
                    results[-1]["scores"] = torch.stack(results[-1]["scores"])
                    results[-1]["boxes"] = torch.stack(results[-1]["boxes"])
                    results[-1]["labels"] = results[-1]["labels"]
                    #perform nms
                    if len(results[-1]["boxes"]) > 0:
                        keep = torchvision.ops.nms(results[-1]["boxes"], results[-1]["scores"], 0.7)
                        results[-1]["scores"] = results[-1]["scores"][keep]
                        results[-1]["boxes"] = results[-1]["boxes"][keep]
                        results[-1]["labels"] = [results[-1]["labels"][i] for i in keep]
                else:
                    results[-1] = {"scores": torch.tensor([]), "labels": torch.tensor([]), "boxes": torch.tensor([])}
            
            
            
            return results
                
                    
                    
                    
                    
        else:
            for idx, (s, b, p) in enumerate(zip(scores, boxes, probs)):
                score = s[s > box_threshold]
                box = b[s > box_threshold]
                prob = p[s > box_threshold]
                label_ids = get_phrases_from_posmap(prob > text_threshold, input_ids[idx])
                label = self.batch_decode(label_ids)
                results.append({"scores": score, "labels": label, "boxes": box})

        return results


def get_results_from_token_scores(s, b, p, token_span_lens, box_threshold, last_start_idx):
    cur_scores = []
    cur_boxes = []
    cur_labels = []

    for class_id in range(len(token_span_lens)):
        start_token_idx = last_start_idx
        end_token_idx = last_start_idx + token_span_lens[class_id]

        token_probs = p[:, start_token_idx:end_token_idx]

        # token_probs_summed = torch.sum(token_probs, dim=-1)
        token_probs_mean = torch.mean(token_probs, dim=-1)
        max_box = torch.argmax(token_probs_mean)

        class_box_indices = token_probs_mean > box_threshold
        class_boxes = b[class_box_indices]
        class_scores = s[class_box_indices]

        # print(self.batch_decode(input_ids[idx][start_token_idx:end_token_idx]))
        # print(token_probs_mean.max())

        for box, score in zip(class_boxes, class_scores):
            cur_scores.append(score)
            cur_boxes.append(box)
            cur_labels.append(class_id)

        last_start_idx = end_token_idx + 1

    return cur_scores, cur_boxes, cur_labels
