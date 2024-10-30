from pathlib import Path
import numpy as np
import torch
from joblib import Parallel, delayed
import logging
from torch.nn.functional import box_iou

class ImageObjectDetector:
    def __init__(self, cfg, owl, vlm, detection_model, state_detector):
        """
        Initialize the ImageObjectDetector with required models and configuration.
        
        Args:
            cfg: Configuration object containing settings
            owl: OWL model for object detection
            vlm: Vision Language Model
            detection_model: Object detection model
            state_detector: State detection model
        """
        self.cfg = cfg
        self.owl = owl
        self.vlm = vlm
        self.detection_model = detection_model
        self.state_detector = state_detector
        self.lvis_classes = self._load_lvis_classes()

    def _load_lvis_classes(self):
        """Load and process LVIS classes from vocabulary file."""
        with open(self.cfg.vocab_file) as f:
            classes = f.read().splitlines()
            return np.array([x[x.find(':') + 1:] for x in classes])

    def get_objects_in_image(self, images, temporal_threshold=0.12, bsz=2, 
                           predefined_objects=None, use_som=True, 
                           use_som_mask=False, simple_ablation=False, 
                           use_obj_det_clip_baseline=False):
        """
        Main method to detect and analyze objects in images.
        
        Returns:
            tuple: (detected_objects, synonym_classes) or None if detection fails
        """
        # Initialize parameters
        predefined_objects = [] if predefined_objects is None else predefined_objects
        n_images_for_detection = 16
        image_subset = self._get_image_subset(images, n_images_for_detection)
        
        # Handle VLM-only predictions if configured
        if self.cfg.use_vlm_predictions_only:
            return self._handle_vlm_only_predictions(image_subset)

        # Process images with SOM prompting
        if not use_obj_det_clip_baseline:
            vlm_object_names, synonym_classes = self._process_with_som(
                image_subset, bsz, use_som, use_som_mask, simple_ablation)
            
            if vlm_object_names is None:
                return None

            # Filter and process detected objects
            return self._post_process_detections(
                vlm_object_names, 
                image_subset, 
                predefined_objects, 
                synonym_classes
            )
        
        return None

    def _get_image_subset(self, images, n_samples):
        """Select a subset of images for detection."""
        indices = np.linspace(0, len(images) - 1, n_samples, dtype=int)
        return images[indices]

    def _handle_vlm_only_predictions(self, image_subset):
        """Handle cases where only VLM predictions are used."""
        vlm_objects = self.vlm.get_objects_in_image(image_subset[0])
        if vlm_objects is not None:
            vlm_objects_list = [item.strip() for obj in vlm_objects.keys() 
                              for item in obj.split(",")]
            return vlm_objects_list + [self.lvis_classes[0]]
        return None

    def _process_with_som(self, image_subset, bsz, use_som, use_som_mask, simple_ablation):
        """Process images using SOM (Spatial Object Memory) approach."""
        n_frames = 1 if simple_ablation else 8
        frame_indices = np.linspace(0, len(image_subset) - 1, n_frames, dtype=int)
        init_obj_det_frames = image_subset[frame_indices]
        
        # Prepare annotated images
        annotated_images = self._prepare_annotated_images(
            init_obj_det_frames, bsz, use_som, use_som_mask)
        
        # Get VLM results
        results = self._get_vlm_results(annotated_images, n_frames)
        
        if simple_ablation:
            return results[0], None
            
        if any(result is None for result in results):
            return None, None

        return self._process_vlm_results(results, init_obj_det_frames, bsz)

    def _prepare_annotated_images(self, frames, bsz, use_som, use_som_mask):
        """Prepare annotated images for processing."""
        annotated_images = []
        robot_detection = self.detection_model.detect_objects(
            frames, ["robot gripper"], threshold=0.1)
            
        for idx, frame in enumerate(frames):
            if use_som:
                img_annotated = create_som_annotated_image(
                    np.array(frame), bsz, idx, use_som_mask)
                annotated_images.append(img_annotated.get_image())
            else:
                annotated_images.append(np.array(frame))
        
        return annotated_images

    def _get_vlm_results(self, annotated_images, n_frames):
        """Get results from Vision Language Model."""
        if "prismatic" in self.vlm.__class__.__name__.lower():
            return [self.vlm.get_objects_in_image(img) for img in annotated_images]
        
        return Parallel(n_jobs=n_frames, backend="threading")(
            delayed(get_objects_in_image_batched)(img) 
            for img in annotated_images
        )

    def _process_vlm_results(self, results, frames, bsz):
        """Process VLM results to get final object detections."""
        processed_results = self._process_detections(results, frames, bsz)
        if processed_results is None:
            return None, None
            
        vlm_objects, synonym_classes = processed_results
        return self._filter_results(vlm_objects), synonym_classes

    def _filter_results(self, objects):
        """Filter out robot-related objects and duplicates."""
        return [obj for obj in objects 
                if "robot" not in obj["name"].lower() 
                and "gripper" not in obj["name"].lower() 
                and "arm" not in obj["name"].lower()]

    def _post_process_detections(self, vlm_object_names, image_subset, 
                               predefined_objects, synonym_classes):
        """Post-process detected objects and handle predefined objects."""
        ov_detections, valid_classes = filter_object_names_by_confidence(
            self.detection_model, image_subset, vlm_object_names)
            
        if predefined_objects:
            vlm_object_names = self._handle_predefined_objects(
                vlm_object_names, predefined_objects, image_subset, 
                ov_detections, valid_classes)
            
        return vlm_object_names, synonym_classes

    def _handle_predefined_objects(self, vlm_objects, predefined_objects, 
                                 images, ov_detections, valid_classes):
        """Handle predefined objects and their interactions with detected objects."""
        predefined_prompts = [
            f"{obj['color']} {obj['name']}" for obj in predefined_objects
        ]
        pred_detections = self.detection_model.detect_objects(
            np.array(images), predefined_prompts, reduce_threshold=True)
            
        overlap_counter = detect_high_overlap_objects(
            ov_detections, pred_detections)
            
        high_overlap_ids = [
            k for k, v in overlap_counter.items() 
            if v >= len(pred_detections) // 2 and k in valid_classes
        ]
        
        return [obj for idx, obj in enumerate(vlm_objects) 
                if idx not in high_overlap_ids]
        
    Refactored Image Object Detection Class

from pathlib import Path
import numpy as np
import torch
from joblib import Parallel, delayed
import logging
from torch.nn.functional import box_iou

class ImageObjectDetector:
    def __init__(self, cfg, owl, vlm, detection_model, state_detector):
        """
        Initialize the ImageObjectDetector with required models and configuration.
        
        Args:
            cfg: Configuration object containing settings
            owl: OWL model for object detection
            vlm: Vision Language Model
            detection_model: Object detection model
            state_detector: State detection model for CLIP-based scoring
        """
        self.cfg = cfg
        self.owl = owl
        self.vlm = vlm
        self.detection_model = detection_model
        self.state_detector = state_detector
        self.lvis_classes = self._load_lvis_classes()

    # ... [previous methods remain the same until _process_vlm_results] ...

    def _process_vlm_results(self, results, frames, bsz):
        """Process VLM results to get final object detections."""
        vlm_object_names = []
        obj_set_detections = []
        results_cleaned = []
        surface_objects = []
        surface_obj_scores = []
        possible_surface_objects = []
        possible_surface_scores = []

        # Get region proposals from OWL
        n_boxes = 15
        region_proposals, objectness, grounding_scores, class_embeds = self.owl.get_region_proposals(
            frames, bsz, n_boxes=n_boxes, text_classifier=None
        )

        # Process each result
        for idx, result in enumerate(results):
            # Create prompts for object verification
            ov_prompts = [f"{obj['color']} {obj['name']}" for obj in result]
            ov_prompts = [name.replace(".", "") for name in ov_prompts]
            
            # Detect objects using the detection model
            detections = self.detection_model.detect_objects(
                np.array(frames), ov_prompts, bsz=8
            )
            
            # Get CLIP-based scores for detected objects
            clip_scores = self._get_clip_class_scores(detections, frames, ov_prompts)
            
            frame_detections = self._process_frame_detections(
                detections, clip_scores, frames, result, 
                region_proposals, objectness, 
                surface_objects, surface_obj_scores,
                possible_surface_objects, possible_surface_scores
            )
            
            obj_set_detections.append(frame_detections)
            results_cleaned.append(result[1:])
            vlm_object_names += result

        # Process surface objects
        best_surface_obj = self._get_best_surface_object(
            surface_objects, surface_obj_scores,
            possible_surface_objects, possible_surface_scores
        )

        # Calculate temporal co-occurrences
        return self._calculate_temporal_cooccurrences(
            frames, obj_set_detections, results_cleaned, 
            best_surface_obj, vlm_object_names
        )

    def _get_clip_class_scores(self, detections, frames, class_prompts):
        """
        Calculate CLIP-based scores for detected objects using the state detector.
        
        Args:
            detections: Object detections from the detection model
            frames: Input image frames
            class_prompts: Text prompts for each class
            
        Returns:
            list: CLIP scores for each detection
        """
        all_scores = []
        
        for frame_idx, frame in enumerate(frames):
            frame_detections = detections[frame_idx]
            if len(frame_detections) == 0:
                all_scores.append([])
                continue

            # Extract crops for each detection
            crops = []
            for box in frame_detections.xyxy:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)

            # Get CLIP scores from state detector
            clip_scores = self.state_detector.get_clip_scores(
                crops, class_prompts
            )
            all_scores.append(clip_scores)

        return all_scores
        
    def _process_frame_detections(self, detections, clip_scores, frames, result,
                                region_proposals, objectness, 
                                surface_objects, surface_obj_scores,
                                possible_surface_objects, possible_surface_scores):
        """Process detections for each frame incorporating CLIP scores."""
        all_frame_detections = []
        
        for frame_idx, detection in enumerate(detections):
            image_area = frames[frame_idx].shape[0] * frames[frame_idx].shape[1]
            
            # Process surface objects
            surface_obj, surface_score, possible_obj, possible_score = \
                self._process_surface_objects(detection, image_area, result)
            
            surface_objects.append(surface_obj)
            surface_obj_scores.append(surface_score)
            possible_surface_objects.append(possible_obj)
            possible_surface_scores.append(possible_score)

            # Filter and score detections
            filtered_detections = self._filter_and_score_detections(
                detection, clip_scores[frame_idx],
                region_proposals[frame_idx], objectness[frame_idx]
            )
            
            all_frame_detections.append(filtered_detections)
            
        return all_frame_detections
    
  def _filter_and_score_detections(self, detections, clip_scores, 
                                   region_proposals, objectness):
        """Filter detections and incorporate CLIP and objectness scores."""
        # Remove surface objects
        non_surface_mask = detections.class_id != 0
        detections = detections[non_surface_mask]
        clip_scores = clip_scores[non_surface_mask]
        
        # Filter by confidence
        conf_mask = detections.confidence > 0.2
        detections = detections[conf_mask]
        clip_scores = clip_scores[conf_mask]
        
        # Incorporate CLIP scores
        detections_with_clip = self._incorporate_clip_scores(
            detections, clip_scores
        )
        
        # Get best detection per class
        best_detections = self._get_best_detection_per_class(
            detections_with_clip
        )
        
        # Incorporate objectness scores
        return self._incorporate_objectness_scores(
            best_detections, region_proposals, objectness
        )