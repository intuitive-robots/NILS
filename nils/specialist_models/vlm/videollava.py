import base64
import io
import json
import logging
import re
import time

import numpy as np
import torch
from PIL import Image
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor


class VideoLLava:
    def __init__(self):
        
        torch.cuda.empty_cache()
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


        self.model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", device_map="auto")
        self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
        
        
        
    def query_grounding(self, frames,prompts):
        
        
        assert len(frames[0]) == 8
        
        
        
        
        #frames = torch.stack(frames)
        responses = []
      
        prompt_formatted = "USER: <video>What did the robot do? Write a description including what object the robot interacted with, and the inital and final object relations. ASSISTANT:"
        #prompt_ours_formatted = f"USER: <video>{prompts} ASSISTANT:"
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(text=[prompt_formatted] * len(frames), videos=list(frames), return_tensors="pt")
        out = self.model.generate(**inputs, max_new_tokens=200,temperature=0.4)
        response = self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        return  response
    
    
        
    
    def get_grounding_predictions_baseline_vlm_batched(self,keystates, batch, possible_actions, open_ended=True):
        images = []
        for keystate_idx, ks in enumerate(keystates):
            ks = min(ks, len(batch["frame_name"]) - 1)
            if keystate_idx == 0:
                last_keystate_idx = 0
            if ks < 4:
                last_keystate_idx = ks
                continue
            length = ks - last_keystate_idx
    
            length = min(8, length)
            start_idx = max(0, ks - length)
    
            image_final_index = min(ks, len(batch["frame_name"]) - 1)
            images_keystate = batch["rgb_static"][start_idx:image_final_index]
            images.append(images_keystate)
        
        results = []
        prepared_images = []
        for image_subset in images:
            
            n_frames = 8
            frame_indices = np.linspace(0, len(image_subset) - 1, n_frames).astype(int)
            
            image_subset = [image_subset[i] for i in frame_indices]
            
            if len(image_subset) < 8:
                image_subset += [image_subset[-1]] * (8 - len(image_subset))
            
            image_subset = torch.stack(image_subset)
            prepared_images.append(image_subset)
            
        
        prepared_images = torch.stack(prepared_images)
        
        batched_images = []
        bsz = 8
        for i in range(0, len(prepared_images), bsz):
            batched_images.append(prepared_images[i:i + 8])
        
        for batched_image in batched_images:
            responses = self.query_grounding(batched_image[[0]], possible_actions)
            results.extend(responses)
        bsz = 8
        
    


        return results

def get_action_prediction_prompt_baseline(possible_actions, interacted_objects, use_som, observations=None,
                                          open_ended=False):
    actions_nl = ["- " + action for action in possible_actions]
    actions_nl = "\n".join(actions_nl)
    prompt = """Given the video frames, what task did the robot perform?
    Sometimes multiple tasks are possible. Output all possible tasks delimited by commas, up to two tasks."""

    # You will also be provided with a textual information of the robot interactions with the scene. 
    # If information is missing in the language description (indicated by not detected), infer it from the images
    prompt_som = """The objects in the frames are marked with bounding boxes and labeled."""
    prompt_p2 = """Choose up to two tasks from this task list:
```
[TASK_LIST]
```
"""
    # Observations:
    # ```
    # [OBSERVATIONS]
    # ```
    if use_som:
        prompt = prompt + prompt_som + prompt_p2
    else:
        prompt = prompt + prompt_p2
    prompt = prompt.replace("[TASK_LIST]", actions_nl)
    if open_ended:
        prompt = """Given the image sequence, what task did the robot perform? The robot only performed one task.
Output the task the robot performed. Output three paraphrases of the task, delimited by comma. The task should include the interacted object and the object's relation to other objects"""
        prompt = """Given the image sequence, what task did the robot perform? Output the task the task the robot performed as a short instruction with three paraphrases, delimited by comma. The task should include the interacted object and the object's resulting relation to other objects.
Example tasks: "Place the pot to the left of the fruit", "Move the dishrag to the bottom of the table next to the towel", "Put the pot to the right of the fruit", "Turn on stove", "Open the microwave", "Put the knife inside the sink".
"""
        prompt = """Given the image sequence, what task did the robot perform? Output the task the task the robot performed as a short instruction with three paraphrases, delimited by comma. The task should include the interacted object and the object's resulting relation to other objects.
Example tasks: "Place the pot to the left of the fork", "Move the dishrag to the bottom of the table next to the blue towel", "Pick up the cucumber and place it to the right of the bowl", "Turn on stove", "Open the microwave", "Put the knife inside the sink".

Output valid json in the following format:
{,
"tasks": "the predicted tasks. The tasks should always include the final object positions if possible."}
"""

    prompt_our = """Given the image sequence, what task did the robot perform?
Follow these guidelines to determine the task:
Step 1: Determine the objects that appear in the images. List all objects.
Step 2: Determine the object movement and the resulting object relations. Think about where the object and its relational objects are located in the scene on a global scale. Think step by step and list the locations and relations of all objects. Explain the object movements.
Step 3: Determine what tasks result in the object relations from Step 2.
Step 4: Output tasks that that accomplish the observations as short instructions. Focus on simple, single-step tasks that only require interaction with the determined object from Step 1. Focus on tasks that include changing the object relation and moving the object.
Example tasks: "Place the pot to the left of the fruit", "Move the dishrag to the bottom of the table next to the towel", "Put the pot to the right of the fruit", "Turn on stove", "Open the microwave", "Put the knife inside the sink".
Follow the steps above. Explain your reasoning. Output the reasoning delimited by ***.
After, produce your output as JSON. The format should be:
```{
"tasks": "The determined tasks, delimited by commas. Output 3 different task instructions.",
"confidence": "A confidence score for each task between 0 and 10, delimited by commas. Be pessimistic."
}```"""

    # prompt = prompt.replace("[OBSERVATIONS]", observations)
    # return prompt_our
    return prompt




