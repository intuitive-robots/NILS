import base64
import io
import json
import logging
import re
import time

import numpy as np
from joblib import Parallel, delayed
from openai import OpenAI
from PIL import Image

from nils.specialist_models.llm.parse_utils import parse_json
from nils.utils.plot import crop_images_with_boxes

api_key = "your_key"


class GPT4V:
    def __init__(self):
        api_key = "your_key"
        anyscale_key = "your_key"
        # self.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.model_name = "gpt-4-0125-preview"
        self.model_name = "gpt-3.5-turbo"
        self.gpt3 = "gpt-3.5-turbo"
        self.gpt4 = "gpt-4-turbo"
        if self.model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            self.client = OpenAI(api_key=anyscale_key, base_url="https://api.endpoints.anyscale.com/v1")
        else:
            self.client = OpenAI(api_key=api_key)

       

    
    
    
    def get_object_states(self,image,obj,possible_states,crop_box = None):
        # parameters = {
        #     "max_output_tokens": sampling_params["max_output_tokens"],
        #     "temperature": sampling_params["temperature"],
        #     "top_p": sampling_params["top_p"]
        # }
        parameters = {
            "max_output_tokens": 80,
            "temperature": 0.0,
            "top_p": 1
        }
        
       
        if crop_box is not None:
            cropped_image  = crop_images_with_boxes(image, crop_box)
        else:
            cropped_image = image
            
        prompt = self.get_prompt(obj,possible_states)
        
        cropped_image_pil = Image.fromarray(np.array(cropped_image))
        longest_side = max(cropped_image_pil.size)
        max_side = 400
        scale = max_side / longest_side
        
        img_byte_arr = io.BytesIO()
        cropped_image_pil.resize((int(cropped_image_pil.width*scale), int(cropped_image_pil.height*scale))).save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
      
        
        generation_config = GenerationConfig(
            temperature=0.2,
            top_p=1.0,
            top_k=32,
            max_output_tokens=60,
        )
        
        
        
        

        response = self.model.generate_content([preprocessed_image, prompt], generation_config=generation_config)
        text = response.text.strip().lower()
        
        if "unknown" in text:
            return "unknown"
        print(text)
        
        matched_states = []
        for state in possible_states:
            if state in text:
                matched_states.append(state)
        if len(matched_states) == 0 or len(matched_states)>1:
            
            logging.error(f"Error getting state for {obj}. Text: {text}. Matched: {matched_states}")
            return None
        
        return matched_states[0]

        #return text
    


    def get_objects_in_image(self,image,som = False,already_detected_objects = None,single_prompt = False):

        prompt = """What objects are in the image? If there are multiple similar objects distinguish them by color. Output all objects as a json list with the following properties:
```
{"name": "the name of the object",
"movable": "whether the object is movable or stationary",
"container": "whether the object can contain other objects",
"states": "possible states the object can usually be in. can be null".
} 
```
"""        
        prompt = """Create a json list where each entry is an object in the image. The objects are labeled with numbers. The entry should have the keys "name" with a unique, concise, up to six-word description of the object and "color", containing the object color. Output the surface the objects are placed on as the first entry. An example:
```
[{"name": pot with handle,
"color": "silver"}]
```
"""
        
        prompt_som = """What objects are in the image? The objects are labeled with numbers. Output the surface the objects are placed on as number 0. Output all objects as a json list with the following format:
```
[{"id": "the number of the object as indicated in the image",
"name": the unique name of the object, as specific as possible. If multiple objects are similar, distinguish the.",
"color": "the color of the object"}]
```
"""
        prompt_som = """Create a json list where each entry is an object in the image. The objects are labeled with numbers. The entry should have the keys "id" representing the id of the object as shown in the image, "name" with a concise, up to six-word description of the object and "color", containing the object color. Output the surface the objects are placed on as number 0. An example:
```
[{"id": "1",
"name": pot with handle,
"color": "silver"},
{"id": "2",
"name": spoon,
"color": "blue"}
]
```
"""
        prompt_som_simple = """What is the object labeled with a white number 1? Be as specific as possible. Output JSON with the following format:
```
[{"id": "the number of the object",
"name": the name of the object, as specific as possible and output at least two words.",
"color": "the color of the object"}]
}
"""     
        
        if single_prompt:
            prompt_som = prompt_som_simple
        
        if already_detected_objects is not None:
            det_objects_str= ",".join(already_detected_objects)
            prompt_som += "The following objects are already detected: " + det_objects_str + "." + "Use the name of these objects if you detect them. Only output objects labeled with numbers."

        # ```
        # {"name": "the name of the object",
        # "number": "the number of the object",
        # "movable": "whether the object is movable or stationary",
        # "container": "whether the object can contain other objects",
        # "states": "possible states the object can usually be in. can be null".
        # } 
        # ```
        
        if som:
            prompt = prompt_som
        image_pil = Image.fromarray(np.array(image))
        
        longest_side = max(image_pil.size)
        max_side = 512
        scale = max_side / longest_side

        img_byte_arr = io.BytesIO()
        image_pil.resize((int(image_pil.width * scale), int(image_pil.height * scale))).save(
            img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        img_str = base64.b64encode(img_byte_arr).decode('utf-8')
        
        
        img_url = f"data:image/jpeg;base64,{img_str}"

        messages = [{"role": "user", "content": [

            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": img_url
                }
            }
            
        ]
            }]
        

        n_retries = 2
        cur_tries = 0
        success = False
        while cur_tries < n_retries and not success:
            try:
                response = self.client.chat.completions.create(
                    messages=messages,

                    model=self.gpt4,
                    logprobs=True,
                    # top_logprobs=2,
                    temperature=0.3
                    ,
                    max_tokens=512,
                    # top_p=0.9,
                )
                response = response.choices[0].message.content

                success = True
            except Exception as e:
                time.sleep(30)
                logging.error(f"Error generating content: {e}")
                cur_tries += 1

        text = response.strip().lower()
        
        logging.info(f"VLM Object Retrieval Response: {text}")


        text = text.replace("json","")
        text = text.replace("JSON","")
        text = text.replace("\n","")
        json_string = re.search(r"```({.*})```", text, re.DOTALL)
        if json_string is None:
            json_string = re.search(r"(\[.*\])", text, re.DOTALL)

        if json_string is None:
            json_string = text
        try:
            json_extracted = json.loads(json_string.group(1))

            return json_extracted
        
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            return None


    def get_prompt(self, obj,possible_states):
        formatted_states = " or ".join(possible_states)
        #formatted_states = "[" + formatted_states + "]"
        
        
        
        
        #prompt = f"What is the state of the {obj}? Choose one: {formatted_states}"
        #prompt = f"Is the {obj} {formatted_states}? Respond unknown if you are not sure. Answer with one word only."
        prompt = f"What is the state of the {obj}? {formatted_states}?"

        return prompt
    
    def get_nl_prompt(self, prompt_dict):
        gemini_prefix = "Multi-choice problem: What task did the robot perform? Choose a task from the list?\n" + \
                        prompt_dict["task_list"]
        prompt_with_in_context = gemini_prefix + "\nObservations: " + prompt_dict[
            "observations"] + "\nThe robot performed the task:"

        return prompt_with_in_context
    

    def get_action_prediction_prompt(self, possible_actions,interacted_object):
        actions_nl = ["- " + action for action in possible_actions]
        actions_nl = "\n".join(actions_nl)
        prompt = """Look at the provided Initial Observation Image and Final Observation Image. The Final Observation Image shows a scene after the robot interacted with it. The box annotations help you to identify the objects.
Select tasks from this list that best describe the robots actions:
```
[TASK_LIST]
```
Follow these guidelines:
Step 1: Determine the object the robot interacted with and then determine tasks that include that object.
Output the possible tasks after this step delimited by commas.
Step 2: Determine the object movement and the resulting object relations. Think about where the object and its relational objects are located in the scene on a global scale. Think step by step and list the locations and relations of all objects. Pay special attention to the object relations from Step 1.
Step 3: Determine what tasks result in the object relations from Step 3. Not all object relations have to be present in the observations. If one resulting object relation from a task matches the observations, it can be considered valid. Incorporate this uncertainty in a confidence score. It is better to output more tasks.
Produce your output as JSON. The format should be:
```{
"reasons": "The reasons for the prediction, based on the Steps 1-4",
"synonyms": "The synonyms found in Step 1",
"task candidates": "Possible tasks from the list after Step 2",
"object relations": "Object relations determined in Step 3",
"task": "The predicted task",
"confidence": "A confidence score between 0 and 10"
}```
[OBJECT]

JSON:
"""
        prompt = prompt.replace("[TASK_LIST]", actions_nl)
        prompt = prompt.replace("[OBJECT]", interacted_object)



        return prompt
    
    def get_action(self,images, possible_actions, interacted_object,use_som = True,observations = None,open_ended = False):

        interacted_object = interacted_object.replace("and it moved","")

        generation_config = GenerationConfig(
            temperature=0.2,
            top_p=1.0,
            top_k=32,
            max_output_tokens=1024,
        )
        prompt_nl = get_action_prediction_prompt_baseline(possible_actions,interacted_object,use_som,observations,open_ended)
        
        n_images = 4
        
        image_indices = np.linspace(0, len(images) - 1, n_images, dtype=int)
        
        images_prompt =  images[image_indices]
        
        image_processed = []
        
        for image in images_prompt:
            image_pil = Image.fromarray(np.array(image))
            longest_side = max(image_pil.size)
            max_side = 400
            scale = max_side / longest_side

            img_byte_arr = io.BytesIO()
            image_pil.resize(
                (int(image_pil.width * scale), int(image_pil.height * scale))).save(img_byte_arr,format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            preprocessed_image = ImageGCloud.from_bytes(img_byte_arr)
            image_processed.append(preprocessed_image)
        if n_images == 2:
            response = self.model.generate_content(["Initial Observation Image:\n"] + [image_processed[0]] + ["\nFinal Observation Image:"]  + [image_processed[1]] + [prompt_nl], generation_config=generation_config)
            response_text = response.text.strip()
            response_text = response_text.replace("JSON","")
            response_text = response_text.replace("json","")
            response_text = response_text.replace("\n", "")

            produced_json = re.search(r"```({.*})```", response_text, re.DOTALL)
            if produced_json is None:
                produced_json = re.search(r"```.*({.*})```", response_text, re.DOTALL)

            if not produced_json:
                try:
                    json_extracted = json.loads(response)
                    action_candidates = json_extracted["task candidates"].split(",")
                    task = json_extracted["task"]
                    confidence  = json_extracted["confidence"]



                except Exception:
                    logging.error(f"Error parsing LLM response: {response_text}")
            elif produced_json:
                json_extracted = produced_json.group(1)
                try:
                    json_parsed = json.loads(json_extracted)
                    action_candidates = json_parsed["task candidates"].split(",")
                    task = json_parsed["task"]
                    confidence  = json_parsed["confidence"]

                except Exception as e:
                    logging.error(f"Error parsing LLM response: {e}")



        else:
            response = self.model.generate_content(image_processed + [prompt_nl], generation_config=generation_config)

            response_text = response.text.strip()
            
            response_dict = parse_json(response_text,expected_keys=["tasks"])

            confidence = 1
            if response_dict is None:
                response_text = response_text.split(",")
                if len(response_text) == 1:
                    response_text = response.text.strip().split("\n")
                tasks  = response_text
                confidence = 1
            
            else:
                tasks = response_dict["tasks"].split(",")
            

        return tasks,confidence
        
      

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
        prompt = """Given the image sequence, what task did the robot perform? SThe robot only performed one task.
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
    #return prompt_our
    return prompt





def get_grounding_predictions_baseline_vlm_batched_list(images, possible_actions, open_ended=True):
    

    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(query_grounding_parallel)(image_subset, possible_actions, open_ended) for
        image_subset in images)
    
    llm_responses = [result[0] for result in results]
    
    scores = [result[1] for result in results]

    return llm_responses, scores



def get_grounding_predictions_baseline_vlm_batched(keystates, batch, possible_actions, open_ended=True):
    images = []
    for keystate_idx, ks in enumerate(keystates):
        ks = min(ks, len(batch["rgb_static"]) - 1)
        if keystate_idx == 0:
            last_keystate_idx = 0
        if ks < 4:
            last_keystate_idx = ks
            continue
        length = ks - last_keystate_idx

        length = min(8, length)
        start_idx = max(0, ks - length)

        image_final_index = min(ks, len(batch["rgb_static"]) - 1)
        images_keystate = batch["rgb_static"][start_idx:image_final_index]
        images.append(images_keystate)

    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(query_grounding_parallel)(image_subset, possible_actions, open_ended) for
        image_subset in images)

    llm_responses = [result[0] for result in results]
    scores = [result[1] for result in results]

    return llm_responses, scores


def query_grounding_parallel(images, possible_actions, open_ended):
    
    
    client = OpenAI(api_key=api_key)
    
    use_som = False


    prompt_nl = get_action_prediction_prompt_baseline(possible_actions, possible_actions, use_som, [],
                                                      open_ended)

    n_images = 4

    image_indices = np.linspace(0, len(images) - 1, n_images, dtype=int)

    images_prompt = images[image_indices]

    image_processed = []
    image_urls = []
    for image in images_prompt:
        image_pil = Image.fromarray(np.array(image))
        
        longest_side = max(image_pil.size)
        max_side = 512
        scale = max_side / longest_side

        img_byte_arr = io.BytesIO()
        image_pil.resize((int(image_pil.width * scale), int(image_pil.height * scale))).save(
            img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        img_str = base64.b64encode(img_byte_arr).decode('utf-8')

        img_url = f"data:image/jpeg;base64,{img_str}"
        image_urls.append(img_url)
        image_processed.append(img_str)

    messages = [
        {
            "role": "user",
            "content": [
                prompt_nl,
                *map(lambda x: {"image": x, "resize": 512}, image_processed),
            ],
        },
    ]
    
    # messages = [{"role": "user", "content": [
    # 
    #     {
    #         "type": "text",
    #         "text": prompt
    #     },
    #     {
    #         "type": "image_url",
    #         "image_url": {
    #             "url": img_url
    #         }
    #     }
    # 
    # ]
    #              }]

    n_retries = 2
    cur_tries = 0
    success = False
    while cur_tries < n_retries and not success:
        try:
            response = client.chat.completions.create(
                messages=messages,

                model="gpt-4o-mini",
                logprobs=True,
                # top_logprobs=2,
                temperature=0.2
                ,
                max_tokens=512,
                # top_p=0.9,
            )
            response = response.choices[0].message.content

            success = True
        except Exception as e:
            time.sleep(30)
            logging.error(f"Error generating content: {e}")
            cur_tries += 1
        
    

    response_text = response

    response_dict = parse_json(response_text, expected_keys=["tasks"])

    confidence = 1
    if response_dict is None:
        response_text = response_text.split(",")
        if len(response_text) == 1:
            response_text = response.text.strip().split("\n")
        tasks = response_text
        confidence = 1

    else:
        tasks = response_dict["tasks"].split(",")

    return tasks, confidence
