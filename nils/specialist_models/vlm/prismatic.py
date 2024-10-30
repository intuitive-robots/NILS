import io
import json
import logging
import re

import numpy as np
import torch
from PIL import Image
from prismatic import load

from nils.utils.plot import crop_images_with_boxes

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

class PrismaticVLM:
    def __init__(self,model_path = "prism-dinosiglip+13b"):

        model_id = "prism-dinosiglip+7b"
        
        
        #self.model = GenerativeModel("gemini-1.5-pro-preview-0409")
        self.model = load(model_id,hf_token = "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")
        self.model = self.model.to(device,dtype = torch.bfloat16)
        

    
    
    
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
        preprocessed_image = ImageGCloud.from_bytes(img_byte_arr)

        
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
        prompt = """Create a json list where each entry is an object in the image. The entry should have the keys "name" with a unique, concise, up to six-word description of the object and "color", containing the object color. Output the surface the objects are placed on as the first entry. An example:
```
[{"name": pot with handle,
"color": "silver"},
{"name": spoon,
"color": "blue"}
]
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


        if som:
            prompt = prompt_som
        image_pil = Image.fromarray(np.array(image))
        
        longest_side = max(image_pil.size)
        max_side = 512
        scale = max_side / longest_side
        
        image_pil = image_pil.resize((int(image_pil.width * scale), int(image_pil.height * scale))
                                    )

        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        
      
        success = False
        current_try = 0
        n_retries = 2
        while not success and current_try < n_retries:
            response = self.model.generate(
                image_pil,
                prompt_text,
                do_sample=True,
                temperature=0.4,
                max_new_tokens=350,
                min_length=1,
            )
                
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
                
                success = True
                return json_extracted
            
            except Exception as e:
                logging.error(f"Error parsing LLM response: {e}")
                current_try += 1
                #return None

        return []
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
    