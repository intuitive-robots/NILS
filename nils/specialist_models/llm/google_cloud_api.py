import json
import logging
import re

import vertexai
from vertexai import generative_models
from vertexai.generative_models import GenerationConfig, GenerativeModel
from vertexai.language_models import (
    TextEmbeddingInput,
    TextEmbeddingModel,
    TextGenerationModel,
)

from nils.annotator.prompts import (
    OBJECT_EXTRACTION_PROMPT,
    OBJECT_PROPERTIES_PROMPT,
    STATE_DETECTION_PROMPT,
)
from nils.specialist_models.llm.parse_utils import (
    parse_json_markdown,
)
from nils.utils.utils import retry_on_exception

vertexai.init(project="your_project", location="us-central1")
parameters = {
    "max_output_tokens": 80,
    "temperature": 0.1,
    "top_p": 1
}

from joblib import Parallel, delayed
from tqdm import tqdm



def get_objects_from_annotation(annotation):
    prompt="""You will be provided with a task instruction. Based on the instruction, list all objects that are mentioned in the instruction. Remove specific names. Output the objects delimited by commas.
    Examples:
    Task: Place the bag of chips in the bottom drawer
    Objects: bag of chips, bottom drawer
    Task: Knock over the can
    Objects: can
    Task:Pick up the can from the middle drawer
    Objects: can, middle drawer
    Task: Move the RXchocolate bar to the left on the counter
    Objects: chocolate bar, counter
    Task:"""

    prompt = prompt + annotation

    response = query_parallel(prompt)

    try:
        response_objects = response.split("Objects:")[1].strip().split(",")
    except Exception as e:
        logging.error(f"Error parsing LLM response: {e}")
        response_objects = None


    return response_objects

def get_aggregation_prompt(task_list):
    prompt = """You will be provided with a list of tasks performed by a robot delimited by triple quotes. Aggregate these tasks to a single, high level task. Output 3 different, diverse task instructions, delimited by ;
        Examples:
        Tasks:
        '''Task 1: Move the gripper to the left towards the fruit; Task 2: Move the gripper backwards and behind to the fruit; Task 3: Move the fruit to the top left, next to the bowl; Task 4: Move the gripper towards the sink;'''
        Instructions:
        Pick up the fruit and place it near the bowl; Move the arm towards the fruit, pick it up, place it near the bowl and move the arm towards the sink; Move the gripper to the left towards the fruit;
        Tasks:
        '''Task 1: Move the pot lid to the left of the stovetop; Task 2: Place the mushroom inside the pot; Task 3: Move the pot to the right of the stovetop; Task 4: Pick the mushroom and place it next to the sink;'''
        Instructions:
        Open the lid, and boil the mushroom; Place the mushroom inside the pot and move the pot to the right of the stovetop; Move the pot to the right of the stovetop; Put the mushroom in the pot for cooking;
        Tasks:
        '''[TASK_LIST]'''
        Instructions:
    """

    task_list_string = ""
    for i in range(len(task_list)):
        task_list_string += f"Task {i + 1}: {task_list[i]}; "
    prompt = prompt.replace("[TASK_LIST]", task_list_string)
    
    
    return prompt

def aggregate_tasks(prompts):
    
    
    results = Parallel(n_jobs=-1,backend='threading')(
        delayed(query_parallel)(prompt)
        for idx, prompt in enumerate(tqdm(prompts)))

    llm_outputs = [result for result in results]
    
    llm_outputs_splt = [output.split(";") for output in llm_outputs]
    
    
    return llm_outputs_splt
    
@retry_on_exception(Exception, retries=3)
def query_parallel(prompt,candidate_count=1, temperature = 0,model_id="gemini-1.0-pro-002"):
    
    model = GenerativeModel(model_id)
    safety_config = [
        generative_models.SafetySetting(
            category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        generative_models.SafetySetting(
            category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        generative_models.SafetySetting(
            category=generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        generative_models.SafetySetting(
            category=generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ]
    
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=1.0,
        max_output_tokens=800,
        candidate_count=candidate_count
    )
    

    response = model.generate_content(prompt, generation_config=generation_config,
                                      safety_settings=safety_config)

                                
        

    text = response.text.strip()


    return text


def get_object_task_dicts_parallel(all_objects,tasks = None,task_centric = True):
    if task_centric:
        assert tasks is not None
    prompts = []

    for objects,task in zip(all_objects,tasks):
        task_list = ["-" + t for t in [task]]
        task_list = "\n".join(task_list)

        objects = [obj.split(",")[0] for obj in objects]

        objects = ",".join(objects)

        if task_centric:
            prompt = OBJECT_EXTRACTION_PROMPT.replace("[TASK_LIST]", task_list)
        else:
            prompt = OBJECT_PROPERTIES_PROMPT.replace("[OBJECT_LIST]", objects)

        prompts.append(prompt)

    results = Parallel(n_jobs=-1,backend='threading')(
        delayed(query_parallel)(prompt)
        for idx, prompt in enumerate(tqdm(prompts)))

    results_json = []

    for result in results:
        try:
            response_dict = parse_json_markdown(result)
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            response_dict = None
        results_json.append(response_dict)



    return results_json




def get_objects_from_annotation_parallel(tasks):


    prompt="""You will be provided with a task instruction. Based on the instruction, list all objects that are mentioned in the instruction. Remove specific names. Output the objects delimited by commas.
Task: Place the bag of chips in the bottom drawer
Objects: bag of chips, bottom drawer
Task: Knock over the can
Objects: can
Task:Pick up the can from the middle drawer
Objects: can, middle drawer
Task: Move the RXchocolate bar to the left on the counter
Objects: chocolate bar, counter
Task:"""

    prompts = [prompt + task  + "\nObjects:" for task in tasks]

    responses = Parallel(n_jobs=-1,backend='threading')(
        delayed(query_parallel)(prompt)
        for idx, prompt in enumerate(tqdm(prompts)))


    objects = []
    for response in responses:
        try:
            response_objects = response.strip().split(",")
            response_objects = [obj.strip() for obj in response_objects]
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            response_objects = None

        objects.append(response_objects)
    return  objects


def get_tasks_parallel(prompts):


    

    results = Parallel(n_jobs=-1,backend='threading')(
        delayed(query_parallel)(prompt)
        for idx, prompt in enumerate(tqdm(prompts)))

    llm_outputs = [result for result in results]
    
    
   
    all_tasks = []
    all_candidates = []
    all_scores = []
    
    for response in llm_outputs:
        try:
            if response is None:
                all_tasks.append([])
                all_candidates.append([])
                all_scores.append([])
                continue

            response_cleaned = response.replace("\n", "")
            response_cleaned = response_cleaned.replace("json", "")
            produced_json = re.search(r"```({.*})```", response_cleaned, re.DOTALL)
            candidates = []
            if produced_json:
                response_dict = json.loads(produced_json.group(1))
                if  response_dict["tasks"] is None:
                    tasks = []
                    scores = []
                else:
                    tasks = response_dict["tasks"].split(";")
                    tasks = [task.strip() for task in tasks]
                    scores = response_dict["confidence"].split(",")
                if "task candidates" in response_dict.keys():
                    candidates = response_dict["task candidates"].split(",")
                else:
                    candidates = []
        
                # candidates = response_dict["task candidates"].split(",")
            
                
                scores_cleaned = []
                
                for score in scores:
                    try:
                        scores_cleaned.append(float(score))
                    except:
                        scores_cleaned.append(0)
                
                scores = scores_cleaned
            else:
                produced_json = re.search(r"({.*})", response_cleaned, re.DOTALL)
                if produced_json:
                    response_dict = json.loads(produced_json.group(1))
                    tasks = response_dict["tasks"].split(";")
                    tasks = [task.strip() for task in tasks]
                    if "task candidates" in response_dict.keys():
                        candidates = response_dict["task candidates"].split(",")
                    else:
                        candidates = []
        
                    # candidates = response_dict["task candidates"].split(",")
                    scores = response_dict["confidence"].split(",")
                    scores = [float(score) for score in scores]
                else:
                    tasks = []
                    candidates = []
                    scores = []
            tasks = [task for task in tasks if task != ""]        

        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            tasks = []
            candidates = []
            scores = []
                
        
        all_tasks.append(tasks)
        
        
        
        all_candidates.append(candidates)
        all_scores.append(scores)
        
    
    return all_tasks, all_candidates, all_scores


class VertexAIAPI:
    def __init__(self):
        self.model = GenerativeModel("gemini-1.0-pro-002")
        self.model_states = TextGenerationModel.from_pretrained("text-bison@002")
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-preview-0409")
        
        
        
    def get_text_embeddings(self,texts,task):
        
        inputs = [TextEmbeddingInput(text,task) for text in texts]
        
        embeddings = self.embedding_model.get_embeddings(inputs)
        
        return [embedding.values for embedding in embeddings]
        
    def query(self, prompt, sampling_params=None, save=False, save_dir="", model_type="base",candidate_count=1, temperature = 0)   :
        # parameters = {
        #     "max_output_tokens": sampling_params["max_output_tokens"],
        #     "temperature": sampling_params["temperature"],
        #     "top_p": sampling_params["top_p"]
        # }
        parameters = {
            "max_output_tokens": 500,
            "temperature": 0.1,
            "top_p": 1
        }

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=1.0,
            max_output_tokens=800,
            candidate_count = candidate_count
        )
        safety_config = {
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }

        # prompt_with_in_context = CALVIN_IN_CONTEXT_PROMPTS + prompt["observations"] + "The robot performed the task:"

        # gemini_prefix = "Multi-choice problem: What task did the robot perform? Choose a task from the list?\n" + prompt["task_list"]
        # if append_prefix:
        #     
        #     prompt_with_in_context =  gemini_prefix +"\n" + "Observations: " + prompt["observations"] + "The robot performed the task:"
        # else:
        #     prompt_with_in_context = prompt["simple"]

        if model_type == "state_model":
            response = self.model_states.predict(prompt, temperature=0,
                                                 top_p=1.0,
                                                 max_output_tokens=60)
        else:
            response = self.model.generate_content(prompt, generation_config=generation_config,
                                                   )
        try:
            text = response.text.strip()
        except Exception:
            return ["Error in response from LLM"]

        return text

    def get_task(self, prompt, sampling_params=None, save=False, save_dir=""):
        # n_candidates = 3
        # repsonses = []
        # for i in range(n_candidates):
        #
        #     response = self.query(prompt, model_type="base",candidate_count=1, temperature = 0.4)
        #     repsonses.append(response)
        response = self.query(prompt, model_type="base", candidate_count=1, temperature=0.1)

        response_cleaned = response.replace("\n", "")
        response_cleaned = response_cleaned.replace("json", "")
        produced_json = re.search(r"```({.*})```", response_cleaned, re.DOTALL)
        candidates = []
        if produced_json:
            response_dict = json.loads(produced_json.group(1))
            tasks = response_dict["tasks"].split(";")
            tasks = [task.strip() for task in tasks]
            if "task candidates" in response_dict.keys():
                candidates = response_dict["task candidates"].split(",")
            else:
                candidates = []

            
            #candidates = response_dict["task candidates"].split(",")
            scores = response_dict["confidence"].split(",")
            scores = [float(score) for score in scores]
        else:
            produced_json = re.search(r"({.*})", response_cleaned, re.DOTALL)
            if produced_json:
                response_dict = json.loads(produced_json.group(1))
                tasks = response_dict["tasks"].split(";")
                tasks = [task.strip() for task in tasks]
                if "task candidates" in response_dict.keys():
                    candidates = response_dict["task candidates"].split(",")
                else:
                    candidates = []
                    
                #candidates = response_dict["task candidates"].split(",")
                scores = response_dict["confidence"].split(",")
                scores = [float(score) for score in scores]
            else:
                tasks = []
                candidates = []
                scores = []
    
        return tasks,candidates, scores
    def get_nl_prompt(self, prompt_dict):
        gemini_prefix = "Multi-choice problem: What task did the robot perform? Choose a task from the list?\n" + \
                        prompt_dict["task_list"]
        prompt_with_in_context = gemini_prefix + "\nObservations: " + prompt_dict[
            "observations"] + "\nThe robot performed the task:"

        return prompt_with_in_context

    def get_nl_prompt_keystate_reasons(self, task_list, observations,open_ended=False,aggregated = False, n_paraphrases = 3):

        prompt_system = '''You will be provided with observations of a robot interaction with an environment, delimited by triple quotes.
Select a task from this list that best describes the robots actions:
```
[TASK_LIST]
```
Follow these guidelines:
Step 1: Determine the object the robot interacted with and then determine tasks that include that object.
Output the possible tasks after this step delimited by commas.
Step 2: Determine the object movement and the resulting object relations. Think about where the object and its relational objects are located in the scene on a global scale. Think step by step and list the locations and relations of all objects. Pay special attention to the object relations from Step 1.
Step 3: Determine what tasks result in the object relations from Step 2. Explain why the task accomplishes the object relations. If the task is not clear, output None. If multiple tasks are possible, output multiple tasks with a low score.
Step 4: Some tasks do not have specific object relations, but instead require moving objects in some direction. Also consider these tasks by examining the object movements.

Follow the steps above. Explain your reasoning. Output the reasoning delimited by ***.

After, produce your output as JSON. The format should be:
```{
"task candidates": "Possible tasks from the list after Step 1, delimited by commas.",
"tasks": "The tasks that can be considered valid, delimited by comma. Make sure to output all tasks that match the description. Output up to 2 tasks.",
"confidence": "A confidence score for each task between 0 and 10, delimited by commas. Be pessimistic."
}```
'''
        
        
        open_ended_prompt = '''You will be provided with observations of a robot interaction with an environment, delimited by triple quotes.
Determine the task the robot could have solved. The robot can only solve one task. If the observations indicate that the robot interacted with multiple objects, focus on the most frequent and precise observations.

Follow these guidelines:
Step 1: Answer what objects appear in the observation. List all objects. Then, determine the object for which the observations align the best.
Step 2: Determine the object movement and the resulting object relations. Think about where the object and its relational objects are located in the scene on a global scale. Think step by step and list the locations and relations of all objects. Explain the object movements.
Step 3: Determine what tasks result in the object relations from Step 2.
Step 4: Output tasks that that accomplish the observations as short instructions. Focus on simple, single-step tasks that only require interaction with the determined object from Step 1. Focus on tasks that include changing the object relation and moving the object.
Example tasks: "Place the pot to the left of the fruit"; Slide the dishrag to the bottom of the table next to the towel"; Pick up the spoon and place it  at the bottom left of the table;"Put the lid on the pot to close it";"Move the eggplant to the left and forward away from the other objects";"Put the pot to the right of the fruit"; "Move the pot forward and to the left"; "Turn on stove"; "Open the microwave"; "Relocate the knife inside the sink".
Follow the steps above. Explain your reasoning. Output the reasoning delimited by ***.
After, produce your output as JSON. The format should be:
```{
"tasks": "The determined tasks, delimited by semicolons. Output [N_PARAPHRASES] different,diverse task instructions. The instructions should cover all observations and each include different observations. Example: Place the pot to the left of the fruit; Move the pot backward and to the right; Relocate the pot at the left of the table to the center of the table; Lift up the pot and place it next to the spoon;",
"confidence": "A confidence score for each task between 0 and 10, delimited by commas. Be pessimistic."
}```
'''
        open_ended_prompt = open_ended_prompt.replace("[N_PARAPHRASES]", str(n_paraphrases))

        open_ended_prompt_aggregated = '''You will be provided with observations of a robot interaction with an environment, delimited by triple quotes. The robot solved multiple tasks in a row.
Determine the high-level task the robot could have solved.
Follow these guidelines:
Step 1: Answer what objects appear in the observation. List all objects.
Step 2: Determine the objects movement and the resulting object relations. Think about where the object and its relational objects are located in the scene on a global scale. Think step by step and list the locations and relations of all objects. Explain the object movements.
Step 3: Determine what tasks result in the object relations from Step 2.
Step 4: Output tasks that that accomplish the observations as short instructions.
Example tasks: "Prepare the table for eating";"Slide the dishrag to the bottom of the table next to the towel"; Pick up the spoon and place it at the bottom left of the table; "Move the pot forward and to the left"; "Turn on stove"; "Open the microwave"; "Relocate the knife inside the sink"; "Prepare to cook the food"; "Make a coffee".
Follow the steps above. Explain your reasoning. Output the reasoning delimited by ***.
After, produce your output as JSON. The format should be:
```{
"tasks": "The determined high-level task instruction, which summarizes all sub-tasks. Output 2 different task instructions delimited by ;."
"confidence": "A confidence score for each task between 0 and 10, delimited by commas. Be pessimistic."
}```
        '''
        prompt_system = prompt_system.replace("[TASK_LIST]", task_list)
        
        
        if open_ended:
            if aggregated:
                prompt_system = open_ended_prompt_aggregated
            else:
                prompt_system = open_ended_prompt


        prompt_user = "Observations: ```" + observations + "```"

        prompt = prompt_system + prompt_user
        prompt+= "\n***Step 1:"

        return prompt

    def get_new_states(self, pred_action, init_states):

        prompt = self.get_new_state_prompt(pred_action, init_states)

        response = self.query(prompt, model_type="state_model")

        response_dict = json.loads(response)
        # response_dict = response_json["result"]

        return response_dict

    def get_new_state_prompt(self, pred_action, init_states):

        states_json = json.dumps(init_states)
        prompt = STATE_DETECTION_PROMPT.replace("[TASK]", pred_action).replace("[OBJECT_STATES]", states_json)

        return prompt
    
    def get_possible_actions(self, objects):
        
        example_tasks = ["Place the tin can to the left of the pot.",
                         "Move the dishrag to the bottom of the table next to the towel"
            , "Put the pot to the right of the fruit", "Turn on stove", "Open the microwave"]
        example_tasks = ",".join(example_tasks)
        
        prompt = """You will be provided with a list of objects observed by a robot, delimited by triple quotes. Based on the objects, give possible instructions to the robot. Infer the type of environment from the provided objects. Ignore objects that do not fit the environment.
Follow these guidelines:
- Keep the instructions simple. Focus on tasks that only require a single step.
- Include instructions like placing an object inside another object or moving the object. Only for movable objects.
- Dont assume the presence of any objects not listed. 
- Output a complete list of instructions with all object relations, including every object at least 4 times.
- Output unique tasks only.
Output at least 25 possible instructions delimited by comma. Do not use any numbers.
Here are a few examples:
[EXAMPLE_TASKS]

Objects:
```[OBJECTS]```

Instructions List:

"""
        objects_nl = ", ".join(objects)
        prompt = prompt.replace("[OBJECTS]", objects_nl)
        prompt = prompt.replace("[EXAMPLE_TASKS]", example_tasks)
        
        response = self.query(prompt, model_type="base", candidate_count=1, temperature=0.4)
        
        try:
            parsed = response.split(",")
            if len(parsed) == 1:
                parsed = response.split("\n")
            parsed = [p.strip() for p in parsed]



        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            parsed = None
        return parsed

        

    def get_object_task_list(self, tasks, objects, task_centric):

        task_list = ["-" + task for task in tasks]
        task_list = "\n".join(task_list)
        
        objects = [obj.split(",")[0] for obj in objects]
        
        objects = ",".join(objects)
        
        if task_centric:
            prompt = OBJECT_EXTRACTION_PROMPT.replace("[TASK_LIST]", task_list)
        else:
            prompt = OBJECT_PROPERTIES_PROMPT.replace("[OBJECT_LIST]", objects)

        prompt = prompt

        llm_response = self.query(prompt)
        
        response_dict = parse_json_markdown(llm_response)
        
        if response_dict is None:
            raise ValueError("Error in parsing LLM response")
        else:
            return response_dict
        
    
        return response_dict


def get_prompt_simple(observations):


    open_ended_prompt = '''You will be provided with observations of a robot interaction with an environment, delimited by triple quotes.
Determine the task the robot solved. The robot can only solve one task. If the observations indicate that the robot interacted with multiple objects, focus on the most frequent and precise observations.
Focus on the resulting object relations and the object movements.

Example tasks: "Place the pot to the left of the fruit"; Slide the dishrag to the bottom of the table next to the towel"; Pick up the spoon and place it  at the bottom left of the table;"Put the lid on the pot to close it";"Move the eggplant to the left and forward away from the other objects";"Put the pot to the right of the fruit"; "Move the pot forward and to the left"; "Turn on stove"; "Open the microwave"; "Relocate the knife inside the sink".

Produce your output as JSON. The format should be:
```{
"tasks": "The determined tasks, delimited by semicolons. Output 3 different task instructions. The instructions should cover all observations and each include different observations.",
"confidence": "A confidence score for each task between 0 and 10, delimited by commas. Precise descriptions should have a higher score."
}```
'''


    prompt_system = open_ended_prompt

    prompt_user = "Observations: ```" + observations + "```"

    prompt = prompt_system + prompt_user


    return prompt
