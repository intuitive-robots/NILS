import json
import logging
import re

import numpy as np
from joblib import Parallel, delayed
from openai import OpenAI
from tqdm import tqdm

from nils.annotator.prompts import TASK_IDENTIFICATION_PROMPT


class OpenAIAPI:
    def __init__(self, gpt_version="gpt-4o-mini"):
        api_key = "YOUR_API_KEY"
        anyscale_key = "YOUR_ANYSCALE_API_KEY"
        # self.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.model_name = gpt_version
        self.gpt3 = "gpt-3.5-turbo"
        self.gpt4 = "gpt-4o"
        self.gpt4_mini = "gpt-4o-mini"
        if self.model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            self.client = OpenAI(api_key=anyscale_key, base_url="https://api.endpoints.anyscale.com/v1")
        else:
            self.client = OpenAI(api_key=api_key)

    def prompt_llm(self, prompt, sampling_params=None, model_name="gpt3", temperature=0.2):
        messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]

        if "assistant" in prompt.keys():
            messages.append({"role": "assistant", "content": prompt["assistant"]})
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.gpt4_mini,
            logprobs=True,

            temperature=temperature,
            # top_p=0.9,

        )

        llm_response = chat_completion.choices[0].message.content
        return llm_response

    def query(self, prompt, sampling_params, save=False, save_dir=""):

        messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]

        if "assistant" in prompt.keys():
            messages.append({"role": "assistant", "content": prompt["assistant"]})

        chat_completion = self.client.chat.completions.create(
            messages=messages,

            model=self.model_name,
            logprobs=True,
            temperature=0.1
        )

        # Clean completion:
        llm_response = chat_completion.choices[0].message.content
        produced_json = re.search(r"```({.*})```", llm_response, re.DOTALL)
        if not produced_json:
            try:
                json_extracted = json.loads(llm_response)
                action_candidates = json_extracted["task candidates"].split(",")
                task = json_extracted["task"]



            except Exception:
                logging.error(f"Error parsing LLM response: {llm_response}")
        elif produced_json:
            json_extracted = produced_json.group(1)
            try:
                json_parsed = json.loads(json_extracted)
                action_candidates = json_parsed["task candidates"].split(",")
                task = json_parsed["task"]
            except Exception as e:
                logging.error(f"Error parsing LLM response: {e}")

        total_log_probs = []
        logprobs = chat_completion.choices[0].logprobs.content
        start_count = False

        for i, prob in enumerate(logprobs):
            # print(prob)
            if prob.token == "task" and ":" in logprobs[i + 1].token and '"' in logprobs[i + 2].token:
                start_count = True
            if start_count:
                total_log_probs.append(np.exp(prob.logprob))
            if prob.token in ['",\n', '"'] and start_count:
                total_log_probs.pop()
                break
        total_log_probs = np.prod(total_log_probs)

        return task, action_candidates, total_log_probs

    def get_task(self, prompt, sampling_params=None):
        messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]

        chat_completion = self.client.chat.completions.create(
            messages=messages,

            model=self.gpt4,
            logprobs=True,
            temperature=0.2,
            max_tokens=700
        )

        llm_response = chat_completion.choices[0].message.content
        llm_response = "```" + llm_response
        response_cleaned = llm_response.replace("\n", "")
        response_cleaned = response_cleaned.replace("json", "")
        produced_json = re.search(r"```({.*})```", response_cleaned, re.DOTALL)
        if produced_json:
            response_dict = json.loads(produced_json.group(1))
            tasks = response_dict["tasks"].split(",")
            if "task candidates" in response_dict.keys():
                candidates = response_dict["task candidates"].split(",")
            else:
                candidates = []
            scores = response_dict["confidence"].split(",")
            scores = [float(score) for score in scores]
        else:
            produced_json = re.search(r"({.*})", response_cleaned, re.DOTALL)
            if produced_json:
                response_dict = json.loads(produced_json.group(1))
                tasks = response_dict["tasks"].split(",")
                if "task candidates" in response_dict.keys():
                    candidates = response_dict["task candidates"].split(",")
                else:
                    candidates = []
                scores = response_dict["confidence"].split(",")
                scores = [float(score) for score in scores]
            else:
                tasks = []
                candidates = []
                scores = []

        return tasks, candidates, scores

    def get_nl_prompt_keystate_reasons(self, task_list, observations, open_ended=False, aggregated=False):

        prompt_system_open_ended = '''You will be provided with observations of a robot interaction with an environment, delimited by triple quotes.
Determine the task the robot could have solved. The robot can only solve one task. If the observations indicate that the robot interacted with multiple objects, focus on the most frequent and precise observations.

Follow these guidelines:
Step 1: Answer what objects appear in the observation. List all objects. Then, determine the object for which the observations align the best.
Step 2: Determine the object movement and the resulting object relations. Think about where the object and its relational objects are located in the scene on a global scale. Think step by step and list the locations and relations of all objects. Explain the object movements.
Step 3: Determine what tasks result in the object relations from Step 2.
Step 4: Output tasks that that accomplish the observations as short instructions. Focus on simple, single-step tasks that only require interaction with the determined object from Step 1. Focus on tasks that include changing the object relation and moving the object.
Example tasks: "Place the pot to the left of the fruit", "Move the dishrag to the bottom of the table next to the towel", "Put the pot to the right of the fruit", "Turn on stove", "Open the microwave", "Put the knife inside the sink".
Follow the steps above. Explain your reasoning. Output the reasoning delimited by ***.
After, produce your output as JSON. The format should be:
```{
"tasks": "The determined tasks, delimited by commas. Output 3 different task instructions.",
"confidence": "A confidence score for each task between 0 and 10, delimited by commas. Be pessimistic."
}```
'''
        prompt_user_open_ended = "Observations: ```" + observations + "```"

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

        prompt_system = prompt_system.replace("[TASK_LIST]", task_list)

        prompt_user = "Observations: ```" + observations + "```"

        if open_ended:
            prompt_system = prompt_system_open_ended
            prompt_user = prompt_user_open_ended

        return {"system": prompt_system, "user": prompt_user}

    def get_possible_actions(self, object_list):

        prompt = build_possible_actions_prompt(object_list)
        llm_response = self.prompt_llm(prompt, "gpt4", temperature=0.1)

        try:
            parsed = llm_response.split(",")
            if len(parsed) == 1:
                parsed = llm_response.split(";")
            if len(parsed) == 1:
                parsed = llm_response.split("\n")
            parsed = [p.strip().replace("\n", "") for p in parsed]
            parsed = [''.join([char for char in p if not char.isdigit()]) for p in parsed]




        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            parsed = None

        return parsed

    def get_object_task_list(self, tasks):

        task_list = ["-" + task for task in tasks]
        task_list = "\n".join(task_list)
        prompt = TASK_IDENTIFICATION_PROMPT.replace("[TASK_LIST]", task_list)

        prompt_dict = {"system": "", "user": prompt}

        llm_response = self.prompt_llm(prompt_dict)

        s = 1

    def eval_correctness(self, pred_task, gt_task):
        pred_task_nl = (",").join(pred_task)
        system_prompt = """You will be provided with a list of task predictions and a ground truth task. Determine if any task from the list accomplishes the same thing as the ground truth task. 
A task accomplished the same thing when the following criteria are met: 
1. In both tasks, the same objects are interacted with.
2. The resulting object relations after performing the task are the same.
Objects can have synonyms, for instance a pot and a saucepan.
Output 'Yes' if any task from the task accomplishes the same thing as the ground truth task, else output 'No'."""
        user_prompt = f"Prediction Task List: {pred_task_nl}\nGround Truth: {gt_task}\n."

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.gpt3,
            logprobs=True,
            # top_logprobs=2,
            temperature=0.0,
            # top_p=0.9,
        )
        response = chat_completion.choices[0].message.content

        return response


example_tasks = ["Place the tin can to the left of the pot",
                 "Move the dishrag to the bottom of the table next to the towel"
    , "Put the pot to the right of the fruit", "Turn on stove", "Open the microwave", "Put the knife inside the sink"]
example_tasks = ";".join(example_tasks)


def build_possible_actions_prompt(object_list):
    objects_nl = [f"{obj.split(',')[0]}" for obj in object_list]

    objects_nl = ", ".join(object_list)

    system_prompt = (
                        "You will be provided with a list of objects observed by a robot. Based on the objects, give possible instructions to the robot. Infer the type of environment from the provided objects. Ignore objects that do not fit the environment.\n"
                        "Follow these guidelines:"
                        "- Keep the instructions simple. Focus on tasks that only require a single step.\n"
                        "- Include tasks that change the relation of objects to each other, such as to the left .\n"
                        "- Dont assume the presence of any objects not listed. \n"
                        "- Output a complete list of tasks, including every object at least 4 times.\n"
                        "Output at least 30 possible instructions delimited by comma.\n\n"
                        "Here are a few examples: \n") + example_tasks
    user_prompt = "Objects:```{}```.\n".format(objects_nl)

    system_prompt = """You will be provided with a list of objects observed by a robot, delimited by triple quotes. Based on the objects, give possible instructions to the robot. Infer the type of environment from the provided objects. Ignore objects that do not fit the environment.
Follow these guidelines:
- Keep the instructions simple. Focus on tasks that only require a single step.
- Include tasks that change the relation of objects to each other. Relations include: inside, to the left, to the right, to the bottom, to the top, next to, on top of, and below. 
- Make sure the produced tasks are applicable and make sense in the given environment.
- Output a complete list of instructions with all object relations, including every object at least 4 times.
Output at least 25 different instructions, delimited by comma.
Here are a few examples:
[EXAMPLE_TASKS]"""
    assistant_prompt = "Instructions:"

    return {"system": system_prompt, "user": user_prompt, "assistant": assistant_prompt}


def query_parallel(prompt, model="gpt-4o-mini"):
    messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
    client = OpenAI(api_key="YOUR_KEY")

    n_retries = 2
    success = False
    while n_retries > 0 and not success:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                # model= self.gpt3 if model_name == "gpt3" else self.gpt4,
                model=model,
                temperature=0.2,
            )
            success = True
        except Exception as e:
            logging.error(f"Error in LLM query: {e}")
            n_retries -= 1
            if n_retries == 0:
                return ["Error in response from LLM"]

    llm_response = chat_completion.choices[0].message.content

    return llm_response


def get_tasks_parallel_gpt(prompts):
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(query_parallel)(prompt)
        for idx, prompt in enumerate(tqdm(prompts)))

    llm_outputs = [result for result in results]

    all_tasks = []
    all_candidates = []
    all_scores = []

    for response in llm_outputs:

        response_cleaned = response.replace("\n", "")
        response_cleaned = response_cleaned.replace("json", "")
        produced_json = re.search(r"```({.*})```", response_cleaned, re.DOTALL)
        candidates = []

        task_key = "tasks" if "tasks" in response_cleaned else "task"

        if produced_json:
            response_dict = json.loads(produced_json.group(1))
            if response_dict[task_key] is None:
                tasks = []
                scores = []
            else:
                if task_key == "tasks":
                    tasks = response_dict["tasks"].split(";")
                    tasks = [task.strip() for task in tasks]
                    scores = response_dict["confidence"].split(",")
                else:
                    tasks = [response_dict["task"]]
                    scores = [response_dict["confidence"]]
            if "task candidates" in response_dict.keys():
                candidates = response_dict["task candidates"].split(",")
            else:
                candidates = []

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
        all_tasks.append(tasks)
        all_candidates.append(candidates)
        all_scores.append(scores)

    return all_tasks, all_candidates, all_scores


def get_simple_prompt_nl_reasons_gpt(observations):
    prompt_system_open_ended = '''You will be provided with observations of a robot interaction with an environment, delimited by triple quotes.
Determine the task the robot solved. The robot can only solve one task. If the observations indicate that the robot interacted with multiple objects, focus on the most frequent and precise observations.

Example tasks: "Place the pot to the left of the fruit"; Slide the dishrag to the bottom of the table next to the towel"; Pick up the spoon and place it  at the bottom left of the table;"Put the lid on the pot to close it";"Shift the eggplant to the left and forward away from the other objects";"Put the pot to the right of the fruit"; "Move the pot forward and to the left"; "Turn on stove"; "Open the microwave"; "Place the spoon on top of the cloth".

roduce your output as JSON. The format should be:
```{
"tasks": "The determined tasks, delimited by semicolons. Infer the task from the observations and be as precises as possible. Output 3 different task instructions.",
"confidence": "A confidence score for each task between 0 and 10, delimited by commas. Precise descriptions and tasks that make sense should have a higher score."
}```
'''
    prompt_user_open_ended = "Observations: ```" + observations + "```"

    prompt_system = prompt_system_open_ended
    prompt_user = prompt_user_open_ended

    return {"system": prompt_system, "user": prompt_user}
