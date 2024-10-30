# TASK_IDENTIFICATION_PROMPT = {
#     "system": "You are an expert in identifying actions performed by a robot. "
#               "Given a sequence of observations over multiple frames, "
#               "select the most likely task performed by the robot from a provided list of tasks. \n."
#               "You are also provided with the state of the environment before the robot started to interact with the environment. \n"
#               "The available tasks are: \n[TASK_LIST]\n"
#               "Analyze each frame step by step. Pay special attention to the distances of the gripper to the objects in the scene.\n"
#               """Analyze the frames based on the following steps:\n
# Step 1:  Analyze the distances of objects to the robot gripper. Interactions are likely to happen in later frames, after the robot moved to the object.\n
# Step 2: Analyze the movements of of the objects\n
# Step 3: Analyze if the robot and any object moved similarly, e.g. both moved right.\n"""
#               "Output the most likely action from the list delimited by triple quotes. \n"
#               "If you are not certain about the performed action, output 'None'.\n",
#
#     "user_prefix":  """Select the most likely task performed by the robot from the following list of tasks: \n[TASK_LIST]\nYou are provided with observations of the robot for multiple frames. Each observation has a score attached, which represents the distance of the object to the robot gripper. You are also provided with the robot movements.\nThe robot made the following observations:\n""",
#     "user_suffix": """What task from the list did the robot perform? Think step by step.\n""",
#     "user": "Frame [FRAME_NUMBER]: [OBSERVATIONS]\n\n",
#
# }
import json

import numpy as np

from nils.utils.utils import get_gripper_movement_nl_3d
from nils.utils.visualizer import Visualizer

TASK_IDENTIFICATION_PROMPT = {
    "system": "You are an expert in identifying actions performed by a robot. "
              "Given a sequence of observations over multiple frames, "
              "select the most likely task performed by the robot from a provided list of tasks. \n"
              """Follow these steps:
Step 1: Analyze the frames and object movements frame by frame, especially the gripper object distances.
Step 2: Analyze whether the object states changed.
Step 3: Choose a task from the list of possible actions.
Output the task only after your reasoning."""

              "Always pick the most specific matching action",

    "user_prefix": "The available tasks are: \n[TASK_LIST]\n",
    "user_suffix": """Output the most likely action from the list in the following format: {action: predicted_action}. \n 
If you are unsure about the performed action, output 'None'. \n
 """,

    "user": "Frame [FRAME_NUMBER]:\n [OBSERVATIONS]\n",

}

one_shot_prompt = " The available tasks are: \ngrasp the red block in the drawer\nin the cabinet grasp the red block\nlift the red block\npush the red block towards the left\npush the red block towards the right\nrotate left the red block\nrotate right the red block\nstack blocks on top of each other\npush the button to turn off the led\nslide down the switch\npush the button to turn on the led\nslide up the switch\ntake off the block that is on top of the other one\nAnalyze each frame step by step. Make sure to consider the initial state of the environment before the robot interacted with it, presented in Frame 1.\n\n\nObservations: \nFrame 1: light_switch is on the right of red_block. red_block is on top of table_region. red_block is inside cabinet_region.\nInitial object states: slider was right. drawer was open. lightbulb was off. led was on. \n\n Distances to Robot Gripper:\n{light_switch: 26.0,red_block: 7.1,}\n\n\nFrame 2: light_switch is on the right of red_block. red_block is on top of table_region. red_block is inside cabinet_region.\nObject Movements:\n {}\n Distances to Robot Gripper:\n{light_switch: 15.1,red_block: 0.0,}\n\n\nFrame 3: red_block is on top of table_region. red_block is inside cabinet_region.\nObject Movements:\n {robot(left,0),red_block(left,-1),}\n Distances to Robot Gripper:\n{light_switch: 23.0,red_block: 0.0,}\n\n\nFrame 4: red_block is inside table_region.\nObject Movements:\n {robot(down,11),red_block(down,14),}\n Distances to Robot Gripper:\n{black_button: 27.1,red_block: 0.0,}\n\n\nOutput the most likely action from the list in the following format: {action: predicted_action}. \n \nIf you are unsure about the performed action, output 'None'. \n\nAlso, output the new object states after the robot performed your predicted action in form {object: new_state} \n\nAnalyze each frame before outputting the action."

one_shot_example = """Frame 1 is the initial State. The closest object to the gripper is the red_block. \n
In Frame 2, the distance to the red_block and light_switch decreased. The robot distance to the red block is very small, indicating the robot is holding it. \n
In Frame 3, the distance to the red_block is still very small, the robot is still holding it. \n
In Frame 4, the distance to the red block is still very small, the robot is still holding it. The robot and the the red block are both moving down, indicating that the robot moved the block. \n
The robot interacted with the red block, as it was closest to the robot for most of the time. As the initial state of the red block was 'inside cabinet_region', the robot likely performed 
{action: in the cabinet grasp the red block}. \n"""

STATE_DETECTION_PROMPT = (
    "A robot performed the task '[TASK]'. Do any of these object states change? \n[OBJECT_STATES] \n "
    "It is possible that no state changed. Dont make any assumptions about followup tasks. Output the new state dict or unmodified state dict as json. Final Object States:")

TASK_IDENTIFICATION_PROMPT_CODE = {
    "system": "You are an expert in identifying actions performed by a robot. "
              "Given a sequence of observations over multiple frames, "
              "select the most likely task performed by the robot from a provided list of tasks. \n"
              "The available tasks are: \n[TASK_LIST]\n"
              "Analyze each frame step by step. Pay special attention to the distances of the gripper to the objects in the scene.\n"
              "Output the most likely action from the list delimited as a python variable named performed_action. \n",
    "user_prefix": """Select the most likely task performed by the robot from the following list of tasks: \n[TASK_LIST]\nYou are provided with observations of the robot for multiple frames. Each observation has a score attached, which represents the distance of the object to the robot gripper. You are also provided with the robot movements.\nThe robot made the following observations:\n""",
    "user_suffix": """What task from the list did the robot perform? Think step by step.\nA:""",
    "state_prefix": "# Initial states of objects\n",
    "observation_prefix": "# Observations made by the robot\n"

}

OBJECT_EXTRACTION_PROMPT = """A robot can perform multiple tasks in an environment. Extract Objects in the environment from the list of tasks. \n
Infer what other objects might be necessary to complete the task.
Determine the following properties for each object from the list of actions:\n
states: List of possible states, can be None
container: Can the object contain other objects? True or False
movable: Can the object be moved into an container? True or False
interactable: Can the robot interact with the object or is it just a container? True or False\n

Output the result in json. Example:
Task List:\n
- place the pink block inside the box
- place the pink block inside the drawer
- place the pink block on the table
- grasp the sliding door and move it left
- open the drawer
- turn on the stove
- move the fruit next to the pot 
- put the fruit inside the pot
Objects:\n
{"pink block": {"states": null, "movable": true, "container": false, "interactable": true},
"drawer": {"states": ["open", "closed"], "movable": false, "container": true, "interactable": true},
"box": {"states": null, "movable": false, "container": true, "interactable": false},
"top of the drawer": {"states": [], "movable": false, "container": false, "interactable": true},
"table": {"states": null, "movable": false, "container": false, "interactable": true},
"stove": {"states": ["On", "Off"], "movable": false, "container": true, "interactable": false},
"stove_knob": {"states": null, "movable": false, "container": false, "interactable": true},
"sliding_door": {"states": [left,right], "movable": false, "container": false, "interactable": true},
"fruit": {"states": null, "movable": true, "container": false, "interactable": true},
"pot": {"states": null, "movable": true, "container": true, "interactable": true}
}
\n\n
Task List:\n
[TASK_LIST]\n
Objects:\n
"""

OBJECT_PROPERTIES_PROMPT = """You will be provided with a list of object a robot observes in an environment. Determine the following properties for each object:\n
states: List of possible states, can be None
container: Can the object contain other objects? True or False
movable: Can the object be moved into an container? True or False
interactable: Can the robot interact with the object by changing its state or moving components? True or False\n
Output the result in json. Example:
Object list:\n
fridge, microwave oven, baking tray, pink bowl, white bowl, toaster, toast, pot,sink, faucet, stove top, toy microwave
Objects:\n
{
"pot": {"states": null, "movable": true, "container": true, "interactable": true},
"fridge": {"states": ["open","closed"], "movable": false, "container": true, "interactable": true},
"microwave oven": {"states": ["open","closed"], "movable": false, "container": true, "interactable": true},
"baking tray": {"states": null, "movable": true, "container": false, "interactable": true},
"pink bowl": {"states": null, "movable": true, "container": true, "interactable": true},
"white bowl": {"states": null, "movable": true, "container": true, "interactable": true},
"toaster": {"states": null, "movable": false, "container": true, "interactable": true},
"toast": {"states": null, "movable": true, "container": false, "interactable": true},
"sink": {"states": null, "movable": false, "container": true, "interactable": false},
"faucet": {"states": null, "movable": false, "container": false, "interactable": false},
stove top: {"states": null, "movable": false, "container": false, "interactable": false},
toy microwave: {"states": ["open","closed"], "movable": false, "container": true, "interactable": true},
wooden box: {"states": ["open","closed"], "movable": false, "container": true, "interactable": true},
}
\n\n
Object list:\n
[OBJECT_LIST]\n
Objects:\n
"""

CALVIN_IN_CONTEXT_PROMPTS = """
Multi-choice problem: What task did the robot perform?
-Close the drawer
-Lift the blue block from the drawer
-Lift the blue block from the cabinet
-Lift the blue block from the table
-Lift the pink block from the drawer
-Lift the pink block from the cabinet
-Lift the pink block from the table
-Lift the red block from the drawer
-Lift the red block from the cabinet
-Lift the red block from the table
-Move the slider to the left
-Move the slider to the right
-Open the drawer
-Place the object in the gripper in the drawer
-Place the the object in the gripper in the cabinet
-Push the blue block to the left
-Push the blue block to the right
-Push the object in the gripper into the drawer
-Push the pink block to the left
-Push the pink block to the right
-Push the red block to the left
-Push the red block to the right
-Rotate the blue block to the left
-Rotate the blue block to the right
-Rotate the pink block to the left
-Rotate the pink block to the right
-Rotate the red block to the left
-Rotate the red block to the right
-Stack the block
-Press the button to turn off the LED
-Move down the switch to turn off the lightbulb
-Press the button to turn on the LED
-Move up the switch to turn on the lightbulb
-Unstack the block
-None

Observations: Frame 1: drawer is inside drawer_region.
Initial object states: {slider:left,}{drawer:closed,}{lightbulb:off,}{led:off,}
 Distances to Robot Gripper:
{drawer: 23.0,black_button: 26.0,}

Frame 2: drawer is inside drawer_region.
Object Movements:
 {robot(down,12),}
 Distances to Robot Gripper:
{drawer: 3.6,}

Frame 3: drawer is inside drawer_region.The robot holds drawer. 
Object Movements:
 {robot(down,9),}
 Distances to Robot Gripper:
{drawer: 0.0,}

Frame 4: drawer is inside drawer_region.The robot holds drawer. 
Object Movements:
 {drawer(left,10),drawer(down,10),robot(down,7),}
 Distances to Robot Gripper:
{drawer: 0.0,}
The robot performed the task: Open the drawer

Observations: Frame 1: black_button is inside table_region.
Initial object states: {slider:left,}{drawer:open,}{lightbulb:off,}{led:off,}
 Distances to Robot Gripper:
{black_button: 37.7,}

Frame 2: black_button is inside table_region.
Object Movements:
 {robot(left,-8),robot(up,-16),}
 Distances to Robot Gripper:
{slider: 13.0,black_button: 29.7,blue_block: 33.0,}

Frame 3: black_button is inside table_region.
Object Movements:
 {robot(left,1),}
 Distances to Robot Gripper:
{black_button: 20.6,blue_block: 0.0,}

Frame 4: black_button is inside table_region.
Object Movements:
 {}
 Distances to Robot Gripper:
{black_button: 1.4,}
The robot performed the task: Press the button to turn on the LED

Observations: Frame 1: blue_block is on the left of table_region. blue_block is inside cabinet_region.
Initial object states: {slider:left,}{drawer:open,}{lightbulb:off,}{led:on,}
 Distances to Robot Gripper:
{blue_block: 4.0,}

Frame 2: blue_block is on the left of table_region. blue_block is inside cabinet_region.
Object Movements:
 {}
 Distances to Robot Gripper:
{blue_block: 0.0,}

Frame 3: blue_block is on the left of table_region. blue_block is inside cabinet_region.
Object Movements:
 {}
 Distances to Robot Gripper:
{blue_block: 0.0,}

Frame 4: blue_block is on the left of table_region. blue_block is inside cabinet_region.The robot holds blue_block. 
Object Movements:
 {}
 Distances to Robot Gripper:
{black_button: 26.9,blue_block: 0.0,}
The robot performed the task: Lift the pink block from the cabinet

Observations: Frame 1: red_block is inside cabinet_region.The robot holds red_block. 
Initial object states: {slider:right,}{drawer:open,}{lightbulb:off,}{led:on,}
 Distances to Robot Gripper:
{red_block: 0.0,}

Frame 2: red_block is inside table_region.The robot holds red_block. 
Object Movements:
 {robot(down,11),red_block(down,14),}
 Distances to Robot Gripper:
{black_button: 27.1,red_block: 0.0,}

Frame 3: red_block is inside table_region.The robot holds red_block. 
Object Movements:
 {black_button(down,6),red_block(down,10),}
 Distances to Robot Gripper:
{red_block: 0.0,}

Frame 4: red_block is inside drawer_region.
Object Movements:
 {robot(right,8),red_block(right,8),red_block(down,6),}
 Distances to Robot Gripper:
{drawer: 26.4,red_block: 0.0,}
The robot performed the task: Place the object in the gripper in the drawer

Observations: Frame 1: light_switch is on the right of cabinet_region.
Initial object states: {slider:right,}{drawer:closed,}{lightbulb:off,}{led:on,}
 Distances to Robot Gripper:
{pink_block: 0.0,light_switch: 24.0,}

Frame 2: light_switch is on the right of cabinet_region.
Object Movements:
 {}
 Distances to Robot Gripper:
{pink_block: 5.0,light_switch: 36.7,}

Frame 3: 
Object Movements:
 {robot(right,18),}
 Distances to Robot Gripper:
{light_switch: 18.8,}

Frame 4: 
Object Movements:
 {robot(up,-5),light_switch(up,-3),}
 Distances to Robot Gripper:
{pink_block: 15.5,light_switch: 7.0,}
The robot performed the task: Move up the switch to turn on the lightbulb

Observations: Frame 1: pink_block is inside table_region.
Initial object states: {slider:right,}{drawer:closed,}{lightbulb:off,}{led:on,}
 Distances to Robot Gripper:
{pink_block: 0.0,light_switch: 24.0,}

Frame 2: pink_block is inside table_region. pink_block is on the right of cabinet_region.The robot holds pink_block. 
Object Movements:
 {robot(left,0),pink_block(left,-5),}
 Distances to Robot Gripper:
{pink_block: 0.0,light_switch: 22.0,}

Frame 3: The robot holds pink_block. 
Object Movements:
 {robot(left,-7),robot(up,-10),pink_block(left,-9),pink_block(up,-15),}
 Distances to Robot Gripper:
{pink_block: 0.0,}

Frame 4: pink_block is inside cabinet_region.
Object Movements:
 {robot(right,6),pink_block(right,7),}
 Distances to Robot Gripper:
{pink_block: 0.0,light_switch: 24.0,}

The robot performed the task: Place the the object in the gripper in the cabinet"""


def get_gripper_rotations_nl(robot_obs):
    rot_euler = robot_obs[:, 3:6]

    rot_euler = np.diff(rot_euler, axis=0)
    rotations = []

    rotations.append("No Rotation")
    for rot in rot_euler:
        yaw = rot[-1]  # Assuming the yaw is the third element

        # Normalize the yaw to the range [-pi, pi]
        # yaw = yaw.diff()
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        # print(yaw)
        if yaw < -1:
            print(yaw)
            rotations.append('Gripper rotated right by ' + str(int(abs(np.rad2deg(yaw)))) + ' degrees')
        elif yaw > 1:
            rotations.append('Gripper rotated left by ' + str(int(abs(np.rad2deg(yaw)))) + ' degrees')
        else:
            rotations.append("No rotation")

    return rotations


def create_visualization(image_np, mask, obj_name):
    visualizer = Visualizer(image_np)
    mask = visualizer.draw_binary_mask_with_number(mask, text=obj_name, label_mode=1, alpha=0.4,
                                                   anno_mode=["Mark", "Mask"], color="red")

    return mask.get_image()



def create_observation_prompt(averaged_final_sg, averaged_init_sg, cur_reasons, movements_nl_2d):
    prompt_nl_reasons = ""
    for method, keystate_reason in cur_reasons.items():
        # cur_reason = keystate_reason[keystate_idx]
        cur_reason = keystate_reason
        if cur_reason is not None:
            if isinstance(cur_reason, list):
                cur_reason = " ".join(cur_reason)

            if "movement" in method:
                prompt_nl_reasons += cur_reason + movements_nl_2d + "\n"
            elif "scene_graph" in method:
                prompt_nl_reasons += "Object relation changes:\n Initial relations: " + averaged_init_sg.__str__() + "Final relations: " + averaged_final_sg.__str__() + "\n"
            else:
                prompt_nl_reasons += cur_reason + '\n'
    return prompt_nl_reasons


def get_nl_position_on_surface(object_bbox, surface_bbox):
    large_x_min, large_y_min, large_x_max, large_y_max = surface_bbox
    large_width = large_x_max - large_x_min
    large_height = large_y_max - large_y_min

    # Define the boundaries of the object bbox
    obj_x_min, obj_y_min, obj_x_max, obj_y_max = object_bbox
    obj_width = obj_x_max - obj_x_min
    obj_height = obj_y_max - obj_y_min

    # Calculate the center of the large object bbox
    large_center_x = large_x_min + large_width / 2
    large_center_y = large_y_min + large_height / 2

    # Calculate the center of the object bbox
    obj_center_x = obj_x_min + obj_width / 2
    obj_center_y = obj_y_min + obj_height / 2

    # Determine the vertical position based on the 3x3 grid
    if obj_center_y < large_y_min + large_height / 3:
        vertical_position = "top"
    elif obj_center_y > large_y_min + 2 * large_height / 3:
        vertical_position = "bottom"
    else:
        vertical_position = ""

    # Determine the horizontal position based on the 3x3 grid
    if obj_center_x < large_x_min + large_width / 3:
        horizontal_position = "left"
    elif obj_center_x > large_x_min + 2 * large_width / 3:
        horizontal_position = "right"
    else:
        horizontal_position = ""

    if vertical_position == "" and horizontal_position == "":
        return "center"
    else:
        return vertical_position + " " + horizontal_position
