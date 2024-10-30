import numpy as np

from nils.annotator.keystate_predictors.keystate_predictor import (
    KeystatePredictor,
)


class GripperClosePredictor(KeystatePredictor):
    
    def __init__(self,name, downsample_interval=1, gripper_close_to_open = True, gripper_open_to_close = False,gripper_frame_offset = 0):
        super(GripperClosePredictor, self).__init__(name,downsample_interval)

        self.gripper_close_to_open = gripper_close_to_open
        self.gripper_open_to_close = gripper_open_to_close
        self.gripper_frame_offset = gripper_frame_offset
        
        
    
    def predict(self, data):
        
        batch = data["batch"]
        
        
        gripper_close_actions = batch["actions"][:, -1]
        

            

        past_action = 1  # open

        keystates = []
        closed_step = 0
        opened_step = 0
        for i, gripper_action in enumerate(gripper_close_actions):

            if gripper_action <= 0:  # closed gripper

                # open -> closed
                if past_action == 1:
                    closed_step = i
                    if i - opened_step > 2 and self.gripper_open_to_close:
                        keystates.append(i)

                # else:
                #     # On gripper closed
                #     # Was closed and remained closed
                #
                #     self.closed_gripper(dct)
            else:
                # closed -> open
                if past_action <= 0:
                    opened_step = i
                    if i - closed_step > 2 and self.gripper_close_to_open:
                        keystates.append(i + self.gripper_frame_offset)
            past_action = gripper_action
        
        self.keystates = keystates
        
        self.keystate_scores = [1] * len(keystates)
        
        for obj in data["object_manager"].objects.keys():
            if obj not in self.keystates_by_object:
                self.keystates_by_object[obj] = []
            self.keystates_by_object[obj]= keystates
            self.object_keystate_scores[obj] = [0.6] * len(keystates)
        
        return keystates




def get_keystates_from_gripper_close(batch,offset = 0):
   

    gripper_close_actions = batch["actions"][:, -1]

    past_action = 1  # open

    keystates = []
    closed_step = 0
    opened_step = 0
    for i, gripper_action in enumerate(gripper_close_actions):

        if gripper_action <= 0:  # closed gripper

            # open -> closed
            if past_action == 1:
                closed_step = i

            # else:
            #     # On gripper closed
            #     # Was closed and remained closed
            #
            #     self.closed_gripper(dct)
        else:
            # closed -> open
            if past_action <= 0:
                opened_step = i
                if i - closed_step >= 2:
                    keystates.append(i)
                else:
                    s = 1
        past_action = gripper_action

    return np.array(keystates) + offset