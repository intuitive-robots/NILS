import numpy as np


class KeystatePredictor:
    def __init__(self,name,downsample_interval):
        
        self.name = name
        
        self.downsample_interval = downsample_interval
        
        self.keystates = None
        self.keystate_reasons = None
        self.keystate_reasons_nl = None
        
        self.keystate_scores = None
        
        self.object_keystate_scores = None
        
        self.keystate_reasons_by_object = None
        self.keystate_reasons_by_object_nl = None
        
        
        self.keystate_pred_confidence = None
        
        
        self.keystates_by_object = None
        
        self.placeholder_prompt = ""
        
    def init_keystates_by_object(self,objects):
        self.keystates_by_object = {obj: [] for obj in objects}
        self.object_keystate_scores = {obj: [] for obj in objects}
        self.keystate_reasons_by_object = {obj: [] for obj in objects}
        self.keystate_reasons_by_object_nl = {obj: [] for obj in objects}
        

    def predict(self, batch, scene_graphs=None, labeled_gripper_cam_data=None):
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def get_keystate_reasons(self, combined_keystates, combined_keystates_threshold,keystate_objects):
        
        matched_keystate_reasons = []
        if self.keystate_reasons_nl is None or len(self.keystate_reasons_nl) == 0:
            return [None] * len(combined_keystates)
        
        for idx,combined_keystate in enumerate(combined_keystates):
            cur_obj = keystate_objects[idx]
            keystate_dists = np.abs(self.keystates_by_object[cur_obj] - combined_keystate)
            if len(keystate_dists) == 0:
                matched_keystate_reasons.append(self.placeholder_prompt)
                continue
            min_dist = np.min(keystate_dists)
            if min_dist < combined_keystates_threshold:
                #reason = self.keystate_reasons_nl[np.argmin(keystate_dists)]
                reason = self.keystate_reasons_by_object_nl[cur_obj][np.argmin(keystate_dists)]
                matched_keystate_reasons.append(reason)
            else:
                matched_keystate_reasons.append(self.placeholder_prompt)
        return matched_keystate_reasons
          
            
    
    


    
    
    
    
