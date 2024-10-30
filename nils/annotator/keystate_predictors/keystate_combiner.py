import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from nils.annotator.keystate_predictors.keystate_predictor import (
    KeystatePredictor,
)
from nils.annotator.keystate_voting import (
    score_keystates,
)
from nils.utils.utils import remove_and_keep_highest


class KeystateCombiner:
    def __init__(self,keystates_for_voting = None):
        self.predictors = []

        self.keystates = None

        self.object_specific_keystates = None
        self.object_specific_scores = None

        self.combined_keystates = None

        self.combined_keystates_threshold = None

        self.keystate_scores = None
        self.keystate_objects = []

        self.method_weights = {

            "object_movement_predictor": 0.9,
            "scene_graph_predictor": 0.7,
            "gripper_close_predictor": 0.6,
            "gripper_position_predictor": 0.8,
            "object_state_predictor": 1.4,

        }
        
        self.keystates_for_voting  = keystates_for_voting

        self.use_object_specific_voting = True

    def add_predictor(self, predictor):
        if not isinstance(predictor, KeystatePredictor):
            raise ValueError("Predictor must be an instance of KeystatePredictor")
        self.predictors.append(predictor)
        
    def init_keystates_by_object(self,objects):
        for predictor in self.predictors:
            predictor.init_keystates_by_object(objects)

    def predict_keystates(self, data):
        
        keystates = {}
        for predictor in self.predictors:
            
            if predictor.name in self.keystates_for_voting: 
                
                keystates[predictor.name] = predictor.predict(data)
            else:
                predictor.predict(data)

        self.keystates = keystates

        self.object_specific_keystates = self.get_object_specific_keystates()
        self.object_specific_scores = self.get_object_specific_scores()

        return keystates

    def get_object_specific_keystates(self):

        obj_keystates = {}
        all_objects = np.unique(np.concatenate(
            [np.array([obj for obj in predictor.keystates_by_object.keys()]) for predictor in self.predictors]))

        for obj in all_objects:
            obj_keystates[obj] = {}
            for predictor in self.predictors:
                if predictor.name not in self.keystates_for_voting:
                    continue
                if obj in predictor.keystates_by_object:
                    obj_keystates[obj][predictor.name] = predictor.keystates_by_object[obj]
                else:
                    obj_keystates[obj][predictor.name] = []

        return obj_keystates
    
    def get_object_specific_scores(self):
            
            obj_scores = {}
            all_objects = np.unique(np.concatenate(
                [np.array([obj for obj in predictor.keystates_by_object.keys()]) for predictor in self.predictors]))
    
            for obj in all_objects:
                obj_scores[obj] = {}
                for predictor in self.predictors:
                    if predictor.name not in self.keystates_for_voting:
                        continue
                    if obj in predictor.object_keystate_scores:
                        obj_scores[obj][predictor.name] = predictor.object_keystate_scores[obj]
                    else:
                        obj_scores[obj][predictor.name] = []
    
            return obj_scores

    def combine_keystates(self, match_threshold=12, score_threshold=0.5,min_keystate_distance = 3):

            
            


                


        if self.use_object_specific_voting:
            object_keystates = self.object_specific_keystates.copy()
            all_keystates = []
            all_keystates_unfiltered = []
            all_keystate_scores_unfiltered = []
            all_keystate_objects_unfiltered = []

            all_scores = []
            keystate_objects = []
            obj_keystates = {}

            for obj, cur_obj_keystates in object_keystates.items():
                keystates = self.object_specific_keystates[obj]
                object_keystate_scores = self.object_specific_scores[obj]
                if len(keystates) == 1:

                    scored_keystates = list(keystates.values())[0]
                    scores = np.ones(len(scored_keystates))
                else:


                    scored_keystates, scores = score_keystates(keystates,object_keystate_scores, match_threshold, self.method_weights)

                # merge close keystates:
                df = pd.DataFrame({"keystates": scored_keystates, "scores": scores})

                df = df.sort_values(by="scores").drop_duplicates(subset="keystates", keep="last")
                df = df.sort_values(by="keystates")


                df['group'] = (df['keystates'].diff() > 4).cumsum()

                merged_keystates = df.sort_values("scores").groupby('group')['keystates'].last().astype(int)
                merged_scores = df.groupby('group')['scores'].max()


                merged_scores += ((df.groupby('group')['keystates'].count()) * 0.01)-0.1

                unique_keystates = merged_keystates.values
                unique_keystate_scores = merged_scores.values




                #unique_keystates, unique_keystate_indices = np.unique(scored_keystates, return_index=True)
                #unique_keystate_scores = np.array(scores)[unique_keystate_indices]






                topk_scores = np.sort(unique_keystate_scores)[-2:]



                if len(unique_keystates) == 0:
                    continue
                good_keystate_indices = np.where(np.array(unique_keystate_scores) >= score_threshold)[0]
                good_keystate_indices = np.where(np.array(unique_keystate_scores) >= max(topk_scores[0]-0.1,0.3))[0]
                good_keystates = np.array(unique_keystates)[good_keystate_indices]
                good_keystate_scores = np.array(unique_keystate_scores)[good_keystate_indices]


                all_keystates_unfiltered.append(unique_keystates)
                all_keystate_scores_unfiltered.append(unique_keystate_scores)
                all_keystate_objects_unfiltered.append(np.array([obj] * len(unique_keystates)))


                all_keystates.append(good_keystates)
                all_scores.append(good_keystate_scores)
                obj_keystates[obj] = good_keystates
                keystate_objects.append(np.array([obj] * len(good_keystates)))
            if len(all_keystates) == 0:
                return []
            keystates = np.concatenate(all_keystates)
            keystate_scores = np.concatenate(all_scores)
            keystate_objects = np.concatenate(keystate_objects)


            all_keystates_unfiltered = np.concatenate(all_keystates_unfiltered)
            all_keystate_scores_unfiltered = np.concatenate(all_keystate_scores_unfiltered)
            all_keystate_objects_unfiltered = np.concatenate(all_keystate_objects_unfiltered)

            use_gripper_close_as_anchor = False
            if use_gripper_close_as_anchor:
                #construct cost matrix:
                gripper_close_keystates = [predictor for predictor in self.predictors if predictor.name == "gripper_close_predictor"][0]
                gripper_close_keystates = gripper_close_keystates.keystates

                cost_matrix = np.zeros((len(gripper_close_keystates),len(all_keystates_unfiltered)))
                for ks_idx in range(len(all_keystates_unfiltered)):
                    for gc_idx in range(len(gripper_close_keystates)):
                        if np.abs(gripper_close_keystates[gc_idx] - all_keystates_unfiltered[ks_idx]) > 8:
                            cost_matrix[gc_idx,ks_idx] = 10000
                        cost_matrix[gc_idx,ks_idx] += 1 - all_keystate_scores_unfiltered[ks_idx]

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                all_keystate_scores_unfiltered[col_ind] += 0.4
                all_keystate_scores_unfiltered = np.clip(all_keystate_scores_unfiltered,0,1)

            #keystates by object:

            ks = pd.DataFrame({"keystates":all_keystates_unfiltered,"scores":all_keystate_scores_unfiltered,"objects":all_keystate_objects_unfiltered})
            obj_keystates  = {obj: group.keystates.values for obj,group in ks.groupby("objects")}
            obj_scores = {obj: group.scores.values for obj,group in ks.groupby("objects")}

            s = 1

            max_idx = all_keystates_unfiltered.max()
            if max_idx>250:
                topk_idx = -3
            else:
                topk_idx = -2
            #topk_idx = -2

            use_all_keystates = True

            final_obj_ks = {}
            final_obj_scores = {}
            for obj,keystates in obj_keystates.items():

                scores = obj_scores[obj]
                sorted_scores = np.sort(scores)
                top_2_scores = np.sort((scores))[topk_idx:]


                if 0.3 > sorted_scores[-1] > 0.2:
                    thresh =  sorted_scores[-1]
                    print(sorted_scores[-1])
                    print(thresh)
                else:
                    thresh = max(0.3,(top_2_scores[0]) -0.1)

                if use_all_keystates:
                    thresh = 0.0
                final_ks = keystates[scores >= thresh]
                final_scores = scores[scores >= thresh]
                final_obj_ks[obj] = final_ks
                final_obj_scores[obj] = final_scores

            final_df = pd.DataFrame({"keystates":np.concatenate(list(final_obj_ks.values())),"scores":np.concatenate(list(final_obj_scores.values())),"objects":np.concatenate([[obj]*len(ks) for obj,ks in final_obj_ks.items()])})





            if len(keystates) == 0:
                return []
            #ind_to_keep = remove_and_keep_highest(keystates,keystate_scores,keystate_objects,4)

            # keystates_filtered = keystates[ind_to_keep]
            # keystate_scores_filtered = keystate_scores[ind_to_keep]
            # keystate_objects_filtered = keystate_objects[ind_to_keep]

            final_df = final_df.sort_values("keystates", ascending=True)


            keystates = final_df["keystates"].values
            keystate_scores = final_df["scores"].values
            keystate_objects_filtered = final_df["objects"].values

            if len(keystates)  == 0:
                return keystates
            ind_to_keep = remove_and_keep_highest(keystates,keystate_scores,keystate_objects_filtered,min_keystate_distance)

            keystates = keystates[ind_to_keep]
            keystate_scores = keystate_scores[ind_to_keep]
            keystate_objects_filtered = keystate_objects_filtered[ind_to_keep]




        else:

            scored_keystates, scores = score_keystates(keystates, match_threshold, self.method_weights)

            unique_keystates, unique_keystate_indices = np.unique(scored_keystates, return_index=True)
            unique_keystate_scores = np.array(scores)[unique_keystate_indices]

            good_keystate_indices = np.where(np.array(unique_keystate_scores) >= score_threshold)[0]
            good_keystates = np.array(unique_keystates)[good_keystate_indices]
            good_keystate_scores = np.array(unique_keystate_scores)[good_keystate_indices]
            keystates = good_keystates
            keystate_scores = good_keystate_scores
        self.keystate_scores = keystate_scores

        
    
        
        self.combined_keystates = keystates
        self.combined_keystates_threshold = match_threshold
        self.keystate_objects = keystate_objects_filtered
        
        return keystates

    def get_keystate_reasons(self):

        if self.combined_keystates is None:
            raise ValueError("Keystates not computed")

        reasons = {}

        for predictor in self.predictors:
            method_reasons = predictor.get_keystate_reasons(self.combined_keystates, self.combined_keystates_threshold,self.keystate_objects)
            reasons[predictor.name] = method_reasons
        return reasons

    def get_keystate_scores(self):

        if self.keystate_scores is None:
            raise ValueError("Keystate scores not computed")

        return self.keystate_scores
