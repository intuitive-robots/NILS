import numpy as np

from nils.annotator.keystate_predictors.keystate_predictor import (
    KeystatePredictor,
)


class SceneGraphChangeKeystatePredictor(KeystatePredictor):

    def __init__(self,name, downsample_interval=1, only_trigger_object_relations = True):
        super(SceneGraphChangeKeystatePredictor, self).__init__(name,downsample_interval)

        self.only_trigger_object_relations =  True
        
        self.placeholder_prompt = "Object relation changes:\n No object relation change detected"
        

    def predict(self, data):

        scene_graphs = data["scene_graphs"]
        labeled_gripper_cam_data = data["labeled_gripper_cam_data"]
        batch = data["batch"]
        interacted_robot_objects = data["interacted_robot_objects"]






        scene_graph_change = []
        keystate_reasons_by_object = {}
        keystates = []
        for i in range(len(scene_graphs) - 1):

            # If we use gripper cam, changes might not be detected immediately due to occlusion.
            # Look at (smoothed) gripper interaction and sg changed when gripper object changed
            if interacted_robot_objects is not None:
                change_index = np.where(np.array(interacted_robot_objects[i:][:-1]) != np.array(
                    interacted_robot_objects[i + 1:]))
                if len(change_index[0]) > 0:
                    change_index = change_index[0][0]
                else:
                    change_index = 0
                for fwd_idx in range(1, change_index + 2):
                    if scene_graphs[i] != scene_graphs[i + fwd_idx]:
                        if i + fwd_idx in keystates:
                            break
                        keystates.append(i + fwd_idx)
                        edge_change = set(scene_graphs[i + fwd_idx].edges.values()) - set(
                            scene_graphs[i].edges.values())
                        prev_edge_change = set(scene_graphs[i].edges.values()) - set(
                            scene_graphs[i + fwd_idx].edges.values())
                        scene_graph_change.append((prev_edge_change, edge_change))
                        break
            else:
                if scene_graphs[i] != scene_graphs[i + 1]:


                    #Check if object actually moved since last relation change:

                    edge_change = set(scene_graphs[i + 1].edges.values()) - set(scene_graphs[i].edges.values())
                    prev_edge_change = set(scene_graphs[i].edges.values()) - set(scene_graphs[i + 1].edges.values())

                    all_edge_changes = edge_change.union(prev_edge_change)

                    movable_start_nodes = [edge.start for edge in all_edge_changes if edge.start.obj.movable_to_container]
                    len_movable = len(movable_start_nodes)

                    last_sg_change = False if len_movable == 0 else True
                    moved_start_nodes = []

                    if not last_sg_change:
                        continue

                    if len(moved_start_nodes) == 0:
                        moved_start_nodes = movable_start_nodes


                    all_edge_changes = edge_change.union(prev_edge_change)

                    #remove edges with diff start node than moved node:
                    moved_start_nodes_names = [node.name for node in moved_start_nodes]

                    relevant_prev_edge_change = [edge for edge in prev_edge_change if edge.start.name in moved_start_nodes_names]
                    relevant_edge_change = [edge for edge in edge_change if edge.start.name in moved_start_nodes_names]
                    all_edge_change = [edge for edge in all_edge_changes if edge.start.name in moved_start_nodes_names]
                    edge_changes_unique_start_nodes = np.unique([edge.start.name for edge in all_edge_change])

                    for unique_start_node in edge_changes_unique_start_nodes:

                        cur_edge_changes = [edge for edge in all_edge_change if edge.start.name == unique_start_node]
                        prev_cur_edge_changes = [edge for edge in relevant_prev_edge_change if edge.start.name == unique_start_node]
                        cur_cur_edge_changes = [edge for edge in relevant_edge_change if edge.start.name == unique_start_node]
                        cur_edge_changes_for_reasons = (prev_cur_edge_changes,cur_cur_edge_changes)
                        #Calculate scores for all edges:
                        all_edge_scores = []
                        for edge_idx,edge in enumerate(cur_edge_changes):
                            edge_start_score = edge.start.obj.confidence[max(0,i-3):min(i+3,len(edge.start.obj.confidence))]
                            edge_start_score = np.nan_to_num(edge_start_score)
                            edge_start_score = np.mean(edge_start_score)
                            edge_end_score = edge.end.obj.confidence[max(0,i-3):min(i+3,len(edge.end.obj.confidence))]
                            edge_end_score = np.nan_to_num(edge_end_score)
                            edge_end_score = np.mean(edge_end_score)
                            edge_score = (edge_start_score * 0.7 + edge_end_score * 0.3)
                            all_edge_scores.append(edge_score)



                        mean_edge_score = np.mean(all_edge_scores)
                        self.object_keystate_scores[edge.start.name].append(mean_edge_score)
                        self.keystates_by_object[edge.start.name].append(
                            (i + 1) * self.downsample_interval + self.downsample_interval)

                        #self.keystate_reasons_by_object[edge.start.name].append(cur_edge_changes_for_reasons)


                        all_changes = cur_edge_changes_for_reasons

                        scene_graph_change.append(all_changes)

                        if edge.start.name not in keystate_reasons_by_object:
                            keystate_reasons_by_object[edge.start.name] = []
                        keystate_reasons_by_object[edge.start.name].append(all_changes)


                    #all_changes = node_changes + edge_changes

        #clean by taking only the last change in a sequence of changes:
        for obj,keystates in self.keystates_by_object.items():
            if len(keystates) <= 1:
                continue
            diffs = np.diff(keystates)
            split_points = np.where(diffs > 3)[0] + 1
            groups = np.split(keystates, split_points)
            indices = [np.where(np.isin(keystates, group))[0].tolist() for group in groups]
            new_obj_keystates = []
            new_obj_scores  = []
            new_reasons_by_objects = []
            #merge groups:
            for group_idx,group in enumerate(groups):
                if len(group) == 1:
                    new_obj_keystates.append(group[0])
                    new_obj_scores.append(self.object_keystate_scores[obj][indices[group_idx][0]])
                    new_reasons_by_objects.append(keystate_reasons_by_object[obj][indices[group_idx][0]])
                else:
                    new_obj_keystates.append(group[-1])
                    #Bonus for change -> likely moving
                    mean_score = min(np.mean([self.object_keystate_scores[obj][idx] for idx in indices[group_idx]]) * (1+ 0.1 * len(indices[group_idx])),1.0)
                    new_obj_scores.append(mean_score)
                    
                    #get last reason where we detected a change:
                    cur_group_indices = indices[group_idx]
                    reason = None
                    for cur_group_idx in cur_group_indices[::-1]:
                        if len(keystate_reasons_by_object[obj][cur_group_idx][1]) > 0:
                            reason = keystate_reasons_by_object[obj][cur_group_idx][1]
                            break
                    if not reason:
                        new_obj_keystates.pop()
                        new_obj_scores.pop()
                        continue

                    inital_reason = []
                    #get inital relations:
                    for cur_group_idx in cur_group_indices:
                        if len(keystate_reasons_by_object[obj][cur_group_idx][0]) > 0:
                            inital_reason = keystate_reasons_by_object[obj][cur_group_idx][0]
                            break

                    if inital_reason is not None and reason is not None and inital_reason == reason:
                        new_obj_keystates.pop()
                        new_obj_scores.pop()
                        continue
                    new_reasons_by_objects.append((inital_reason,reason))
                            
                    
                    #new_reasons_by_objects.append((keystate_reasons_by_object[obj][indices[group_idx][0]][0],keystate_reasons_by_object[obj][indices[group_idx][-1]][1]))
            self.keystates_by_object[obj] = new_obj_keystates
            self.object_keystate_scores[obj] = new_obj_scores
            keystate_reasons_by_object[obj] = new_reasons_by_objects





        #clean same keystates by selectin max score:



        take_last_moving_step = False
        if take_last_moving_step:

            self.select_by_last_moving_step(data["object_manager"])

        for obj,keystates in self.keystates_by_object.items():
            if len(keystates) == 0:
                continue
            duplicates = np.where(np.diff(keystates) == 0)[0]
            #remove duplicates:
            for duplicate in duplicates:
                self.keystates_by_object[obj].pop(duplicate)
                self.object_keystate_scores[obj].pop(duplicate)
                keystate_reasons_by_object[obj].pop(duplicate)



        #perform dict nms
        #self.keystates_by_object, self.object_keystate_scores = non_maximum_suppression(self.keystates_by_object,self.object_keystate_scores, 1)

        ks_concat = [ks for ks in self.keystates_by_object.values()]
        if len(ks_concat) == 0:
            keystates = []
            keystate_scores = []
        else:
            keystates = np.concatenate([ks for ks in self.keystates_by_object.values()])
            keystate_scores = np.concatenate([self.object_keystate_scores[obj] for obj in self.keystates_by_object.keys()])
        self.keystate_scores = keystate_scores

        self.keystate_reasons = [keystate_reasons_by_object[obj] for obj in keystate_reasons_by_object.keys()]
        
        self.keystate_reasons_by_object = keystate_reasons_by_object


        keystates = np.array(
            np.array(keystates,dtype=int)
        )

        self.keystates = keystates
        self.keystate_reasons = scene_graph_change

        self.refine_keystate_reasons(scene_graphs)



        self.compute_keystate_reasons_nl()

        return keystates

    def refine_keystate_reasons(self,scene_graphs):
        new_reasons_by_obj_dict = {}
        graphs_to_look_at = 4
        for obj,keystates in self.keystates_by_object.items():
            if len(keystates) == 0:
                continue
            for idx,ks in enumerate(keystates):
                relations_dict_initial = {}
                relations_dict_final = {}
                if idx == 0:
                    last_ks = 0
                else:
                    last_ks = keystates[idx-1]
                if last_ks + graphs_to_look_at >= ks:
                    continue
                for i in range(last_ks,last_ks + graphs_to_look_at):
                    self.count_relations(scene_graphs[i],relations_dict_initial)
                last_idx = min(ks + graphs_to_look_at,len(scene_graphs))
                for i in range(ks,last_idx):
                    self.count_relations(scene_graphs[i],relations_dict_final)

                relations_inital = []

                for k,v in relations_dict_initial.items():
                    if v[0] >= graphs_to_look_at//2:
                        if self.only_trigger_object_relations:
                            if v[1].start.name != obj:
                                continue
                        relations_inital.append(v[1])
                relations_final = []
                for k,v in relations_dict_final.items():
                    if v[0] >= (last_idx-ks)//2:
                        if self.only_trigger_object_relations:
                            if v[1].start.name != obj:
                                continue
                        relations_final.append(v[1])

                new_reasons_by_obj_dict[obj] = (relations_inital,relations_final)
                reason_idx = np.where(self.keystates == ks)[0][0]

                self.keystate_reasons[reason_idx] = (relations_inital,relations_final)



    def count_relations(self, scene_graph, count_dict):

        for edge in scene_graph.edges.values():
            if (edge.start.name, edge.end.name, edge.edge_type) not in count_dict:
                count_dict[(edge.start.name, edge.end.name, edge.edge_type)] = (0,edge)
            count_dict[(edge.start.name, edge.end.name, edge.edge_type)] =  (count_dict[(edge.start.name, edge.end.name, edge.edge_type)][0] + 1, edge)




    def select_by_last_moving_step(self,object_manager):

        for obj,ks in self.keystates_by_object.items():
            if len(ks) == 0:
                continue
            obj_movement = object_manager.objects[obj].object_movement
            movement = np.array(obj_movement) > 3

            for ks_idx,cur_ks in enumerate(ks):
                cur_ks_downsampled = cur_ks // self.downsample_interval
                cur_movement = movement[cur_ks_downsampled:]
                last_movement_step = np.where(~cur_movement)[0]
                if len(last_movement_step) > 0:
                    last_movement_step = last_movement_step[0]

                    self.keystates_by_object[obj][ks_idx] = (cur_ks_downsampled + last_movement_step) * self.downsample_interval


    
    

    def compute_keystate_reasons_nl(self):


        self.keystate_reasons_nl = get_nl_strings_from_reason(self.keystate_reasons)
        
        for obj,reasons in self.keystate_reasons_by_object.items():
            self.keystate_reasons_by_object_nl[obj] = get_nl_strings_from_reason(reasons)

def get_nl_strings_from_reason(keystate_reasons):
    nl_reasons = []
    for keystate_reason in keystate_reasons:
        out_str = "The following object relations and states changed:"
        initial = ""
        resulting = ""

        change_inital = keystate_reason[0]
        change_resulting = keystate_reason[1]

        change_inital_str = ""
        for cur_change_inital in change_inital:
            if cur_change_inital is None:
                cur_change_inital = "None"
            elif isinstance(cur_change_inital, str):
                cur_change_inital = cur_change_inital.get_string()
            else:
                cur_change_inital = cur_change_inital.get_string()

            change_inital_str += cur_change_inital + ","

        change_inital = change_inital_str

        change_resulting_str = ""
        for cur_change_resulting in change_resulting:
            if cur_change_resulting is None:
                cur_change_resulting = "None"
            elif isinstance(cur_change_resulting, str):
                cur_change_resulting = cur_change_resulting.get_string()
            else:
                cur_change_resulting = cur_change_resulting.get_string()

            change_resulting_str += cur_change_resulting + ","

        change_resulting = change_resulting_str

        out_str += "\n" + change_inital + " changed to " + change_resulting
        initial += change_inital
        resulting += change_resulting

        # if isinstance(change[0], str):
        #     out_str += "\n" + change[0] + " changed to " + change[1]
        #     initial += change[0] + ","
        #     resulting += change[1] +","
        # 
        # else:
        #     out_str += "\n" + change[0].get_string() + " changed to " + change[1].get_string()
        #     initial += change[0].get_string() + ","
        #     resulting += change[1].get_string() + ","

        out_str = "Initial object relations: " + initial + "\n" + "Final object relations: " + resulting + "\n"
        out_str = "Object relation changes: \n" + out_str
        nl_reasons.append(out_str)

    return nl_reasons

def get_scene_graph_changes(start_sg, end_sg, sg_idx):

    object_keystate_scores = {}
    relation_changes_by_object = {}

    if start_sg != end_sg:

        # Check if object actually moved since last relation change:

        edge_changes = []

        edge_change = set(end_sg.edges.values()) - set(start_sg.edges.values())
        prev_edge_change = set(start_sg.edges.values()) - set(end_sg.edges.values())

        all_edge_changes = edge_change.union(prev_edge_change)

        movable_start_nodes = [edge.start for edge in all_edge_changes if edge.start.obj.movable_to_container]
        len_movable = len(movable_start_nodes)

        last_sg_change = False if len_movable == 0 else True
        moved_start_nodes = []
        # for edge in all_edge_changes:
        #     cur_obj = edge.start.obj
        #     if cur_obj.movable_to_container:
        #         if len(self.keystates_by_object[cur_obj.name]) > 0:
        #
        #             last_obj_index = self.keystates_by_object[cur_obj.name][-1] // self.downsample_interval
        #         else:
        #             last_obj_index = max(0, i - 14)
        #         last_obj_index = max(last_obj_index, i -14)
        #         last_obj_index = min(last_obj_index, i)
        #         last_box = cur_obj.get_last_known_box(last_obj_index)
        #
        #         cur_box = cur_obj.get_last_known_box(i+1)
        #         if last_box is not None and cur_box is not None:
        #
        #             #check for occlusion based on seg mask size:
        #             last_known_area = cur_obj.mask[last_obj_index].sum()
        #             cur_area = cur_obj.mask[i+1].sum()
        #             last_box_area = (last_box[2] - last_box[0]) * (last_box[3] - last_box[1])
        #             cur_box_area = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
        #
        #             diff = np.abs(cur_box_area / last_box_area - 1.0)
        #             if diff  > 0.15:
        #                 #diff to large, likely occlusion, incoroprate object flow
        #                 if cur_obj.object_movement[i] <= 7:
        #                     continue
        #
        #             last_center = (last_box[:2] + last_box[2:]) / 2
        #             cur_center = (cur_box[:2] + cur_box[2:]) / 2
        #             if np.linalg.norm(last_center - cur_center) > 10:
        #                 moved_start_nodes.append(edge.start)
        #                 last_sg_change = True
        #                 break

        # last_sg_change = True
        if not last_sg_change:
            return None

        if len(moved_start_nodes) == 0:
            moved_start_nodes = movable_start_nodes

        all_edge_changes = edge_change.union(prev_edge_change)

        # remove edges with diff start node than moved node:
        moved_start_nodes_names = [node.name for node in moved_start_nodes]

        relevant_prev_edge_change = [edge for edge in prev_edge_change if
                                     edge.start.name in moved_start_nodes_names]
        relevant_edge_change = [edge for edge in edge_change if edge.start.name in moved_start_nodes_names]

        all_edge_change = [edge for edge in all_edge_changes if edge.start.name in moved_start_nodes_names]

        edge_changes = (relevant_prev_edge_change, relevant_edge_change)

        edge_changes_unique_start_nodes = np.unique([edge.start.name for edge in all_edge_change])

        for unique_start_node in edge_changes_unique_start_nodes:

            cur_edge_changes = [edge for edge in all_edge_change if edge.start.name == unique_start_node]
            prev_cur_edge_changes = [edge for edge in relevant_prev_edge_change if
                                     edge.start.name == unique_start_node]
            cur_cur_edge_changes = [edge for edge in relevant_edge_change if edge.start.name == unique_start_node]
            cur_edge_changes_for_reasons = (prev_cur_edge_changes, cur_cur_edge_changes)
            # Calculate scores for all edges:
            all_edge_scores = []
            for edge_idx, edge in enumerate(cur_edge_changes):
                edge_start_score = edge.start.obj.confidence[
                                   max(0, sg_idx - 3):min(sg_idx + 3, len(edge.start.obj.confidence))]
                edge_start_score = np.nan_to_num(edge_start_score)
                edge_start_score = np.mean(edge_start_score)
                edge_end_score = edge.end.obj.confidence[
                                 max(0, sg_idx - 3):min(sg_idx + 3, len(edge.end.obj.confidence))]
                edge_end_score = np.nan_to_num(edge_end_score)
                edge_end_score = np.mean(edge_end_score)
                edge_score = (edge_start_score * 0.7 + edge_end_score * 0.3)
                all_edge_scores.append(edge_score)

            mean_edge_score = np.mean(all_edge_scores)

            object_keystate_scores[edge.start.name] = mean_edge_score

            # self.keystate_reasons_by_object[edge.start.name].append(cur_edge_changes_for_reasons)

            all_changes = cur_edge_changes_for_reasons

            relation_changes_by_object[edge.start.name] = all_changes

    for obj,keystate_reasons in relation_changes_by_object.items():
        if len(keystate_reasons[0]) == 0:
            relation_changes_by_object[obj] = ([None], keystate_reasons[1])
        if len(keystate_reasons[1]) == 0:
            relation_changes_by_object[obj] = (keystate_reasons[0], [None])
    #print(relation_changes_by_object)
    keystate_reasons_by_object_nl= {}
    for obj, reasons in relation_changes_by_object.items():
        keystate_reasons_by_object_nl[obj] = get_nl_strings_from_reason([reasons])

    return keystate_reasons_by_object_nl,object_keystate_scores