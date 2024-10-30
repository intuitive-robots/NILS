import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

method_weights_hc = {

    "object_movement": 0.4,
    "state_change": 0.8,
    "scene_graph_change": 1,
    "gripper_close": 0.6
}


def score_keystates(keystates, object_keystate_scores, match_threshold, method_weights=None,
                    use_object_specific_voting=False):
    if method_weights is None:
        method_weights = method_weights_hc

    scored_keystates = []
    averaged_keystates = []
    scores = []

    use_static_score = True

    all_weights = np.sum([method_weights[method] for method in keystates.keys()])

    for cur_method, cur_method_keystates in keystates.items():

        #other_keystates = [ks for method,ks in keystates.items() if method != cur_method]
        #other_keystates = np.stack(other_keystates)
        s = 1
        for keystate in cur_method_keystates:
            if use_static_score:
                cur_keystate_scores = [method_weights[cur_method]]
            else:

                cur_keystate_scores = [method_weights[cur_method] * object_keystate_scores[cur_method][
                    cur_method_keystates.index(keystate)]]

            cur_n_predictor_scores = 1
            keystates_to_average = [keystate]
            for other_method, other_keystates in keystates.items():
                if other_method == cur_method:
                    continue
                if len(other_keystates) == 0:
                    continue

                dist = np.abs(keystate - np.array(other_keystates))
                min_dist_idx = np.argmin(dist)
                min_dist = dist[min_dist_idx]

                if "object_state" in other_method and min_dist > (match_threshold - 1):
                    dist = (keystate - np.array(other_keystates))
                    # more tolreance for state change
                    neg_dists = dist[dist <= 0]
                    if len(neg_dists > 0):

                        min_dist_bwd = np.argmax(neg_dists)
                        match_threshold_bwd = -20
                        min_dist_bwd_val = neg_dists[min_dist_bwd]
                        if min_dist_bwd_val > match_threshold_bwd:
                            min_dist_idx = np.where(dist == min_dist_bwd_val)[0][0]
                            if use_static_score:
                                score = method_weights[other_method]
                            else:
                                score = method_weights[other_method] * object_keystate_scores[other_method][
                                    min_dist_idx]
                            #keystates_to_average.append(other_keystates[min_dist_idx])
                            cur_keystate_scores.append(score)
                            cur_n_predictor_scores += method_weights[other_method]
                            continue

                if min_dist < match_threshold:
                    #print(min_dist)
                    if use_static_score:
                        score = method_weights[other_method]
                    else:
                        score = method_weights[other_method] * object_keystate_scores[other_method][min_dist_idx]

                    keystates_to_average.append(other_keystates[min_dist_idx])
                    cur_keystate_scores.append(score)
                    cur_n_predictor_scores += method_weights[other_method]

            cur_keystate_scores = np.sum(cur_keystate_scores)
            cur_n_predictor_scores /= all_weights

            combined_score = cur_keystate_scores * 0.5 + cur_n_predictor_scores * 0.5
            if use_static_score:
                combined_score = cur_keystate_scores / all_weights
            else:
                combined_score = min(cur_keystate_scores / all_weights + cur_n_predictor_scores * 0.4, 1.)

            if len(keystates_to_average) == 0:
                averaged_keystates.append(keystate)
            else:
                #print(keystates_to_average)
                #print(keystates_to_average)
                averaged_keystate = np.mean(keystates_to_average).astype(int)
                #print(averaged_keystate)
                averaged_keystates.append(averaged_keystate)
            scores.append(combined_score)
            scored_keystates.append(keystate)

    return averaged_keystates, scores
