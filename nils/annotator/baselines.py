import difflib
import time

import numpy as np
import torch
from tqdm import tqdm

from nils.utils.plot import annotate_frame


def get_grounding_predictions_baseline_vlm(cfg, object_manager, vlm, keystates, batch, possible_actions,
                                           interacted_object,
                                           keystate_reasons, open_ended=False):
    batch_keystates = []
    batch_pred_actions = []
    local_pred_indices = []
    scores = []
    llm_responses = []
    for keystate_idx, ks in enumerate(tqdm(keystates)):

        ks = min(ks, len(batch["frame_name"]) - 1)

        if keystate_idx == 0:
            last_keystate_idx = 0
        if ks < cfg.prompt_interval:
            last_keystate_idx = ks
            continue
        length = ks - last_keystate_idx
        length = max(cfg.prompt_interval * 2, length)
        length = min(8, length)
        start_idx = max(0, ks - length)

        image_final_index = min(ks, len(batch["frame_name"]) - 1)
        images_keystate = batch["rgb_static"][start_idx:image_final_index]

        # annotate keystate images:

        images_keystate_annotated = []

        if keystate_reasons is not None:
            keystate_reasons_nl = [reason[0] for reason in keystate_reasons.values() if reason[0] is not None]
            keystate_reasons_nl = "\n".join(keystate_reasons_nl)
        else:
            keystate_reasons_nl = ""

        for idx, image in zip(range(start_idx, image_final_index), images_keystate):
            if cfg.som_vlm:
                object_stats_frame = object_manager.get_object_stats_frame(
                    idx // cfg.scene_graph_interval)
                image_annotated = annotate_frame(image, object_stats_frame)

            else:
                image_annotated = image
            images_keystate_annotated.append(image_annotated)
        images_keystate_annotated = np.array(images_keystate_annotated)

        success = False
        n_retries = 2
        current_try = 0
        vlm_response = None
        while not success and current_try < n_retries:

            try:
                vlm_response, score = vlm.get_action(images_keystate_annotated, possible_actions,
                                                     "None", cfg.sot_vlm,
                                                     observations=keystate_reasons_nl, open_ended=open_ended)
                llm_responses.append(vlm_response)
                success = True
            except Exception as e:
                print(e)
                current_try += 1
                time.sleep(60)
                print(f"Retrying {current_try}/{n_retries}")

        if vlm_response is None:
            batch_keystates.append(ks)
            batch_pred_actions.append("error")
            local_pred_indices.append(ks)
            llm_responses.append([])
            scores.append(0)
            continue
        if vlm_response in possible_actions:
            matching_task = vlm_response
        else:
            scores = []
            for a in possible_actions:
                match_score = difflib.SequenceMatcher(None, vlm_response, a).ratio()

            matching_task = possible_actions[np.argmax(match_score)]

        batch_keystates.append(ks)
        batch_pred_actions.append(matching_task)
        local_pred_indices.append(ks)
        scores.append(score)

    return llm_responses, scores


def get_grounding_prediction_baseline_action_recognition(cfg, action_predictor, keystates, batch, possible_actions,
                                                         plot=False):
    batch_keystates = []
    batch_pred_actions = []
    local_pred_indices = []

    scores = []

    for keystate_idx, ks in enumerate(tqdm(keystates)):
        ks = min(ks, len(batch["frame_name"]) - 1)
        if keystate_idx == 0:
            last_keystate_idx = 0
        if ks < cfg.prompt_interval:
            last_keystate_idx = ks
            continue
        length = ks - last_keystate_idx
        length = max(cfg.prompt_interval * 2, length)
        length = min(20, length)
        start_idx = max(0, ks - length)

        images_keystate = batch["rgb_static"][start_idx:ks]

        score_probs = action_predictor.predict(images_keystate, possible_actions)

        predicted_action = np.argmax(score_probs)
        predicted_action_nl = possible_actions[predicted_action]
        top_score = np.max(score_probs)

        del score_probs
        torch.cuda.empty_cache()

        # save_path = f"{self.action_predictor.name}_{ks}.png"
        if plot:
            save_path = None
            action_predictor.visualize(images_keystate, predicted_action_nl, save_path=save_path)

        batch_keystates.append(ks)
        batch_pred_actions.append(predicted_action_nl)
        local_pred_indices.append(ks)
        scores.append(top_score)

    return batch_pred_actions, scores
