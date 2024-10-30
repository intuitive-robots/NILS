import numpy as np


def strict_accuracy(batch, scores):
    """Compares the predicted labels at keyframe indices to predicted action at keyframe
    :param batch:
    :param scores:
    :return:
    """
    key_state_labels = np.array(batch["key_state_labels"])
    predicted_labels = np.argmax(scores[batch["key_state_indices"]], axis=-1)

    assert key_state_labels.shape == predicted_labels.shape

    acc = (predicted_labels == np.array(key_state_labels)).sum() / key_state_labels.shape[0]
    return acc


def average_length_accuracy(batch, scores):
    """Computes the average predicted action over the span of the subgoal and compares it to the ground truth action
    :param batch:
    :param scores:
    :return:
    """
    key_state_indices = batch["key_state_indices"]
    key_state_labels = np.array(batch["key_state_labels"])
    split_array = np.split(scores,np.array(key_state_indices))

    mean_classes_labels = np.array([np.argmax(np.bincount(np.argmax(splt, axis=-1))) for splt in split_array])
    for splt in split_array:
          mean_class = np.argmax(np.bincount(np.argmax(splt, axis=-1)))

    if key_state_labels.shape[0] < mean_class.shape[0]:
        mean_classes_labels = mean_class[:-1]
    acc = (key_state_labels == mean_classes_labels).sum() / key_state_labels.shape[0]

    return acc


def language_perturbed_accuracy(batch, scores, n_paraphrases=2):
    """Measures robustness to perturbations in the language goal. Generates small perturbations and evaluates model
    performance under this metrics. Maybe compute variance for this?
    :param batch:
    :param scores:
    :param n_paraphrases:
    :return:
    """


def get_running_metrics(cur_dict):
    running_precision = cur_dict["TP"] / (cur_dict["TP"] + cur_dict["FP"]) if cur_dict["TP"] + cur_dict[
        "FP"] > 0 else 0
    running_recall = cur_dict["TP"] / (cur_dict["TP"] + cur_dict["FN"]) if cur_dict["TP"] + cur_dict[
        "FN"] > 0 else 0
    running_accuracy = cur_dict["TP"] / (
            cur_dict["TP"] + cur_dict["FP"] + cur_dict["FN"]) if cur_dict["TP"] + cur_dict["FP"] + cur_dict[
        "FN"] > 0 else 0
    running_f1 = 2 * (running_precision * running_recall) / (
            running_precision + running_recall) if running_precision + running_recall > 0 else 0
    
    running_tp_lang = cur_dict["TP_LANG"]
    running_fp_lang = cur_dict["FP_LANG"]

    running_lang_accuracy = cur_dict["TP_LANG"] / (
            cur_dict["TP_LANG"] + cur_dict["FP_LANG"]) if cur_dict["TP_LANG"] + cur_dict["FP_LANG"]  > 0 else 0


    return running_precision, running_recall, running_accuracy, running_f1,running_tp_lang,running_fp_lang,running_lang_accuracy


def update_metrics(metrics_dict, metrics):
    for metric_eps, values in metrics.items():
        if metric_eps not in metrics_dict.keys():
            metrics_dict[metric_eps] = {}
            metrics_dict[metric_eps]["TP"] = 0
            metrics_dict[metric_eps]["FP"] = 0
            metrics_dict[metric_eps]["FN"] = 0
            metrics_dict[metric_eps]["TP_LANG"] = 0
            metrics_dict[metric_eps]["FP_LANG"] = 0
            metrics_dict[metric_eps]["wrong_pred_lang"] = []

        metrics_dict[metric_eps]["TP"] += values["TP"]
        metrics_dict[metric_eps]["FP"] += values["FP"]
        metrics_dict[metric_eps]["FN"] += values["FN"]
        metrics_dict[metric_eps]["TP_LANG"] += values["TP_LANG"]
        metrics_dict[metric_eps]["FP_LANG"] += values["FP_LANG"]
        metrics_dict[metric_eps]["wrong_pred_lang"].extend(values["wrong_pred_lang"])
        



    return metrics_dict



