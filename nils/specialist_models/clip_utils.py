import os

import torch
from tqdm import tqdm

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]
#

def split_labels(x):
    res = []
    for x_ in x:
        x_ = x_.replace(', ', ',')
        x_ = x_.split(',') # there can be multiple synonyms for single class
        res.append(x_)
    return res


def fill_all_templates_ensemble(x_=''):
    res = []
    for x in x_:
        for template in VILD_PROMPT:
            res.append(template.format(x))
    return res, len(res) // len(VILD_PROMPT)

def get_text_classifier(text_embedder,classes,cache_dir):
    
    
    all_text_embedding_cache = {}
    # This consists of COCO, ADE20K, LVIS
    if os.path.exists(cache_dir):
        # key: str of class name, value: tensor in shape of C
        all_text_embedding_cache = torch.load(cache_dir)
        

        
        
    nontemplated_class_names = split_labels(classes)
    print("nontemplated_class_names:", nontemplated_class_names)
    text2classifier = {}
    test_class_names = []
    uncached_class_name = []
    text_classifier = []
    
    
    
    # exclude those already in cache
    for class_names in nontemplated_class_names:
        if not isinstance(class_names, list):
            class_names = [class_names]
        for class_name in class_names:
            if class_name in all_text_embedding_cache:
                text2classifier[class_name] = all_text_embedding_cache[class_name]
            else:
                test_class_names += fill_all_templates_ensemble([class_name])[0]
                uncached_class_name.append(class_name)
    print("Uncached texts:", len(uncached_class_name), uncached_class_name, test_class_names)
    # this is needed to avoid oom, which may happen when num of class is large
    bs = 128
    for idx in tqdm(range(0, len(test_class_names), bs), desc="Embedding texts", disable=True):
        text_classifier.append(text_embedder.get_text_embeddings(test_class_names[idx:idx+bs]))

    if len(text_classifier) > 0:
        text_classifier = torch.cat(text_classifier, dim=0)
        # average across templates and normalization.
        text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
        text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
        text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
        assert text_classifier.shape[0] == len(uncached_class_name)
        for idx in range(len(uncached_class_name)):
            all_text_embedding_cache[uncached_class_name[idx]] = text_classifier[idx]
            text2classifier[uncached_class_name[idx]] = text_classifier[idx]
        torch.save({k:v for k, v in all_text_embedding_cache.items()}, cache_dir)

    text_classifier = []
    for class_names in nontemplated_class_names:
        for text in class_names:
            text_classifier.append(text2classifier[text])
    text_classifier = torch.stack(text_classifier, dim=0)
    
    return text_classifier


def split_label(x: str):
    x = x.replace('_or_', ',')
    x = x.replace('/', ',')
    x = x.replace('_', ' ')
    x = x.lower()
    x = x.split(',')
    x = [_x.strip() for _x in x]
    return x


def get_text_classifier_ov_sam(text_embedder,classes,cache_dir):
    
    
    
    descriptions = []
    candidates = []
    for cls_name in classes:
        labels_per_cls = split_label(cls_name)
        candidates.append(len(labels_per_cls))
        for label in labels_per_cls:
            for template in VILD_PROMPT:
                description = template.format(label)
                descriptions.append(description)


    bsz = 256
    NUM_BATCH = len(descriptions) // bsz

    NUM_BATCH = max(1, NUM_BATCH)
    
    
    bs = len(descriptions)
    local_bs = bs // NUM_BATCH
    if bs % NUM_BATCH != 0:
        local_bs += 1
    feat_list = []
    for i in tqdm(range(NUM_BATCH), desc="Embedding texts", disable=True):
        local_descriptions = descriptions[i * local_bs: (i + 1) * local_bs]
        local_feat = text_embedder.get_text_embeddings(local_descriptions).to(device='cpu')
        feat_list.append(local_feat)
    features = torch.cat(feat_list)


    dim = features.shape[-1]
    candidate_tot = sum(candidates)
    candidate_max = max(candidates)
    features = features.reshape(candidate_tot, len(VILD_PROMPT), dim)
    features = features / features.norm(dim=-1, keepdim=True)
    features = features.mean(dim=1, keepdims=False)
    features = features / features.norm(dim=-1, keepdim=True)

    cur_pos = 0
    classifier = []
    for candidate in candidates:
        cur_feat = features[cur_pos:cur_pos + candidate]
        if candidate < candidate_max:
            cur_feat = torch.cat([cur_feat, cur_feat[0].repeat(candidate_max - candidate, 1)])
        classifier.append(cur_feat)
        cur_pos += candidate
    classifier = torch.stack(classifier)

    if cache_dir is None:
        return classifier
    save_path = cache_dir
    classifier_to_save = classifier
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(classifier_to_save, save_path)
    
    
    return classifier_to_save
