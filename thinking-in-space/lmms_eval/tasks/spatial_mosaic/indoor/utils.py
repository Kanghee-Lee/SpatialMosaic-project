import os
import re
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Any

import datasets


MCA_QUESTION_TYPES = [
    "bestview_count_occ",
    "obj_attribute_occ_ab",
    "obj_attribute_occ_fb",
    "obj_attribute_occ_lr",
    "obj_count_occ",
    "obj_existence_occ_ab",
    "obj_existence_occ_fb",
    "obj_existence_occ_lr",
    "obj_localization_occ",
    "obj_spatial_occ_ab",
    "obj_spatial_occ_fb",
    "obj_spatial_occ_lr"
]
NA_QUESTION_TYPES = [
    "obj_count_occ_na",
    "obj_distance_occ_na",
    "obj_obj_distance_occ_na",
    "obj_size_occ_na",
]

NA_LOCAL_QUESTION_TYPES = [
    "obj_localization_occ_na",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


METRICS_FOR_NA_LOCAL = {
    "MPA:.5:1.4:.1": "partial(mean_point_accuracy, start=.5, end=1.4, interval=.1)",
}
FRAMES_ROOT = 'path_to_img'
FRAMES_EXT  = '.jpg'   

def _resolve_frame_paths(doc: Dict[str, Any]) -> List[str]:

    frame_tokens = doc.get("frames", [])
    paths = []
    for token in frame_tokens:
        if "/" in token or token.endswith((".jpg", ".jpeg", ".png")):
            paths.append(token if os.path.isabs(token) else os.path.join(FRAMES_ROOT, token))
        else:
            scene = doc.get("scene_name", "")
            paths.append(os.path.join(FRAMES_ROOT, scene, 'images', f"{token}{FRAMES_EXT}"))
    return paths

## vsti, llava_vid,llava_onevision
def vsibench_doc_to_visual(doc):
    imgs = []
    for p in _resolve_frame_paths(doc):
        with Image.open(p) as im:
            imgs.append(np.asanyarray(im))
    images = np.stack(imgs)
    return images


# internvl2, longva, gemini
# def vsibench_doc_to_visual(doc):
#     imgs = []
#     for p in _resolve_frame_paths(doc):
#         with Image.open(p).convert("RGB") as im:
#             imgs.append(im)
#     # images = np.stack(imgs)
#     return imgs



def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    if doc['question_type'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    if doc['question_type'] in NA_LOCAL_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_local_post_prompt", "") or "Please answer with the coordinate pair only, formatted as (x,y)."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    if doc['question_type'] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # # 筛选 camera_movement_direction类型的问题
    # dataset = dataset.filter(lambda x: x['question_type'] in ['camera_movement_direction'])
    
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

def point_dist(pred, target):
    return np.sqrt((pred[0] - target[0]) ** 2 + (pred[1] - target[1]) ** 2)

def mean_point_accuracy(pred, target, bbox_2d_diag, start, end, interval):
    thresholds = np.arange(start, end + interval / 2, interval)
    accuracy = point_dist(pred, target) <= bbox_2d_diag * thresholds
    return accuracy.mean()

# WORST_CASE_FOR_METRICS = {
#     "accuracy": 0.,
#     "MRA:.5:.95:.05": 0.,
#     "MPA:.05:.25:.05": 0.,
# }

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
    "MPA:.5:1.4:.1" : 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def to_positive_float(pred):
    pred = to_float(pred)
    if pred is None or pred <= 0:
        return None
    return pred

def to_point(pred):
    try:
        pred = str(pred)
        match = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", pred)
        if match is None:
            match = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", pred)
        if match is not None:
            return float(match.group(1)), float(match.group(2))

        nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    except BaseException as e:
        pass
    return None

def vsibench_process_results(doc, results):
    
    doc['prediction'] = results[0]
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['mc_answer'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    elif doc['question_type'] in NA_LOCAL_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA_LOCAL.items():
            try:
                doc[key] = eval(value)(to_point(doc['prediction']), to_point(doc['ground_truth']), to_positive_float(doc['bbox_2d_diag']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_LOCAL_QUESTION_TYPES:
            for metric in METRICS_FOR_NA_LOCAL.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.
