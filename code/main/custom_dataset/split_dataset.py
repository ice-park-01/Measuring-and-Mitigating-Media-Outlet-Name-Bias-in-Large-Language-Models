import os
import sys
import json
import torch
import random

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass

from prompt import BASE_PROMPT_ALLSIDES, COT_PROMPT_ALLSIDES
from media_list import LEFT_MEDIA_LIST, LEAN_LEFT_MEDIA_LIST, CENTER_MEDIA_LIST, LEAN_RIGHT_MEDIA_LIST, RIGHT_MEDIA_LIST


def load_dataset_allsides(dataset_path):
    raw_data = []
    files = os.listdir(dataset_path)
    for file in tqdm(files, desc="Loading dataset"):
        with open(os.path.join(dataset_path, file), "r") as f:
            data = json.load(f)
            raw_data.append(data)
    return raw_data

                
def split_dataset(dataset_path: str):
    total_media_list = LEFT_MEDIA_LIST + LEAN_LEFT_MEDIA_LIST + CENTER_MEDIA_LIST + LEAN_RIGHT_MEDIA_LIST + RIGHT_MEDIA_LIST
    raw_data = load_dataset_allsides(dataset_path)
    left_data, center_data, right_data = [], [], []
    
    left_mismatch_count = {"left": 0, "center": 0, "right": 0}
    center_mismatch_count = {"left": 0, "center": 0, "right": 0}
    right_mismatch_count = {"left": 0, "center": 0, "right": 0}
    
    for data in tqdm(raw_data, desc="Splitting dataset"):
        source = data['source']
        bias = data['bias_text']
        
        if source in total_media_list:
            if (source in LEFT_MEDIA_LIST or source in LEAN_LEFT_MEDIA_LIST):
                if bias.lower() == "left":
                    left_data.append(data)
                else:
                    original_bias = bias.lower().strip()
                    left_mismatch_count[original_bias] += 1
            elif source in CENTER_MEDIA_LIST:
                if bias.lower() == "center":
                    center_data.append(data)
                else:
                    original_bias = bias.lower().strip()
                    center_mismatch_count[original_bias] += 1
            elif (source in LEAN_RIGHT_MEDIA_LIST or source in RIGHT_MEDIA_LIST):
                if bias.lower() == "right":
                    right_data.append(data)
                else:
                    original_bias = bias.lower().strip()
                    right_mismatch_count[original_bias] += 1
                
    print(f"Left data: {len(left_data)}")
    print(f"Center data: {len(center_data)}")
    print(f"Right data: {len(right_data)}")
    
    print(f"Left mismatch count: {left_mismatch_count}")
    print(f"Center mismatch count: {center_mismatch_count}")
    print(f"Right mismatch count: {right_mismatch_count}")
    
    return left_data, center_data, right_data


def split_dataset_by_random(dataset_path: str):
    raw_data = load_dataset_allsides(dataset_path)
    left_data, center_data, right_data = [], [], []
    
    for data in tqdm(raw_data, desc="Splitting dataset"):
        bias = data['bias_text']
        if bias.lower() == "left":
            left_data.append(data)
        elif bias.lower() == "center":
            center_data.append(data)
        elif bias.lower() == "right":
            right_data.append(data)
    
    left_data = random.sample(left_data, min(len(left_data), 10))
    center_data = random.sample(center_data, min(len(center_data), 10))
    right_data = random.sample(right_data, min(len(right_data), 10))
    
    print(f"Left data: {len(left_data)}")
    print(f"Center data: {len(center_data)}")
    print(f"Right data: {len(right_data)}")
    
    return left_data, center_data, right_data


def split_dataset_exclude_known_medias(dataset_path: str):
    total_media_list = LEFT_MEDIA_LIST + LEAN_LEFT_MEDIA_LIST + CENTER_MEDIA_LIST + LEAN_RIGHT_MEDIA_LIST + RIGHT_MEDIA_LIST
    raw_data = load_dataset_allsides(dataset_path)
    left_data, center_data, right_data = [], [], []
    
    for data in tqdm(raw_data, desc="Splitting dataset"):
        bias = data['bias_text']
        source = data['source']
        
        if bias.lower() == "left":
            if source in total_media_list:
                continue
            else:
                left_data.append(data)
        elif bias.lower() == "center":
            if source in total_media_list:
                continue
            else:
                center_data.append(data)
        elif bias.lower() == "right":
            if source in total_media_list:
                continue
            else:
                right_data.append(data)

    left_data = random.sample(left_data, min(len(left_data), 1500))
    center_data = random.sample(center_data, min(len(center_data), 1500))
    right_data = random.sample(right_data, min(len(right_data), 1500))
        
    print(f"Left data: {len(left_data)}")
    print(f"Center data: {len(center_data)}")
    print(f"Right data: {len(right_data)}")
    
    return left_data, center_data, right_data


if __name__ == "__main__":
    # left_data, center_data, right_data = split_dataset("../../data/allsides/Article-Bias-Prediction/data/jsons/")
    
    # os.makedirs("../../data/allsides/Article-Bias-Prediction/data/custom-split", exist_ok=True)
    # with open("../../data/allsides/Article-Bias-Prediction/data/custom-split/known_medias.json", "w") as f:
    #     json.dump(left_data + center_data + right_data, f, indent=4)

    left_data, center_data, right_data = split_dataset_by_random("../../data/allsides/Article-Bias-Prediction/data/jsons/")
    
    # with open("../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json", "w") as f:
    #     json.dump(left_data + center_data + right_data, f, indent=4)
    with open("../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization_rvt_v2.json", "w") as f:
        json.dump(left_data + center_data + right_data, f, indent=4)
        
    # left_data, center_data, right_data = split_dataset_exclude_known_medias("../../data/allsides/Article-Bias-Prediction/data/jsons/")
    
    # with open("../../data/allsides/Article-Bias-Prediction/data/custom-split/exclude_known_medias.json", "w") as f:
    #     json.dump(left_data + center_data + right_data, f, indent=4)