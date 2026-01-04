import os
import sys
import json
import torch
import random

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass

from media_list import LEFT_MEDIA_LIST, LEAN_LEFT_MEDIA_LIST, CENTER_MEDIA_LIST, LEAN_RIGHT_MEDIA_LIST, RIGHT_MEDIA_LIST


def load_custom_dataset(dataset_path: str):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data


def analyze_label_distribution(dataset_path: str):
    source_dict = {"left": {}, "center": {}, "right": {}, "else": {}}
    data = load_custom_dataset(dataset_path)
    for item in tqdm(data, desc="Analyzing label distribution"):
        source = item['source']
        
        if source not in source_dict['left'].keys() and source not in source_dict['center'].keys() and source not in source_dict['right'].keys() and source not in source_dict['else'].keys():            
            if source in LEFT_MEDIA_LIST or source in LEAN_LEFT_MEDIA_LIST:
                source_dict["left"][source] = 1
            elif source in CENTER_MEDIA_LIST:
                source_dict["center"][source] = 1
            elif source in RIGHT_MEDIA_LIST or source in LEAN_RIGHT_MEDIA_LIST:
                source_dict["right"][source] = 1
            else:
                source_dict["else"][source] = 1
        else:
            if source in LEFT_MEDIA_LIST or source in LEAN_LEFT_MEDIA_LIST:
                source_dict["left"][source] += 1
            elif source in CENTER_MEDIA_LIST:
                source_dict["center"][source] += 1
            elif source in RIGHT_MEDIA_LIST or source in LEAN_RIGHT_MEDIA_LIST:
                source_dict["right"][source] += 1
            else:
                source_dict["else"][source] += 1
                
    print(source_dict)
    
    
if __name__ == "__main__":
    print("Known medias")
    analyze_label_distribution("../../data/allsides/Article-Bias-Prediction/data/custom-split/known_medias.json")
    
    print("Random medias")
    analyze_label_distribution("../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json")
    
    print("Exclude known medias")
    analyze_label_distribution("../../data/allsides/Article-Bias-Prediction/data/custom-split/exclude_known_medias.json")
    