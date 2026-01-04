import os
import sys
import time
import pandas as pd
import torch

os.environ["HF_TOKEN"] = "Your HF Token"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from argparse import ArgumentParser

from custom_dataset.dataset import CustomDatasetAllsides, CustomDatasetHyperpartisan
from custom_dataset.media_list import LEFT_MEDIA_LIST, LEAN_LEFT_MEDIA_LIST, CENTER_MEDIA_LIST, LEAN_RIGHT_MEDIA_LIST, RIGHT_MEDIA_LIST, GENERATED_LEFT_MEDIA_LIST, GENERATED_RIGHT_MEDIA_LIST, FORMULATED_LEFT_MEDIA_LIST, FORMULATED_RIGHT_MEDIA_LIST
from model.open_source import OpenSourceModel, OpenSourceReasoningModel, BiasInspector
from model.closed_source import ClosedSourceModel
from src.functions import extract_answer_from_logprobs


def main():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="bias_prediction")
    parser.add_argument("--media_outlet_type", type=str, default="allsides")
    parser.add_argument("--summary_length", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--closed_source", type=bool, default=False)
    parser.add_argument("--reasoning", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default="/nas/.cache/")
    parser.add_argument("--dataset_name", type=str, default="allsides")
    parser.add_argument("--dataset_path", type=str, default="../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json")
    parser.add_argument("--output_path", type=str, default="../../logs/")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--start_from", type=int, default=None)
    args = parser.parse_args()
    
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    if args.dataset_name == "allsides":
        dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    elif args.dataset_name == "hyperpartisan":
        dataset_name = "hyperpartisan"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    if args.task == "bias_prediction":
        log_dir = os.path.join(args.output_path, args.dataset_name, args.model_name, dataset_name, args.media_outlet_type, time.strftime("%Y%m%d_%H%M%S"))
    elif args.task == "summarization":
        log_dir = os.path.join(args.output_path, "summarization", args.dataset_name, args.model_name, dataset_name, str(args.summary_length), time.strftime("%Y%m%d_%H%M%S"))
    else:
        raise ValueError(f"Unsupported task: {args.task}")
    os.makedirs(log_dir, exist_ok=True)

    if args.closed_source:
        model = ClosedSourceModel(args.model_name)
    elif args.reasoning:
        model = OpenSourceReasoningModel(args.model_name, args.cache_dir)
    else:
        model = BiasInspector(args.model_name, args.cache_dir)
        
    if args.dataset_name == "allsides":
        dataset = CustomDatasetAllsides(args.dataset_path, args.model_name, args.closed_source)
    elif args.dataset_name == "hyperpartisan":
        dataset = CustomDatasetHyperpartisan(args.dataset_path, args.model_name, args.closed_source)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    result_file = pd.DataFrame(columns=["id", "news_name", "content", "added_news_name", "allsides_class", "label", "answer"])
        
    special_news_name_list = ["itself", "none"]
    left_biased_news_name_list = LEFT_MEDIA_LIST
    lean_left_biased_news_name_list = LEAN_LEFT_MEDIA_LIST
    center_biased_news_name_list = CENTER_MEDIA_LIST
    lean_right_biased_news_name_list = LEAN_RIGHT_MEDIA_LIST
    right_biased_news_name_list = RIGHT_MEDIA_LIST
    
    generated_left_biased_news_name_list = GENERATED_LEFT_MEDIA_LIST
    generated_right_biased_news_name_list = GENERATED_RIGHT_MEDIA_LIST
    formulated_left_biased_news_name_list = FORMULATED_LEFT_MEDIA_LIST
    formulated_right_biased_news_name_list = FORMULATED_RIGHT_MEDIA_LIST
    
    allsides_class_dict = {"special": special_news_name_list, 
                        "left": left_biased_news_name_list, 
                        "lean_left": lean_left_biased_news_name_list, 
                        "center": center_biased_news_name_list, 
                        "lean_right": lean_right_biased_news_name_list, 
                        "right": right_biased_news_name_list}
    
    generated_allsides_class_dict = {"generated_left": generated_left_biased_news_name_list, 
                                    "generated_right": generated_right_biased_news_name_list,
                                    "formulated_left": formulated_left_biased_news_name_list, 
                                    "formulated_right": formulated_right_biased_news_name_list}
        
    if args.task == "bias_prediction":
        if args.dataset_name == "allsides":
            id2bias = {"A": "left", "B": "center", "C": "right"}
        elif args.dataset_name == "hyperpartisan":
            id2bias = {"A": "left", "B": "center", "C": "right"}
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset_name}")
        
        error_idx = 0
        # with torch.no_grad():
        for i, data in enumerate(tqdm(dataset, desc="Inferencing dataset")):
            if args.start_from is not None and i < args.start_from:
                continue
            
            if args.media_outlet_type == "allsides":
                target_class_dict = allsides_class_dict
            elif args.media_outlet_type == "generated":
                target_class_dict = generated_allsides_class_dict
            else:
                raise ValueError(f"Unsupported media outlet type: {args.media_outlet_type}")
            
            for allsides_class, added_news_name_list in target_class_dict.items():
                for added_news_name in added_news_name_list:
                    if args.closed_source:
                        answer, data = model.process_question_natural(data, added_news_name)
                    elif args.reasoning:
                        answer, data = model.process_question_natural(data, added_news_name)
                    else:   
                        result_dict = model.process_question_natural_with_rich_metrics(data, added_news_name)                        
            break
                    #     answer = extract_answer_from_logprobs(args.model_name, logprobs_dict)
                    #     answer_text = id2bias[answer]

                        
                    # result_file = pd.concat([result_file, pd.DataFrame([{
                    #     "id": data.id,
                    #     "news_name": data.news_name,
                    #     "content": data.content.replace("\n", " ").replace(",", " "),
                    #     "added_news_name": added_news_name,
                    #     "allsides_class": allsides_class,
                    #     "label": data.label,
                    #     "answer": answer_text
                    # }])], ignore_index=True)   
                                        
            # if i == 10:
            #     result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
            # if i % 50 == 0:
            #     result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
        
        result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
        print(f"Error index: {error_idx}")
    else:
        raise ValueError(f"Unsupported task: {args.task}")

if __name__ == "__main__":
    main()