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
from model.open_source import OpenSourceModel, OpenSourceReasoningModel
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
    elif args.task == "bias_prediction_for_order":
        log_dir = os.path.join(args.output_path, "bias_prediction_for_order", args.dataset_name, args.model_name, dataset_name, time.strftime("%Y%m%d_%H%M%S"))
    elif args.task == "xai":
        log_dir = os.path.join(args.output_path, "xai", args.dataset_name, args.model_name, dataset_name, time.strftime("%Y%m%d_%H%M%S"))
    else:
        raise ValueError(f"Unsupported task: {args.task}")
    os.makedirs(log_dir, exist_ok=True)

    if args.closed_source:
        model = ClosedSourceModel(args.model_name)
    elif args.reasoning:
        model = OpenSourceReasoningModel(args.model_name, args.cache_dir)
    else:
        model = OpenSourceModel(args.model_name, args.cache_dir)
        
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
        with torch.no_grad():
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
                        try:
                            if args.closed_source:
                                answer, data = model.process_question_natural(data, added_news_name)
                            elif args.reasoning:
                                answer, data = model.process_question_natural(data, added_news_name)
                            else:   
                                logprobs_dict, data = model.process_question_natural(data, added_news_name)                        
                                answer = extract_answer_from_logprobs(args.model_name, logprobs_dict)
                            try:
                                answer_text = id2bias[answer]
                            except:
                                answer_text = "error"
                                
                            result_file = pd.concat([result_file, pd.DataFrame([{
                                "id": data.id,
                                "news_name": data.news_name,
                                "content": data.content.replace("\n", " ").replace(",", " "),
                                "added_news_name": added_news_name,
                                "allsides_class": allsides_class,
                                "label": data.label,
                                "answer": answer_text
                            }])], ignore_index=True)    
                            
                        except:
                            result_file = pd.concat([result_file, pd.DataFrame([{
                                "id": data.id,
                                "news_name": data.news_name,
                                "content": data.content,
                                "added_news_name": added_news_name,
                                "allsides_class": allsides_class,
                                "label": data.label,
                                "answer": "Error!!!"
                            }])], ignore_index=True)    
                            print(f"Error: {data.id}, len of content: {len(data.content)}")
                            error_idx += 1
                                            
                if i == 10:
                    result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
                if i % 50 == 0:
                    result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
            
            result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
            print(f"Error index: {error_idx}")
            
    elif args.task == "summarization":
        error_idx = 0
        with torch.no_grad():
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
                        try:
                            if args.closed_source:
                                # TODO: 추후 추가
                                raise ValueError("Closed source model does not support summarization task")
                                # summary, data = model.process_question_summarization(data, added_news_name, args.summary_length)
                            elif args.reasoning:
                                # TODO: 추후 추가
                                raise ValueError("Reasoning model does not support summarization task")
                                # summary, data = model.process_question_summarization(data, added_news_name, args.summary_length)
                            else:   
                                summary, data = model.process_question_summarization(data, added_news_name, args.summary_length)                        

                            # print(f"Model output: \n{summary}")
                            # print("-"*100)
                            
                            result_file = pd.concat([result_file, pd.DataFrame([{
                                "id": data.id,
                                "news_name": data.news_name,
                                "content": data.content.replace("\n", " ").replace(",", " "),
                                "added_news_name": added_news_name,
                                "allsides_class": allsides_class,
                                "label": data.label,
                                "answer": summary
                            }])], ignore_index=True)    
                            
                        except Exception as e:
                            print(f"Error: {e}")
                            result_file = pd.concat([result_file, pd.DataFrame([{
                                "id": data.id,
                                "news_name": data.news_name,
                                "content": data.content,
                                "added_news_name": added_news_name,
                                "allsides_class": allsides_class,
                                "label": data.label,
                                "answer": "Error!!!"
                            }])], ignore_index=True)    
                            print(f"Error: {data.id}, len of content: {len(data.content)}")
                            error_idx += 1
                                            
                if i == 10:
                    result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
                if i % 50 == 0:
                    result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
            
            result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
            print(f"Error index: {error_idx}")
            
    elif args.task == "bias_prediction_for_order":
        error_idx = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataset, desc="Inferencing dataset")):
                if args.start_from is not None and i < args.start_from:
                    continue
                
                if args.media_outlet_type == "allsides":
                    target_class_dict = allsides_class_dict
                else:
                    raise ValueError(f"Unsupported media outlet type: {args.media_outlet_type}")
                
                for allsides_class, added_news_name_list in target_class_dict.items():
                    for added_news_name in added_news_name_list:
                        try:
                            if args.closed_source:
                                answer, data, order = model.process_question_natural_for_order(data, added_news_name)
                            else:   
                                logprobs_dict, data, order = model.process_question_natural_for_order(data, added_news_name)                        
                                answer = extract_answer_from_logprobs(args.model_name, logprobs_dict)
                            
                            if order == "lrc":
                                id2bias = {"A": "left", "B": "right", "C": "center"}
                            elif order == "rlc":
                                id2bias = {"A": "right", "B": "left", "C": "center"}
                            elif order == "rcl":
                                id2bias = {"A": "right", "B": "center", "C": "left"}
                            elif order == "crl":
                                id2bias = {"A": "center", "B": "right", "C": "left"}
                            elif order == "clr":
                                id2bias = {"A": "center", "B": "left", "C": "right"}
                            
                            # print("-"*100)
                            # print(f"order: {order}")
                            # print("-"*100)
                            # print(f"id2bias: {id2bias}")
                            
                            try:
                                answer_text = id2bias[answer]
                            except:
                                answer_text = "error"
                                
                            result_file = pd.concat([result_file, pd.DataFrame([{
                                "id": data.id,
                                "news_name": data.news_name,
                                "content": data.content.replace("\n", " ").replace(",", " "),
                                "added_news_name": added_news_name,
                                "allsides_class": allsides_class,
                                "label": data.label,
                                "answer": answer_text
                            }])], ignore_index=True)    
                            
                        except:
                            result_file = pd.concat([result_file, pd.DataFrame([{
                                "id": data.id,
                                "news_name": data.news_name,
                                "content": data.content,
                                "added_news_name": added_news_name,
                                "allsides_class": allsides_class,
                                "label": data.label,
                                "answer": "Error!!!"
                            }])], ignore_index=True)    
                            print(f"Error: {data.id}, len of content: {len(data.content)}")
                            error_idx += 1
                                            
                if i == 10:
                    result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
                if i % 50 == 0:
                    result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
            
            result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
            print(f"Error index: {error_idx}")
            
    elif args.task == "xai":
        result_file = pd.DataFrame(columns=["id", "news_name", "content", "added_news_name", "allsides_class", "label", "answer", "saliency_ratio", "baseline_saliency_ratio", "probability_change_ratio", "original_probability", "baseline_probability", "probability_change"])
        if args.dataset_name == "allsides":
            id2bias = {"A": "left", "B": "center", "C": "right"}
        elif args.dataset_name == "hyperpartisan":
            id2bias = {"A": "left", "B": "center", "C": "right"}
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset_name}")
        
        error_idx = 0
        with torch.no_grad():
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
                        if added_news_name == "none":
                            continue
                        try:
                            xai_results, data = model.process_question_natural_xai(data, added_news_name)
                            answer_text = xai_results["target_token"]
                            result_file = pd.concat([result_file, pd.DataFrame([{
                                "id": data.id,
                                "news_name": data.news_name,
                                "content": data.content.replace("\n", " ").replace(",", " "),
                                "added_news_name": added_news_name,
                                "allsides_class": allsides_class,
                                "label": data.label,
                                "answer": answer_text,
                                "saliency_ratio": xai_results["news_name_influence_ratio"]["saliency_ratio"],
                                "baseline_saliency_ratio": xai_results["news_name_influence_ratio"]["baseline_saliency_ratio"],
                                "probability_change_ratio": xai_results["news_name_influence_ratio"]["probability_change_ratio"],
                                "original_probability": xai_results["original_probability"],
                                "baseline_probability": xai_results["baseline_probability"],
                                "probability_change": xai_results["probability_change"]
                            }])], ignore_index=True)           
                        except Exception as e:
                            print(f"Error: {e}")
                            error_idx += 1       
                if i == 10:
                    result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
                if i % 50 == 0:
                    result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
        
        result_file.to_csv(os.path.join(log_dir, f"result_{i}.csv"), index=False)
        print(f"Error index: {error_idx}")
                
            
    else:
        raise ValueError(f"Unsupported task: {args.task}")

if __name__ == "__main__":
    main()