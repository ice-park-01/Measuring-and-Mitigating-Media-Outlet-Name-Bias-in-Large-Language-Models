python scripts/inference_xai.py \
--task "xai" \
--model_name "Qwen/Qwen2.5-72B-Instruct" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
--output_path "../../logs/" \
--cuda "2"