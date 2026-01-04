python scripts/inference.py \
--task "summarization" \
--summary_length 10 \
--model_name "Qwen/Qwen2.5-72B-Instruct" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
--output_path "../../logs/" \
--cuda "2"