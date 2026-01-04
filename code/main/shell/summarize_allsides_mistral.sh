python scripts/inference.py \
--task "summarization" \
--summary_length 10 \
--model_name "mistralai/Mistral-Small-24B-Instruct-2501" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
--output_path "../../logs/" \
--cuda "1"