python scripts/inference.py \
--task "summarization" \
--summary_length 5 \
--model_name "mistralai/Mistral-Small-24B-Instruct-2501" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization_rvt_v2.json" \
--output_path "../../logs/" \
--cuda "2"

python scripts/inference.py \
--task "summarization" \
--summary_length 5 \
--model_name "google/gemma-2-27b-it" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization_rvt_v2.json" \
--output_path "../../logs/" \
--cuda "2"

python scripts/inference.py \
--task "summarization" \
--summary_length 5 \
--model_name "Qwen/Qwen2.5-72B-Instruct" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization_rvt_v2.json" \
--output_path "../../logs/" \
--cuda "2"