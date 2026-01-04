python scripts/inference_xai.py \
--task "xai" \
--model_name "mistralai/Mistral-Small-24B-Instruct-2501" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
--output_path "../../logs/" \
--cuda "0"

python scripts/inference_xai.py \
--task "xai" \
--model_name "google/gemma-2-27b-it" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
--output_path "../../logs/" \
--cuda "0"