# python scripts/inference.py \
# --task "summarization" \
# --summary_length 3 \
# --model_name "google/gemma-2-27b-it" \
# --dataset_name "allsides" \
# --dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
# --output_path "../../logs/" \
# --cuda "0"

# python scripts/inference.py \
# --task "summarization" \
# --summary_length 5 \
# --model_name "google/gemma-2-27b-it" \
# --dataset_name "allsides" \
# --dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
# --output_path "../../logs/" \
# --cuda "0"

python scripts/inference.py \
--task "summarization" \
--summary_length 5 \
--model_name "google/gemma-2-27b-it" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization_center.json" \
--output_path "../../logs/" \
--cuda "0"