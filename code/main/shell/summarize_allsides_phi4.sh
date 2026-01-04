# python scripts/inference.py \
# --task "summarization" \
# --summary_length 3 \
# --model_name "microsoft/phi-4" \
# --dataset_name "allsides" \
# --dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
# --output_path "../../logs/" \
# --cuda "1"

# python scripts/inference.py \
# --task "summarization" \
# --summary_length 5 \
# --model_name "microsoft/phi-4" \
# --dataset_name "allsides" \
# --dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
# --output_path "../../logs/" \
# --cuda "1"

python scripts/inference.py \
--task "summarization" \
--summary_length 10 \
--model_name "microsoft/phi-4" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
--output_path "../../logs/" \
--cuda "1"