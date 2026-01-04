python scripts/inference.py \
--model_name "google/gemma-2-27b-it" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json" \
--output_path "../../logs/" \
--cuda "1"

python scripts/inference.py \
--model_name "google/gemma-2-27b-it" \
--dataset_name "hyperpartisan" \
--dataset_path "../../data/hyperpartisan/" \
--output_path "../../logs/" \
--cuda "1"