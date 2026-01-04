python scripts/inference.py \
--model_name "microsoft/phi-4" \
--dataset_name "hyperpartisan" \
--dataset_path "../../data/hyperpartisan/" \
--output_path "../../logs/" \
--cuda "2"

python scripts/inference.py \
--model_name "meta-llama/Llama-3.3-70B-Instruct" \
--dataset_name "hyperpartisan" \
--dataset_path "../../data/hyperpartisan/" \
--output_path "../../logs/" \
--cuda "2"