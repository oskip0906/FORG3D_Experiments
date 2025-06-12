# Example fine-tuning script

MODEL="qwen2-vl-2b-instruct"
DATASET="training_data.jsonl"
IMAGE_DIR="big_unenhanced_data"
OUTPUT_DIR="big_unenhanced_results"

CMD="--model ${MODEL} --dataset ${DATASET} --image_dir ${IMAGE_DIR} --output_dir ${OUTPUT_DIR} --lora"

# if ! python -c "import torch" &> /dev/null; then
#     pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121/
# fi

# if ! pip freeze | grep -q -f requirements.txt; then
#     pip install -r requirements.txt
# fi

python _train.py $CMD