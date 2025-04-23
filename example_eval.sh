# Example evaluation script

DATASET="3dsr"
DATASET_FILE="generated_data.jsonl" # not used for 3dsr
MAX_ENTRIES=500 # not used for 3dsr
BASE_MODEL="qwen2-vl-2b-instruct"
PEFT_PATH="small_enhanced_results"

CMD="--dataset $DATASET --dataset_file $DATASET_FILE --max_entries $MAX_ENTRIES --base_model $BASE_MODEL --peft_path $PEFT_PATH"

PYTHON_EXECUTABLE="python"

if ! python -c "import torch" &> /dev/null; then
    pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121/
fi

if ! pip freeze | grep -q -f requirements.txt; then
    pip install -r requirements.txt
fi

$PYTHON_EXECUTABLE _eval.py $CMD