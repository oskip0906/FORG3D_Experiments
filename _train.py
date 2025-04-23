import unsloth
import wandb
import argparse
import json
import os
import re
import torch
from datasets import Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer
from tqdm.auto import tqdm
import glob
from unsloth import unsloth_train
from peft import LoraConfig, get_peft_model
from accelerate import PartialState
from PIL import Image
import signal, sys

# Argument parsing
parser = argparse.ArgumentParser(description="Fine-tune a vision-language model.")
parser.add_argument('--model', default='Qwen/Qwen2-VL-2B-Instruct', help='Model to fine-tune')
parser.add_argument('--dataset', default='training_data.jsonl', help='Path to the dataset')
parser.add_argument('--image_directory', default='./', help='Directory containing images')
parser.add_argument('--no_continue', action='store_true', default=False, help='Do not continue training if model already exists')
parser.add_argument('--output_directory', default='output/', help='Directory to save the model')
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--evaluate_before_training', action='store_true', default=False)
parser.add_argument('--freeze', choices=['vision', 'text'], required=False)
parser.add_argument('--grad_acc', default=4, type=int)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--eval_steps', default=100, type=int)
parser.add_argument('--eval_strat', default='steps', type=str)
parser.add_argument('--eval_sample_percent', default=0.05, type=float)
parser.add_argument('--save_steps', default=500, type=int)
parser.add_argument('--save_strat', default='steps', type=str)
parser.add_argument('--save_total_limit', default=1, type=int)
parser.add_argument('--lora', action='store_true', default=False, help='Use LoRA')
parser.add_argument('--lora_r', default=8, type=int)
parser.add_argument('--lora_alpha', default=32, type=int)
parser.add_argument('--lora_dropout', default=0.1, type=float)
args = parser.parse_args()

print("Arguments:")
print(json.dumps(vars(args), indent=2))

device_string = PartialState().process_index

# Load Processor
min_pixels = 256 * 28 * 28
max_pixels = 1024 * 28 * 28
processor = AutoProcessor.from_pretrained(
    args.model,                    
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)
# print("Setting padding side to right")
# processor.padding_side = 'right'
max_seq_length = 4096

# Load dataset from JSONL file using the new format
with open(args.dataset, 'r') as f:
    raw_data = [json.loads(line) for line in f]

formatted_dataset = []
for sample in tqdm(raw_data, desc="Formatting data"):
    # Verify that necessary keys are present
    if not all(key in sample for key in ["image", "question", "answer", "options"]):
        continue

    options_str = " ".join([f"{k}: {v}." for k, v in sample["options"].items()])
    full_question = f"{sample['question']} Options: {options_str}"

    # Define the basic messages
    entry = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful visual assistant. For each question, choose A, B, C, or D. Only return one uppercase letter as your final answer. No explanation."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"image": os.path.join(args.image_directory, sample["image"]), "type": "image"},
                {"text": full_question, "type": "text"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"text": sample["answer"], "type": "text"}
            ]
        }
    ]

    prompt = processor.apply_chat_template(entry, tokenize=False)
    formatted_dataset.append({
        "prompt": prompt,
        "images": [os.path.join(args.image_directory, sample["image"])]
    })

print(formatted_dataset[0])

dataset = Dataset.from_list(formatted_dataset)
print(f"Dataset: {dataset}")
# exit()

image_token = "<image>"
if image_token not in processor.tokenizer.additional_special_tokens:
    processor.tokenizer.add_special_tokens({'additional_special_tokens': [image_token]})
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index(image_token)
]

def make_pil_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None

def collate_fn(examples):
    prompts = [e["prompt"] for e in examples]
    images = [[make_pil_image(i) for i in e["images"]] for e in examples]
    answers = [
        re.search(r"<\|im_start\|>assistant\n([A-D])<\|im_end\|>", e["prompt"]).group(1)
        for e in examples
    ]

    # Tokenize the prompt and images
    batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    batch.pixel_values = batch.pixel_values.to(model.dtype)

    input_ids = batch["input_ids"]
    labels = input_ids.clone()

    # Mask everything by default
    labels[:, :] = -100

    for i, answer in enumerate(answers):
        answer_ids = processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
        answer_len = len(answer_ids)

        # Try to find where the answer starts in the input
        for j in range(input_ids.size(1) - answer_len + 1):
            if input_ids[i, j:j + answer_len].tolist() == answer_ids:
                labels[i, j:j + answer_len] = input_ids[i, j:j + answer_len]
                break

    # Mask out padding and image tokens
    labels[input_ids == processor.tokenizer.pad_token_id] = -100
    labels[input_ids == image_token_id] = -100

    batch["labels"] = labels
    return batch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model, 
    torch_dtype="auto", 
    trust_remote_code=True,
    device_map='auto'
)

model.resize_token_embeddings(len(processor.tokenizer))

if args.freeze == "vision":  
    model.visual.requires_grad = False
if args.freeze == "text":
    model.model.requires_grad = False

print(model)

# Training Setup
training_args = SFTConfig(
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_acc,
    output_dir=args.output_directory,
    num_train_epochs=args.num_epochs,
    save_total_limit=args.save_total_limit,
    eval_strategy=args.eval_strat,
    eval_steps=args.eval_steps,
    learning_rate=args.learning_rate,
    save_strategy=args.save_strat,
    save_steps=args.save_steps,
    weight_decay=args.weight_decay,
    max_grad_norm=1.0,
    logging_steps=5,
    use_liger=True,
    resume_from_checkpoint=bool(glob.glob(f"{args.output_directory}/checkpoint-*")) and not args.no_continue,
    bf16=True,
    optim="adamw_8bit",
    gradient_checkpointing_kwargs={'use_reentrant': False},
    remove_unused_columns=False,
    max_seq_length=max_seq_length,
    dataset_kwargs={
      "skip_prepare_dataset": True,
    },
)

if args.lora:
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["self_attn.q_proj", "self_attn.k_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

# Split the dataset into train and eval sets
train_data, eval_data = train_test_split(dataset.to_list(), test_size=args.eval_sample_percent, random_state=42)
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor.tokenizer,
)

total_num_params = sum([torch.prod(torch.tensor(p.shape)) for n, p in model.named_parameters()])
print(f"Total number of parameters: {total_num_params:,}")

if args.lora:
    num_lora_params = sum([torch.prod(torch.tensor(p.shape)) for n, p in model.named_parameters() if "lora_" in n])
    print(f"Number of LoRA parameters: {num_lora_params:,}")
    num_non_lora_params = sum([torch.prod(torch.tensor(p.shape)) for n, p in model.named_parameters() if "lora_" not in n])
    print(f"Number of non-LoRA parameters: {num_non_lora_params:,}")
    assert num_non_lora_params + num_lora_params == total_num_params, "Number of LoRA and non-LoRA parameters don't sum to total number of parameters"

# Get metrics before training
if args.evaluate_before_training:
    trainer.evaluate()

# Save the model if crashes
def _save_and_exit(signum, frame):
    print(f"Received signal {signum}, saving model and processor…")
    trainer.save_model(args.output_directory)
    processor.save_pretrained(args.output_directory)
    sys.exit(0)

signal.signal(signal.SIGTERM, _save_and_exit)
signal.signal(signal.SIGINT,  _save_and_exit)

# Train the model
try:
    unsloth_train(
        trainer,
        resume_from_checkpoint=bool(glob.glob(f"{args.output_directory}/checkpoint-*")) and not args.no_continue
    )
except Exception as e:
    print(f"Training failed with error: {e}")

# Save and push to hub  
trainer.save_model(args.output_directory)
processor.save_pretrained(args.output_directory)
print(f"Model saved to {args.output_directory}")
