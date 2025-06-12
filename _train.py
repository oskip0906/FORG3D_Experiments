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
import gc

wandb.login(key="your_wandb_api_key_here")  # Replace with your actual WandB API key

# Clear CUDA cache at start
torch.cuda.empty_cache()
gc.collect()

# Argument parsing
parser = argparse.ArgumentParser(description="Fine-tune a vision-language model.")
parser.add_argument('--model', default='Qwen/Qwen2-VL-2B-Instruct', help='Model to fine-tune')
parser.add_argument('--dataset', default='training_data.jsonl', help='Path to the dataset')
parser.add_argument('--image_directory', default='./', help='Directory containing images')
parser.add_argument('--no_continue', action='store_true', default=False, help='Do not continue training if model already exists')
parser.add_argument('--output_directory', default='output/', help='Directory to save the model')
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--evaluate_before_training', action='store_true', default=False)
parser.add_argument('--freeze', choices=['vision', 'text'], required=False)
parser.add_argument('--grad_acc', default=8, type=int)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--eval_steps', default=100, type=int)
parser.add_argument('--eval_strat', default='steps', type=str)
parser.add_argument('--eval_sample_percent', default=0.005, type=float)
parser.add_argument('--save_steps', default=500, type=int)
parser.add_argument('--save_strat', default='steps', type=str)
parser.add_argument('--save_total_limit', default=1, type=int)
parser.add_argument('--lora', action='store_true', default=True, help='Use LoRA')
parser.add_argument('--lora_r', default=8, type=int)
parser.add_argument('--lora_alpha', default=16, type=int)
parser.add_argument('--lora_dropout', default=0.1, type=float)
parser.add_argument('--max_image_size', default=512, type=int, help='Max image dimension')
args = parser.parse_args()

print("Arguments:")
print(json.dumps(vars(args), indent=2))

device_string = PartialState().process_index

# Load processor - don't modify it
min_pixels = 256 * 28 * 28 // 4
max_pixels = 512 * 28 * 28
processor = AutoProcessor.from_pretrained(
    args.model,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)

# Max sequence length for the model
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

# Get existing image token ID from the processor's tokenizer
image_token = "<image>"
if image_token in processor.tokenizer.get_vocab():
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
    print(f"Found existing <image> token with ID: {image_token_id}")
else:
    print("Warning: <image> token not found in vocabulary")
    image_token_id = None

def make_pil_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        # Resize image to reduce memory usage
        if args.max_image_size:
            img.thumbnail((args.max_image_size, args.max_image_size), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None

def collate_fn(examples):
    valid_examples = []
    answers = []
    for e in examples:
        m = re.search(r"<\|im_start\|>assistant\s*([A-D])<\|im_end\|>", e["prompt"])
        if not m:
            continue
        answers.append(m.group(1))
        valid_examples.append(e)

    # If nothing valid remains, skip this batch
    if not valid_examples:
        return {}
    
    prompts = [e["prompt"] for e in valid_examples]
    images  = [[make_pil_image(i) for i in e["images"]] for e in valid_examples]

    # Tokenize prompts and images
    batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    batch.pixel_values = batch.pixel_values.to(model.dtype)
    input_ids = batch["input_ids"]

    # Initialize all labels to -100
    labels = torch.full_like(input_ids, -100)

    # Un-mask only the correct answer token for each valid example
    for i, answer in enumerate(answers):
        answer_ids = processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
        L = len(answer_ids)
        for j in range(input_ids.size(1) - L + 1):
            if input_ids[i, j:j+L].tolist() == answer_ids:
                labels[i, j:j+L] = input_ids[i, j:j+L]
                break

    # Mask padding tokens
    pad_id = processor.tokenizer.pad_token_id
    labels[input_ids == pad_id] = -100
    
    # Mask image tokens if they exist
    if image_token_id is not None:
        labels[input_ids == image_token_id] = -100

    batch["labels"] = labels
    return batch

# Load model - don't resize token embeddings
print("Loading model with memory optimizations...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={'': 0},
    max_memory={0: "20GB"},
)

print(f"Tokenizer vocab size: {len(processor.tokenizer)}")
print(f"Model vocab size: {model.config.vocab_size}")

# Freeze vision encoder by default to save memory
if args.freeze == "vision" or not args.freeze:  
    print("Freezing vision encoder to save memory")
    for param in model.visual.parameters():
        param.requires_grad = False
        
if args.freeze == "text":
    for param in model.model.parameters():
        param.requires_grad = False

# Clear cache after model loading
torch.cuda.empty_cache()
gc.collect()

# Training Setup with memory optimizations
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
    resume_from_checkpoint=bool(glob.glob(f"{args.output_directory}/checkpoint-*")) and not args.no_continue,
    bf16=True,
    optim="adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    remove_unused_columns=False,
    max_seq_length=max_seq_length,
    dataset_kwargs={
      "skip_prepare_dataset": True,
    },
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=8,
)

# Use LoRA by default for memory efficiency
if args.lora:
    print("Applying LoRA for memory efficiency")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# Split the dataset into train and eval sets
train_data, eval_data = train_test_split(dataset.to_list(), test_size=args.eval_sample_percent, random_state=42)
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# Clear cache before creating trainer
torch.cuda.empty_cache()
gc.collect()
    
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
    trainer.save_pretrained_kwargs = {"save_peft_format": True}

# Print memory usage
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Get metrics before training
if args.evaluate_before_training:
    trainer.evaluate()

# Save the model if crashes
def _save_and_exit(signum, frame):
    print(f"Received signal {signum}, saving model and processor…")
    trainer.save_model(args.output_directory)
    processor.save_pretrained(args.output_directory)
    torch.cuda.empty_cache()
    gc.collect()
    sys.exit(0)

signal.signal(signal.SIGTERM, _save_and_exit)
signal.signal(signal.SIGINT, _save_and_exit)
signal.signal(signal.SIGUSR1, _save_and_exit) 

# Train the model with memory management
try:
    torch.cuda.empty_cache()
    gc.collect()
    
    unsloth_train(
        trainer,
        resume_from_checkpoint=bool(glob.glob(f"{args.output_directory}/checkpoint-*")) and not args.no_continue
    )
except Exception as e:
    print(f"Training failed with error: {e}")
    torch.cuda.empty_cache()
    gc.collect()
    raise e

# Save and push to hub  
trainer.save_model(args.output_directory)
processor.save_pretrained(args.output_directory)
print(f"Model saved to {args.output_directory}")

# Final cleanup
torch.cuda.empty_cache()
gc.collect()
