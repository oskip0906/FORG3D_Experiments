import os
import json
import requests
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image
import random
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser(description="Evaluate the model on a dataset.")
parser.add_argument("--dataset", choices=["generated", "3dsr"], default="generated", help="Dataset type: 'generated' or '3dsr'.")
parser.add_argument("--dataset_file", default="generated_data.jsonl", help="Path to the dataset file for generated data.")
parser.add_argument("--max_entries", default=100, help="Maximum number of entries to evaluate for generated data.")
parser.add_argument("--peft_path", default=None, help="Path to the PEFT model.")
parser.add_argument("--base_model", default="Qwen/Qwen2-VL-2B-Instruct", help="Path to the base model.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_processor(base_model_path, peft_model_path):
    processor = AutoProcessor.from_pretrained(base_model_path)
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if peft_model_path:
        model = PeftModel.from_pretrained(base_model, peft_model_path)
    else:
        model = base_model
    model.eval()
    model.to(device)
    return processor, model

def evaluate_entry(entry, model, processor, test=False):

    if test:
        image_path = os.path.join("images_data", entry["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        f'{entry["question"]} Options: '
                        f'A: {entry["options"]["A"]}, '
                        f'B: {entry["options"]["B"]}, '
                        f'C: {entry["options"]["C"]}, '
                        f'D: {entry["options"]["D"]}. '
                        "Answer with one letter: A, B, C, or D."
                    )}
                ]
            }
        ]
    else:
        try:
            image_url = entry["image_url"]
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        except Exception as e:
            print(f"Error loading image from URL {image_url}: {e}")
            return None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        f'{entry["question"]} Options: '
                        f'A: {entry["A"]}, '
                        f'B: {entry["B"]}, '
                        f'C: {entry["C"]}, '
                        f'D: {entry["D"]}. '
                    "Answer with one letter: A, B, C, or D."
                    )}
                ]
            }
        ]
    
    # Generate the prompt using the provided chat template.
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize the prompt along with the image.
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(model.device)
    
    # Generate model output.
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=32)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    decoded_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    if decoded_output not in ["A", "B", "C", "D"]:
        decoded_output = None
    
    return {
        "question": entry["question"],
        "predicted": decoded_output,
        "ground_truth": entry["answer"],
    }

def evaluate_generated_entries(jsonl_file, max_entries, model, processor):
    correct = 0

    with open(jsonl_file, 'r') as f:
        entries = [json.loads(line.strip()) for line in f]
        random.shuffle(entries)

        for entry in entries[:max_entries]:
            try:
                eval_result = evaluate_entry(entry, model, processor, test=True)
                if eval_result is not None:
                    if eval_result['predicted'] == eval_result['ground_truth']:
                        print("Correct!")
                        correct += 1
                    else:
                        print("Incorrect.")
            except Exception as e:
                print(f"Error processing entry: {e}")

    return "Accuracy: {:.2f}%, Total: {}".format(
        (correct / max_entries) * 100, max_entries
    )

def evaluate_3dsr_entries(model, processor):
    ds = load_dataset("ccvl/3DSRBench")
    correct = 0
    count = 0

    for category in ["orientation_in_front_of", "orientation_on_the_left", "orientation_viewpoint"]:
        for _, entry in enumerate(ds["test"]):
            if entry["category"] == category:
                try:
                    eval_result = evaluate_entry(entry, model, processor)
                    if eval_result is not None:
                        if eval_result['predicted'] == eval_result['ground_truth']:
                            print(f"Correct for category {category}!")
                            correct += 1
                        else:
                            print(f"Incorrect for category {category}.")
                        count += 1
                except Exception as e:
                    print(f"Error processing entry in category {category}: {e}")

    return "Accuracy: {:.2f}%, Total: {}".format(
        (correct / count) * 100, count
    )

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    max_entries = args.max_entries
    base_model = args.base_model
    peft_path = args.peft_path

    processor, model = load_model_and_processor(base_model, peft_path)
    
    print("Model and processor loaded successfully.")

    if dataset == "generated":
        jsonl_file = args.dataset_file
        accuracy = evaluate_generated_entries(jsonl_file, max_entries, model, processor)
        print(accuracy)
    elif dataset == "3dsr":
        accuracy = evaluate_3dsr_entries(model, processor)
        print(accuracy)
