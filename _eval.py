import os
import json
import requests
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, AutoConfig
from peft import PeftModel
from PIL import Image
import random
from datasets import load_dataset
import argparse
import re

parser = argparse.ArgumentParser(description="Evaluate the model on a dataset.")
parser.add_argument("--dataset", choices=["generated", "3dsr"], default="3dsr", help="Dataset type: 'generated' or '3dsr'.")
parser.add_argument("--dataset_file", default="testing_data.jsonl", help="Path to the dataset file for generated data.")
parser.add_argument("--peft_path", default=None, help="Path to the PEFT model.")
parser.add_argument("--base_model", default="qwen2-vl-2b-instruct", help="Path to the base model.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_processor(base_model_path, peft_model_path):
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if peft_model_path is not None:
        peft_tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    base_vocab = set(base_tokenizer.get_vocab().keys())
    peft_vocab = peft_tokenizer.get_vocab() if peft_model_path is not None else {}
    new_tokens = [tok for tok in peft_vocab if tok not in base_vocab]
    if new_tokens:
        base_tokenizer.add_tokens(new_tokens)
    processor = AutoProcessor.from_pretrained(base_model_path)
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    ).to(device)
    base_model.resize_token_embeddings(len(base_tokenizer))
    if peft_model_path is not None:
        model = PeftModel.from_pretrained(base_model, peft_model_path)
    else:
        model = base_model
    model.eval().to(device)
    return processor, model

def evaluate_entry(entry, model, processor, test=False):

    if test:
        image_path = os.path.join("testing_data", entry["image"])
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

    m = re.search(r'\b([ABCD])\b', decoded_output)
    if m:
        decoded_output = m.group(1)
    else:
        decoded_output = None

    if decoded_output not in ["A", "B", "C", "D"]:
        decoded_output = None
    
    return {
        "question": entry["question"],
        "predicted": decoded_output,
        "ground_truth": entry["answer"],
    }

def evaluate_generated_entries(jsonl_file, model, processor):

    with open(jsonl_file, 'r') as f:
        entries = [json.loads(line.strip()) for line in f]
        random.shuffle(entries)

        category_results = {"orientation_in_front_of": 0, "orientation_on_the_left": 0, "orientation_viewpoint": 0}
        category_counts = {"orientation_in_front_of": 0, "orientation_on_the_left": 0, "orientation_viewpoint": 0}

        for entry in entries:
            try:
                # Determine the category based on the question and answers
                question = entry["question"].lower()
                question_prefix = "consider the real-world 3d locations and orientations of the objects. "
                
                # Default category
                category = "orientation_viewpoint"
                
                # Check option values by converting to lowercase for case-insensitive matching
                option_values = [str(value).lower() for value in entry["options"].values()]
                
                if "front" in " ".join(option_values) and question.startswith(question_prefix + "if"):
                    category = "orientation_in_front_of"
                elif "left" in " ".join(option_values) and question.startswith(question_prefix + "if"):
                    category = "orientation_on_the_left"
                elif "which side" in question:
                    category = "orientation_viewpoint"

                eval_result = evaluate_entry(entry, model, processor, test=True)
                if eval_result is not None:
                    if eval_result['predicted'] == eval_result['ground_truth']:
                        print(f"Correct for category {category}")
                        category_results[category] += 1
                    else:
                        print(f"Incorrect for category {category}")
                    category_counts[category] += 1
            except Exception as e:
                print(f"Error processing entry: {e}")

        overall_correct = sum(category_results.values())
        overall_total = sum(category_counts.values())
        overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0

    return {
        "overall_accuracy": overall_accuracy,
        "overall_total": overall_total,
        "category_results": {
            category: {
                "accuracy": (category_results[category] / category_counts[category]) * 100 if category_counts[category] > 0 else 0,
                "total": category_counts[category]
            }
            for category in category_results
        }
    }

def evaluate_3dsr_entries(model, processor):
    ds = load_dataset("ccvl/3DSRBench")

    category_results = {
        "orientation_in_front_of": {"accuracy": 0, "total": 0},
        "orientation_on_the_left": {"accuracy": 0, "total": 0},
        "orientation_viewpoint": {"accuracy": 0, "total": 0},
    }

    for category in ["orientation_in_front_of", "orientation_on_the_left", "orientation_viewpoint"]:
        category_correct = 0
        category_total = 0

        for _, entry in enumerate(ds["test"]):
            if entry["category"] == category:
                category_total += 1
                try:
                    eval_result = evaluate_entry(entry, model, processor)
                    if eval_result is not None:
                        if eval_result['predicted'] == eval_result['ground_truth']:
                            print(f"Correct for category {category}")
                            category_correct += 1
                        else:
                            print(f"Incorrect for category {category}")
                except Exception as e:
                    print(f"Error evaluating entry for category {category}: {e}")

        category_results[category]["accuracy"] = (
            (category_correct / category_total) * 100 if category_total > 0 else 0
        )
        category_results[category]["total"] = category_total

    overall_correct = sum(
        (res["accuracy"] * res["total"] / 100) for res in category_results.values()
    )
    overall_count = sum(res["total"] for res in category_results.values())
    overall_accuracy = (overall_correct / overall_count) * 100 if overall_count > 0 else 0

    return {
        "overall_accuracy": overall_accuracy,
        "overall_total": overall_count,
        "category_results": category_results
    }

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    base_model = args.base_model
    peft_path = args.peft_path

    processor, model = load_model_and_processor(base_model, peft_path)
    
    print("Model and processor loaded successfully.")

    if dataset == "generated":
        jsonl_file = args.dataset_file
        accuracy = evaluate_generated_entries(jsonl_file, model, processor)
        print(accuracy)
    elif dataset == "3dsr":
        accuracy = evaluate_3dsr_entries(model, processor)
        print(accuracy)
        