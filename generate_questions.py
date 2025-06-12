import json
import csv
import os
import random
import shutil

# Convert "behind" to "back" and handle "None" cases
def check(direction):
    if direction is None:
        return "cannot be determined"
    if direction == "behind":
        return "back"
    return direction

# Ensure only 4 answer choices (A, B, C, D) by removing one incorrect option
def filter_choices(correct_choice, choices):
    incorrect_choices = [ch for ch in choices if ch != correct_choice]
    if incorrect_choices:
        choices.remove(random.choice(incorrect_choices))
    return choices

# Generate questions based on the metadata of a scene
def generate_questions(metadata):
    image_name = metadata["image_filename"]
    obj1 = metadata["ground_object"]
    obj2 = metadata["figure_object"]
    questions = []
    choices = ["left", "right", "front", "back", "cannot be determined"]
    # Facing side questions
    facing_side_obj1 = check(obj1["orientation"])
    facing_side_obj2 = check(obj2["orientation"])
    filtered_choices1 = filter_choices(facing_side_obj1, choices.copy())
    filtered_choices2 = filter_choices(facing_side_obj2, choices.copy())
    questions.extend([
        ("Consider the real-world 3D locations and orientations of the objects. Which side of the {} is facing the camera?".format(obj1["name"]), *filtered_choices1[:4], chr(ord('A') + filtered_choices1.index(facing_side_obj1))),
        ("Consider the real-world 3D locations and orientations of the objects. Which side of the {} is facing the camera?".format(obj2["name"]), *filtered_choices2[:4], chr(ord('A') + filtered_choices2.index(facing_side_obj2))),
    ])
    # Intrinsic relation questions
    obj1_relation = check(obj1.get("intrinsic_relation"))
    obj2_relation = check(obj2.get("intrinsic_relation"))
    if obj1_relation != "cannot be determined":
        if obj1_relation in ["left", "right"]:
            questions.append(
                ("Consider the real-world 3D locations and orientations of the objects. If I stand at the {}'s position facing where it is facing, is the {} on the left or right of me?".format(obj1["name"], obj2["name"]), "on the left", "on the right", None, None, chr(ord('A') + choices.index(obj1_relation)))
            )
        else:
            questions.append(
                ("Consider the real-world 3D locations and orientations of the objects. If I stand at the {}'s position facing where it is facing, is the {} in front of me or behind me?".format(obj1["name"], obj2["name"]), "in front of", "behind", None, None, chr(ord('A') + choices.index(obj1_relation) - 2))
        )
    if obj2_relation != "cannot be determined":
        if obj2_relation in ["left", "right"]:
            questions.append(
                ("Consider the real-world 3D locations and orientations of the objects. If I stand at the {}'s position facing where it is facing, is the {} on the left or right of me?".format(obj2["name"], obj1["name"]), "on the left", "on the right", None, None, chr(ord('A') + choices.index(obj2_relation)))
            )
        else:
            questions.append(
                ("Consider the real-world 3D locations and orientations of the objects. If I stand at the {}'s position facing where it is facing, is the {} in front of me or behind me?".format(obj2["name"], obj1["name"]), "in front of", "behind", None, None, chr(ord('A') + choices.index(obj2_relation) - 2))
            )
    return [(image_name,) + q for q in questions]


# Extract all files from the scenes directory
def extract_all_files(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    # Walk through all directories and subdirectories
    count = 0
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            source_path = os.path.join(root, filename)
            dest_path = os.path.join(destination_folder, filename)
            shutil.copy2(source_path, dest_path)
            count += 1
    print(f"Extracted {count} files to {destination_folder}")

# Generate CSV file with questions
def generate_csv(questions, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "question", "A", "B", "C", "D", "answer"])
        writer.writerows(questions)

# Convert CSV file to JSONL file
def convert_csv_to_jsonl(csv_path, jsonl_path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile, open(jsonl_path, 'w', encoding='utf-8') as jsonlfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            record = {
                "image": row["image"],
                "question": row["question"],
                "options": {
                    "A": row["A"],
                    "B": row["B"],
                    "C": row["C"],
                    "D": row["D"],
                },
                "answer": row["answer"]
            }
            jsonlfile.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    questions = []
    # Navigate to the root directory
    # root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # os.chdir(root_dir)
    # Extract all files from output
    images_source = "output/images"
    images_destination = "testing_data"
    scenes_source = "output/scenes"
    scenes_destination = "scenes_data"
    extract_all_files(images_source, images_destination)
    extract_all_files(scenes_source, scenes_destination)
    # Generate questions
    for filename in os.listdir(scenes_destination):
        with open(os.path.join(scenes_destination, filename), "r") as f:
            metadata = json.load(f)
            questions.extend(generate_questions(metadata))
    # Generate csv and jsonl files
    csv_file = "questions.csv"
    generate_csv(questions, csv_file)
    jsonl_file = "testing_data.jsonl"
    convert_csv_to_jsonl(csv_file, jsonl_file)
    print(f"Generated {len(questions)} questions in {csv_file} and {jsonl_file}")