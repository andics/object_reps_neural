#!/usr/bin/env python3

import os
import json
import csv
import base64
import argparse
import re
from datetime import datetime

################################################################################
# We'll import "OpenAI" from your specialized library
################################################################################
try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "Could not import 'OpenAI' from 'openai'. Ensure your environment matches the usage from your minimal example."
    )

################################################################################
# Utility: check if a string is a purely numeric value
################################################################################

def is_number(s: str) -> bool:
    """
    Return True if 's' can be interpreted as a float or int.
    We skip leading/trailing whitespace.
    """
    s = s.strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False

################################################################################
# MultiModalCSVRequester
################################################################################

class MultiModalCSVRequester:
    def __init__(self, csv_file: str, log_func):
        """
        :param csv_file: path to the CSV file for storing [folder, shape, number, response].
        :param log_func: function(msg) to log messages to console + file.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_file = os.path.join(script_dir, csv_file)
        self.log_func = log_func

        # Load the API key from ../api_key.json
        config_path = os.path.join(script_dir, "..", "api_key.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Could not find api_key.json at: {config_path}\n"
                "Please place your openai api_key in a JSON file, e.g.: "
                '{"api_key": "sk-..."}'
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.api_key = config["api_key"]

    def encode_image_from_file(self, image_path: str) -> str:
        """
        Reads and Base64-encodes the image from local file
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def run_request(
        self,
        folder_name: str,
        shape_label: str,
        number_label: str,
        image_path: str,
        question_text: str,
    ) -> str:
        """
        Actually makes the multi-modal request to GPT-4o, returning the response text.
        Does NOT write to CSV here; that is handled outside.
        """
        self.log_func(f"   [Request Start] folder={folder_name}")

        # 1) Fresh client => brand-new session
        client = OpenAI(api_key=self.api_key)

        # 2) Encode the single image
        self.log_func(f"   Encoding image: {image_path}")
        encoded_image = self.encode_image_from_file(image_path)

        # 3) Build the multi-modal 'content' list
        content_list = [
            {"type": "text", "text": question_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                },
            },
        ]
        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]

        # 4) Make the request
        self.log_func("   Sending request to model='gpt-4o' (multi-modal style).")
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
        except Exception as e:
            err_msg = f"   [ERROR] OpenAI request failed: {str(e)}"
            self.log_func(err_msg)
            self.log_func("   [Request End with Error]\n")
            return f"ERROR: {str(e)}"

        response_text = completion.choices[0].message.content.strip()
        self.log_func(f"   Received response: {response_text}")
        self.log_func("   [Request End]\n")
        return response_text


################################################################################
# CSV Handling: read / write
################################################################################

def read_csv_to_dict(csv_path: str, log_func) -> dict:
    """
    Reads an existing CSV of format [folder_name, shape, number, response]
    into a dict: { folder_name -> (shape, number, response) }

    If CSV doesn't exist, returns empty dict.
    """
    if not os.path.isfile(csv_path):
        log_func(f"[Info] CSV file '{csv_path}' does not exist yet => starting fresh.")
        return {}

    data_dict = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            folder, shape, num, resp = row[:4]
            data_dict[folder] = (shape, num, resp)
    log_func(f"[Info] Read {len(data_dict)} rows from existing CSV '{csv_path}'.")
    return data_dict


def write_dict_to_csv(csv_path: str, data_dict: dict, log_func):
    """
    Writes data_dict: { folder_name -> (shape, number, response) }
    to CSV in alphabetical order of folder_name.
    Format: [folder_name, shape, number, response]
    """
    items_sorted = sorted(data_dict.items(), key=lambda x: x[0].lower())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for folder_name, (shape, number, resp) in items_sorted:
            writer.writerow([folder_name, shape, number, resp])
    log_func(f"[Info] Wrote {len(items_sorted)} rows to CSV '{csv_path}'.")


################################################################################
# Subfolder/Logic
################################################################################

def parse_folder_name(folder_name: str):
    """
    e.g. "flipped_G_convex_0" => shape_label="convex", number_label="0"
    If 'convex'/'concave' not in folder_name => (None, None)
    """
    parts = folder_name.split("_")
    if len(parts) < 2:
        return (None, None)
    lower_name = folder_name.lower()
    if ("convex" not in lower_name) and ("concave" not in lower_name):
        return (None, None)

    shape_label = parts[-2]
    number_label = parts[-1]
    return (shape_label, number_label)

def build_question_text(folder_name: str):
    """
    If folder_name includes 'flipped' => "right caused left"
    else => "left caused right"
    """
    lower_name = folder_name.lower()
    if "flipped" in lower_name:
        return (
            "Consider this image of two objects on a black background."
            "\n"
            "The image is a snippet of a video, where the object on the right moves "
            "towards the object on the left (right to left motion). The frame I have provided you is the exact frame "
            "at which the right object stops, and the left object starts moving with the same velocity and "
            "direction as the right object was moving (right to left)."
            "I want to understand your perception of causality here based on distance between the objects.\n"
            "You should not output anything more than a single number from 1 to 7, answering how much you agree with the following statement: "
            "\"The object on the right caused the object on the left to move\".\n"
            "1 indicates strong disagreement, 7 indicates strong agreement."
        )
    else:
        return (
            "Consider this image of two objects on a black background."
            "\n"
            "The image is a snippet of a video, where the object on the left moves "
            "towards the object on the right (left to right motion). The frame I have provided you is the exact frame "
            "at which the left object stops, and the right object starts moving with the same velocity and "
            "direction as the left object was moving (left to right)."
            "I want to understand your perception of causality here based on distance between the objects.\n"
            "You should not output anything more than a single number from 1 to 7, answering how much you agree with the following statement: "
            "\"The object on the left caused the object on the right to move\".\n" \
            "1 indicates strong disagreement, 7 indicates strong agreement."
        )

def select_middle_crop(crops_dir: str):
    """
    Finds the 'middle' crop_frame_XXXXX.png by numeric sorting, returns path or None.
    """
    if not os.path.isdir(crops_dir):
        return None

    png_files = [
        f for f in os.listdir(crops_dir)
        if f.lower().endswith(".png") and f.startswith("crop_frame_")
    ]
    if not png_files:
        return None

    def parse_frame_number(fname):
        match = re.search(r"crop_frame_(\d+)\.png", fname)
        if match:
            return int(match.group(1))
        return None

    frames_and_files = []
    for pf in png_files:
        num = parse_frame_number(pf)
        if num is not None:
            frames_and_files.append((num, pf))

    if not frames_and_files:
        return None

    frames_and_files.sort(key=lambda x: x[0])
    mid_idx = len(frames_and_files) // 2
    _, middle_file = frames_and_files[mid_idx]
    return os.path.join(crops_dir, middle_file)


################################################################################
# Main
################################################################################

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Process subfolders, each with 'crops_output' containing a middle crop. "
                    "Ask gpt-4o a question with text+image in multi-modal style."
    )
    parser.add_argument("--parent_dir",
                        required=False,
                        default=os.path.join(script_dir, "videos_processed"),
                        help="Path to parent directory with subfolders like 'flipped_G_convex_0'.")
    parser.add_argument("--csv_file",
                        required=False,
                        default=os.path.join(script_dir, "responses.csv"),
                        help="CSV file (in script dir) to store the final [folder, shape, number, response].")
    parser.add_argument("--percentage", type=int, default=100,
                        help="Integer from 0..100. We process that fraction of subfolders in alphabetical order.")
    args = parser.parse_args()

    # Create logs folder
    logs_folder = os.path.join(script_dir, "causality_score_logs")
    os.makedirs(logs_folder, exist_ok=True)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_folder, f"{now_str}.log")

    log_f = open(log_path, "w", encoding="utf-8")

    def log_line(msg):
        print(msg)
        log_f.write(msg + "\n")

    log_line("======================================================")
    log_line("REAL GPT-4o multi-modal requests: Causality Score Script - Starting")
    log_line(f"parent_dir = {args.parent_dir}")
    log_line(f"csv_file   = {args.csv_file}")
    log_line(f"percentage = {args.percentage}")
    log_line(f"Log file   = {log_path}")
    log_line("======================================================\n")

    # 1) Read existing CSV into a dict
    existing_data = read_csv_to_dict(args.csv_file, log_line)
    # existing_data is { folder_name: (shape, number, response) }

    # 2) Build the requester
    mm_requester = MultiModalCSVRequester(csv_file=args.csv_file, log_func=log_line)

    # 3) Gather subfolders
    subfolders = [
        d for d in os.listdir(args.parent_dir)
        if os.path.isdir(os.path.join(args.parent_dir, d))
    ]
    subfolders.sort()
    total_folders = len(subfolders)

    log_line(f"Found {total_folders} total subfolders in '{args.parent_dir}'.")
    fraction = max(0, min(args.percentage, 100)) / 100.0
    count_to_process = round(fraction * total_folders)
    log_line(f"We will process {count_to_process} subfolders (alphabetical order).")
    log_line("")

    subfolders_to_process = subfolders[:count_to_process]

    # 4) Process subfolders
    for folder_name in subfolders_to_process:
        log_line(f"Processing folder: {folder_name}")

        shape_label, number_label = parse_folder_name(folder_name)
        if shape_label is None or number_label is None:
            log_line("   => Folder does not contain 'convex'/'concave'; skipping.\n")
            continue

        # Check if this folder is in existing_data
        folder_existing = existing_data.get(folder_name)

        # If we have a response that is purely numeric => skip
        if folder_existing:
            prev_resp = folder_existing[2]
            if is_number(prev_resp):
                log_line(f"   => Already has numeric response='{prev_resp}' => skipping.\n")
                continue

        # Otherwise, we do a new request
        # find the single 'middle' crop
        folder_path = os.path.join(args.parent_dir, folder_name)
        crops_dir = os.path.join(folder_path, "crops_output")
        middle_crop_path = select_middle_crop(crops_dir)
        if not middle_crop_path:
            log_line("   => No suitable middle crop found; skipping.\n")
            continue

        log_line(f"   => Middle crop: {middle_crop_path}")

        question_text = build_question_text(folder_name)
        log_line("   => Final question:")
        log_line(question_text + "\n")

        # run the request
        new_response = mm_requester.run_request(
            folder_name=folder_name,
            shape_label=shape_label,
            number_label=number_label,
            image_path=middle_crop_path,
            question_text=question_text,
        )

        # store/update in existing_data
        existing_data[folder_name] = (shape_label, number_label, new_response)
        log_line(f"   => Updated in-memory data for folder '{folder_name}'.\n")

    # 5) rewrite CSV in alphabetical order
    write_dict_to_csv(args.csv_file, existing_data, log_line)

    log_line("All requested subfolders processed. Script finished.")
    log_f.close()


if __name__ == "__main__":
    main()
