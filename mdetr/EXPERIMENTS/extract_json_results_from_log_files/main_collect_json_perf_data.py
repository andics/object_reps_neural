import os
import re
import json
import shutil
from pathlib import Path


def extract_metrics_from_log(log_content):
    # Use regex to extract the content between the first '{' and the last '}'
    match = re.search(r'(\{.*\})', log_content, re.DOTALL)
    if match:
        metrics = match.group(1)

        metrics = metrics.replace("'", "\"")  # Replace single quotes with double quotes
        metrics = re.sub(r'(\w+):', r'"\1":', metrics)  # Add quotes around keys if missing

        return json.loads(metrics)  # Convert the string to a JSON object
    else:
        raise ValueError("Metrics not found in log file.")


def create_output_directories_and_files(input_dir, output_dir):
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each log file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.log'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                log_content = file.read()
                metrics = extract_metrics_from_log(log_content)

                # Determine the output folder name based on the log file name
                if "var" in filename:
                    model_type = "var"
                elif "equiconst" in filename:
                    model_type = "equiconst"
                else:
                    raise ValueError(f"Unknown model type in filename: {filename}")

                perc_match = re.search(r'(\d+perc)', filename)
                if perc_match:
                    percentage = perc_match.group(1)
                else:
                    raise ValueError(f"Percentage not found in filename: {filename}")

                output_subdir_name = f"gqa_testdev_finetune_balanced_{model_type}_{percentage}"
                output_subdir_path = os.path.join(output_dir, output_subdir_name)
                Path(output_subdir_path).mkdir(parents=True, exist_ok=True)

                # Write metrics to a results.json file
                results_file_path = os.path.join(output_subdir_path, 'results.json')
                with open(results_file_path, 'w') as results_file:
                    json.dump(metrics, results_file, indent=4)

                # Copy the original log file to the output directory
                shutil.copy(file_path, output_subdir_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract metrics from log files and save as JSON.")
    parser.add_argument('--input_dir', type=str, help="The directory containing the .log files.")
    parser.add_argument('--output_dir', type=str, help="The directory where output should be stored.")

    args = parser.parse_args()

    create_output_directories_and_files(args.input_dir, args.output_dir)