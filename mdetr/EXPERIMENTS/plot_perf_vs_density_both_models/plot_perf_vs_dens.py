import os
import json
import re
import matplotlib.pyplot as plt
from pathlib import Path


def convert_txt_to_json(txt_file_path, json_file_path):
    with open(txt_file_path, 'r') as file:
        content = file.read()

        # Use regex to convert to a JSON-compliant string
        content = content.replace("'", "\"")  # Replace single quotes with double quotes
        content = re.sub(r'(\w+):', r'"\1":', content)  # Add quotes around keys if missing

        # Convert the string to a JSON object
        metrics = json.loads(content)

        # Write the JSON object to a .json file
        with open(json_file_path, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)


def extract_accuracy_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        return data['test_gqa_accuracy_answer_total_unscaled']


def plot_accuracy_graph(density_accuracy_dict_var, density_accuracy_dict_equiconst, output_path):
    plt.figure(figsize=(10, 6))

    # Sort densities for plotting
    sorted_densities_var = sorted(density_accuracy_dict_var.keys())
    sorted_densities_equiconst = sorted(density_accuracy_dict_equiconst.keys())

    accuracies_var = [density_accuracy_dict_var[d] for d in sorted_densities_var]
    accuracies_equiconst = [density_accuracy_dict_equiconst[d] for d in sorted_densities_equiconst]

    plt.plot(sorted_densities_var, accuracies_var, marker='o', color='blue', label='Var Model')
    plt.plot(sorted_densities_equiconst, accuracies_equiconst, marker='o', color='green', label='Equiconst Model')

    plt.xlabel('Density (%)')
    plt.ylabel('Accuracy (test_gqa_accuracy_answer_total_unscaled)')
    plt.ylim(0, 1)
    plt.title('Model Performance across Different Densities')
    plt.legend()
    plt.grid(True)

    # Save the plot to the output path
    plt.savefig(output_path)
    plt.show()


def process_directory(input_dir):
    density_accuracy_dict_var = {}
    density_accuracy_dict_equiconst = {}

    for foldername in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, foldername)

        if os.path.isdir(folder_path):
            # Extract density percentage from folder name
            density_match = re.search(r'(\d+)perc', foldername)
            if density_match:
                density = int(density_match.group(1))
            else:
                continue  # Skip if the folder name doesn't match the expected pattern

            # Determine model type
            if "var" in foldername:
                model_type = "var"
            elif "equiconst" in foldername:
                model_type = "equiconst"
            else:
                continue  # Skip if the folder name doesn't match the expected pattern

            # Locate the results file
            json_file_path = os.path.join(folder_path, 'results.json')
            txt_file_path = os.path.join(folder_path, 'results.txt')

            if not os.path.exists(json_file_path) and os.path.exists(txt_file_path):
                # Convert the .txt file to .json if the .json doesn't exist
                convert_txt_to_json(txt_file_path, json_file_path)

            if os.path.exists(json_file_path):
                accuracy = extract_accuracy_from_json(json_file_path)

                if model_type == "var":
                    density_accuracy_dict_var[density] = accuracy
                elif model_type == "equiconst":
                    density_accuracy_dict_equiconst[density] = accuracy

    return density_accuracy_dict_var, density_accuracy_dict_equiconst


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot accuracy from JSON files.")
    parser.add_argument('--input_dir', type=str, help="The parent directory containing model result folders.")
    args = parser.parse_args()

    density_accuracy_dict_var, density_accuracy_dict_equiconst = process_directory(args.input_dir)

    # Output path for the plot
    output_plot_path = os.path.join(os.getcwd(), 'accuracy_comparison_plot.png')

    plot_accuracy_graph(density_accuracy_dict_var, density_accuracy_dict_equiconst, output_plot_path)