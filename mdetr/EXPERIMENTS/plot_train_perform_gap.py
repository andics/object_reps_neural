import os
import json
import matplotlib.pyplot as plt

def plot_metric_from_logs(storage_dir, epoch, metric_to_plot):
    # Initialize a dictionary to hold the metric values for each category
    categories = ["variable", "equiconst", "baseline"]
    metric_values = {category: [] for category in categories}
    checkpoint_numbers = []

    # Loop through every item in the given storage_dir
    for item in os.listdir(storage_dir):
        # Check if item is a directory and starts with "finetuned_"
        if os.path.isdir(os.path.join(storage_dir, item)) and item.startswith("finetuned_"):
            # Extract the checkpoint number and the category from the folder name
            parts = item.split("_")
            if len(parts) < 4:  # Ensure the folder name has the expected parts
                continue
            checkpoint_number = parts[1]
            # Extract only the digits from the checkpoint_number string
            checkpoint_number = ''.join(filter(str.isdigit, checkpoint_number))
            category = parts[2]
            # Ensure category is one of the expected values
            if category not in metric_values:
                continue

            # Attempt to read the log.txt file
            log_path = os.path.join(storage_dir, item, "log.txt")
            if os.path.exists(log_path):
                with open(log_path, 'r') as file:
                    lines = file.readlines()
                    # Load the JSON object for the specified epoch
                    try:
                        # Ensure there are enough epochs logged in the file
                        if len(lines) >= epoch:
                            data = json.loads(lines[epoch - 1])  # epochs are 1-indexed in this context
                            # Extract the specified metric
                            if metric_to_plot in data:
                                metric_values[category].append(tuple([int(checkpoint_number),
                                                                     data[metric_to_plot]]))
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from file {log_path}")

    # After collecting all data, plot it
    for category, checkpoints_and_metrics in metric_values.items():
        # Unzipping the list of (x, y) tuples into two lists for plotting
        x_values, y_values = zip(*checkpoints_and_metrics)
        plt.plot(x_values, y_values, marker='o', label=category)

    plt.xlabel("Checkpoint Number")
    plt.ylabel(metric_to_plot)
    plt.title(f"Metric {metric_to_plot} across checkpoints")
    plt.legend()
    plt.show()

# Example usage:
storage_dir = "Q:/Projects/Variable_resolution/Programming/mdetr/trained_models"
epoch = 7  # Assuming we're interested in the 3rd epoch
#metric_to_plot = "test_gqa_accuracy_answer_total_unscaled"
metric_to_plot = "test_gqa_accuracy_answer_rel_unscaled"
plot_metric_from_logs(storage_dir, epoch, metric_to_plot)

# Since the code requires access to the filesystem to retrieve and parse the files,
# which isn't possible in this environment, you should run it in your local Python environment.
# Replace '/path/to/storage_dir' with the actual path to your storage directory