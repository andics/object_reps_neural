import os
import json
import matplotlib.pyplot as plt
import argparse


# Function to plot the metrics
def plot_metric_from_logs(storage_dir, epoch, save_dir):
    categories = ["variable", "equiconst", "baseline"]
    metric_values = {}
    checkpoint_numbers = set()

    for item in os.listdir(storage_dir):
        if os.path.isdir(os.path.join(storage_dir, item)) and item.startswith("finetuned_"):
            parts = item.split("_")
            if len(parts) < 4:
                continue
            checkpoint_number = ''.join(filter(str.isdigit, parts[1]))
            category = parts[2]
            if category not in categories:
                continue
            checkpoint_numbers.add(int(checkpoint_number))

            log_path = os.path.join(storage_dir, item, "log.txt")
            if os.path.exists(log_path):
                with open(log_path, 'r') as file:
                    lines = file.readlines()
                    if len(lines) >= epoch:
                        data = json.loads(lines[epoch - 1])
                        for key, value in data.items():
                            if key.startswith("test_gqa_accuracy_"):
                                if key not in metric_values:
                                    metric_values[key] = {cat: [] for cat in categories}
                                metric_values[key][category].append((int(checkpoint_number), value))

    num_metrics = len(metric_values)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 6 * num_metrics))

    for i, (metric, values_by_category) in enumerate(metric_values.items()):
        for category, values in values_by_category.items():
            if values:
                x_values, y_values = zip(*values)
                axs[i].plot(sorted(x_values), [y for _, y in sorted(zip(x_values, y_values))], marker='o',
                            label=category)
        #Metric name is too long, so we need to split it
        metric_name_for_display = "_".join(metric.split("_")[3:])

        axs[i].set_title(f"Num pretrain epochs: x-axis; Num GQA finetuning epochs: {epoch}; GQA eval metric: y-axis")
        axs[i].set_xlabel("Num pretrain epochs")
        axs[i].set_ylabel(f"{metric_name_for_display} after {epoch} GQA finetuning epochs")
        axs[i].legend()

    # Finally, save the plot using the current date and time as name
    if save_dir:
        plt.savefig(os.path.join(save_dir, "plot_all_" + str(epoch) + ".png"))
    else:
        print("No save directory given, plot not saved.")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics from training logs.")
    parser.add_argument('--storage_dir', type=str, required=True,
                        help="Directory where 'finetuned_' folders are located")
    parser.add_argument('--epoch', type=int, required=True, help="Finetuning epoch to extract metrics from")
    parser.add_argument('--save_dir', type=str, help="Path to save the plot to")

    args = parser.parse_args()

    plot_metric_from_logs(args.storage_dir, args.epoch, args.save_dir)