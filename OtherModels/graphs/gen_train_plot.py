import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics_evolution(csv_paths, model_names):

    if len(csv_paths) != len(model_names):
        raise ValueError("The number of CSV paths and model names must match.")

    metrics = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1_score']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for csv_path, model_name in zip(csv_paths, model_names):
            data = pd.read_csv(csv_path)

            if metric not in data.columns:
                raise ValueError(f"Missing required metric '{metric}' in {csv_path}.")

            ax.plot(data['epoch'], data[metric], label=model_name)

        ax.set_title(metric.replace('_', ' ').capitalize())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


csv_paths = ["../EfficientNet_B0/metrics_results.csv", "../EfficientNet_V2M/metrics_results.csv", "../Eva/metrics_results.csv", "../FBNetV3b/metrics_results.csv", "../MobileNetV3/metrics_results.csv", "../MobileNetV4/metrics_results.csv", "../ResNeXt50_32x4d/metrics_results.csv"]
model_names = ["EfficientNet_B0", "EfficientNet_V2M", "Eva", "FBNetV3b", "MobileNetV3", "MobileNetV4", "ResNeXt50_32x4d"]
plot_metrics_evolution(csv_paths, model_names)