import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as patheffects

def plot_models_eval1(csv_paths, model_names, zoom_range=(0.87, 0.880)):
    if len(csv_paths) != len(model_names):
        raise ValueError("The number of CSV paths and model names must match.")

    metrics = ['accuracy', 'precision', 'recall', 'f1-score']

    performance_data = []

    for csv_path, model_name in zip(csv_paths, model_names):
        model_data = pd.read_csv(csv_path).iloc[0].values

        if len(model_data) != len(metrics):
            raise ValueError(f"Expected {len(metrics)} metrics in {csv_path}, but got {len(model_data)}.")

        performance_data.append(model_data)

    performance_df = pd.DataFrame(performance_data, columns=metrics, index=model_names)

    fig = plt.figure(figsize=(12, 6))
    ax_main = fig.add_subplot(111)
    x = np.arange(len(metrics))

    # Markers
    markers = ['o', 's', 'D', 'P', 'X', '^', 'v', 'H']

    for i, model_name in enumerate(model_names):
        y_values = performance_df.loc[model_name]
        ax_main.plot(
            x,
            y_values,
            marker=markers[i % len(markers)],
            label=model_name,
            linewidth=1.5
        )

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(metrics)
    ax_main.set_xlabel("Metrics")
    ax_main.set_ylabel("Values")
    ax_main.set_title("Model Performance Comparison")
    ax_main.grid(axis='y', linestyle='--', alpha=0.7)

    legend = ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    ax_main.set_xlim(-0.5, len(metrics) - 0.5)

    ax_main.axhspan(zoom_range[0], zoom_range[1], color='lightgray', alpha=0.5)

    # Zoomed inset
    inset_width = 0.18
    inset_height = 0.30
    inset_x = 0.78
    inset_y = 0.35

    ax_inset = fig.add_axes([inset_x, inset_y, inset_width, inset_height], facecolor='whitesmoke')
    ax_inset.patch.set_alpha(0.5)

    for i, model_name in enumerate(model_names):
        y_values = performance_df.loc[model_name]
        ax_inset.plot(
            x,
            y_values,
            marker=markers[i % len(markers)],
            label=model_name,
            linewidth=1.5
        )

    # Inset settings
    ax_inset.set_xlim(-0.5, len(metrics) - 0.5)
    ax_inset.set_ylim(*zoom_range)
    ax_inset.set_xticks(x)
    ax_inset.set_xticklabels(metrics, fontsize=7, color='black')
    ax_inset.tick_params(axis='y', labelcolor='black')
    ax_inset.grid(axis='y', linestyle='--', alpha=0.7)
    ax_inset.set_title("Better view", fontsize=9, color='black')

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.7, bottom=0.25, top=0.9)

    plt.show()

def plot_models_eval(csv_paths, model_names, zoom_range=(0.87, 0.880)):
    if len(csv_paths) != len(model_names):
        raise ValueError("The number of CSV paths and model names must match.")
    
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#cccccc'
    
    colors = sns.color_palette('bright', n_colors=len(model_names))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']

    performance_data = []
    for csv_path, model_name in zip(csv_paths, model_names):
        model_data = pd.read_csv(csv_path).iloc[0].values
        if len(model_data) != len(metrics):
            raise ValueError(f"Expected {len(metrics)} metrics in {csv_path}, but got {len(model_data)}.")
        performance_data.append(model_data)

    performance_df = pd.DataFrame(performance_data, columns=metrics, index=model_names)

    fig = plt.figure(figsize=(12, 7), dpi=100)
    
    ax_main = fig.add_subplot(111)
    x = np.arange(len(metrics))

    markers = ['o','h','D','P','*','X','p','^',]

    # Plot lines with enhanced styling
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        y_values = performance_df.loc[model_name]
        line = ax_main.plot(
            x, y_values,
            marker=markers[i % len(markers)],
            label=model_name,
            color=color,
            linewidth=2,
            linestyle='-',
            markersize = 7,
            markeredgewidth = 1.5,
            zorder=3,
            markerfacecolor=color,
        )

    ax_main.grid(True, linestyle='--', alpha=0.3, color='gray', zorder=1)
    ax_main.set_axisbelow(True)

    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.tick_params(labelsize=10)

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax_main.set_xlabel("Metrics", fontsize=12, fontweight='bold', labelpad=10)
    ax_main.set_ylabel("Performance Score", fontsize=12, fontweight='bold', labelpad=10)
    
    title = ax_main.set_title("Model Performance Comparison", 
                             pad=20, 
                             fontsize=14, 
                             fontweight='bold')

    ax_main.axhspan(zoom_range[0], zoom_range[1], color='gray', alpha=0.1, zorder=1)

    ax_inset = fig.add_axes([0.78, 0.35, 0.18, 0.30])
    
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.5)

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        y_values = performance_df.loc[model_name]
        ax_inset.plot(
            x, y_values,
            marker=markers[i % len(markers)],
            color=color,
            linewidth=1,
            markersize=3,
            markeredgewidth=1.5,
            zorder=3
        )

    ax_inset.set_xlim(-0.5, len(metrics) - 0.5)
    ax_inset.set_ylim(*zoom_range)
    ax_inset.set_xticks(x)
    ax_inset.set_xticklabels(metrics, fontsize=8)
    ax_inset.tick_params(axis='both', labelsize=8)
    ax_inset.grid(True, linestyle='--', alpha=0.5, color='gray', zorder=1)
    ax_inset.set_title("Zoomed View", fontsize=10, fontweight='bold', pad=8)

    con1 = ConnectionPatch(
        xyA=(-0.5, zoom_range[0]), xyB=(0, zoom_range[0]),
        coordsA="data", coordsB="data",
        axesA=ax_main, axesB=ax_inset,
        color="gray", linestyle=":", alpha=0.5
    )
    con2 = ConnectionPatch(
        xyA=(-0.5, zoom_range[1]), xyB=(0, zoom_range[1]),
        coordsA="data", coordsB="data",
        axesA=ax_main, axesB=ax_inset,
        color="gray", linestyle=":", alpha=0.5
    )
    ax_main.add_artist(con1)
    ax_main.add_artist(con2)

    legend = ax_main.legend(
        bbox_to_anchor=(0.5, -0.15),
        loc='upper center',
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        borderaxespad=1
    )

    plt.subplots_adjust(left=0.1, right=0.7, bottom=0.25, top=0.9)

    return fig, ax_main, ax_inset

csv_paths = ["../Evals/separate_models/ENB0/eval_metrics.csv", "../Evals/separate_models/ENV2M/eval_metrics.csv", "../Evals/separate_models/EVA/eval_metrics.csv", "../Evals/separate_models/FBN/eval_metrics.csv", "../Evals/separate_models/MNV3/eval_metrics.csv", "../Evals/separate_models/MNV4/eval_metrics.csv", "../Evals/separate_models/ResNeXt/eval_metrics.csv", "../Evals/separate_models/ResEmoteNet/eval_metrics.csv", "../Evals/separate_models/Best_ensemble/eval_metrics.csv"]
model_names = ["EfficientNet_B0", "EfficientNet_V2M", "Eva", "FBNetV3b", "MobileNetV3", "MobileNetV4", "ResNeXt50_32x4d", "ResEmoteNet", "Best5Ensemble"]
fig, axm, axi = plot_models_eval(csv_paths, model_names)
plt.show()
