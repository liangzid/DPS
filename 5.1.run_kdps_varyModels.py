"""
======================================================================
5.1.RUN_KDPS_VARYMODELS --- 

    Author: Ann <xxxxxxxx@xxxxxxxxxx.xxxxx>
    Copyright © 2025, ANN, all rights reserved.
======================================================================
"""


# ------------------------ Code --------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import pickle
from mpl_toolkits.mplot3d import Axes3D
from umap import UMAP
import json
from text2vector import compute_text_embeddings_llm
from collections import OrderedDict

def load_dataset_data(
        dataset_name="wikipedia_mini",
        sample_num=50000,
        model_name="meta-llama/Llama-3.2-1B",
        ):
    dataset_ls=[
        "wikipedia_mini",
        "tulu-3-sft",
        "openo1-sft",
        "hh-rlhf",
        "alpaca",
        ]
    assert dataset_name in dataset_ls
    pickle_save_path=f"PREPARE_KDPS_SAVE_{model_name}.pkl"
    if os.path.exists(pickle_save_path):
        with open(pickle_save_path, 'rb') as f:
            data = pickle.load(f)
        return data[0], data[1], data[2]
    else:
        read_path=f"k_dps__parallel_model__{model_name}/k_dps_output.json"
        with open(read_path, 'r', encoding='utf8') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
        textls = []
        kdps = []
        for item in data:
            textls.append(item["prompt"])
            kdps.append(item["phi_k"])
        print(f"Now Begin to compute the sentence embedding")
        embedds = compute_text_embeddings_llm(
            textls,
            model_name=model_name,
            device="cuda",
            batch_size=64,
            max_length=256,
            pooling_strategy="last",
            )

        print("Performing dimensionality reduction with UMAP...")
        reducer = UMAP(n_components=2, n_neighbors=100, min_dist=0.2, random_state=42)
        embeddings_2d = reducer.fit_transform(embedds)

        # Normalize coordinates to [0, 1]
        x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
        y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1
        norm_x = (embeddings_2d[:, 0] - x_min) / x_range
        norm_y = (embeddings_2d[:, 1] - y_min) / y_range
        norm_coords = np.column_stack([norm_x, norm_y])
        embedds=norm_coords

        pickle_save_path=f"PREPARE_KDPS_SAVE_{dataset_name}.pkl"
        with open(pickle_save_path, 'wb') as f:
            pickle.dump([textls, embedds, kdps], f)
        print("Prepare File Saved. Path:", pickle_save_path)
        return textls, embedds, kdps

def create_contour_subplot(ax, embedds, kdps, dataset_name, inte_method="linear"):
    if embedds is None or len(embedds) == 0:
        return None
   
    x = embedds[:, 0]
    y = embedds[:, 1]
    z = kdps
   
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    XI, YI = np.meshgrid(xi, yi)
   
    ZI = griddata((x, y), z, (XI, YI), method=inte_method)

    if inte_method=="linear" or inte_method=="nearest":
        contour = ax.contour(XI, YI, ZI, 30, colors='black', linewidths=0.4)
    else:
        contour = ax.contour(XI, YI, ZI, 30, colors='black', linewidths=0.8)
    ax.clabel(contour,
              inline=False,
              fontsize=5, colors='black')
    im = ax.contourf(XI, YI, ZI, 30, cmap=cm.viridis, alpha=0.8)
    
    # Add scatter points
    ax.scatter(x, y, c=z, cmap=cm.viridis, s=0.5, alpha=0.3)
   
    ax.set_title(f'{dataset_name}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=8)
    ax.set_ylabel('Dimension 2', fontsize=8)
    ax.tick_params(labelsize=7)
   
    return im

def create_heatmap_subplot(ax, embedds, kdps, dataset_name):
    if embedds is None or len(embedds) == 0:
        return None
   
    x = embedds[:, 0]
    y = embedds[:, 1]
    z = kdps
   
    heatmap, xedges, yedges = np.histogram2d(x, y, weights=z, bins=50)
    counts, _, _ = np.histogram2d(x, y, bins=50)
   
    counts[counts == 0] = 1
    heatmap = heatmap / counts
   
    im = ax.imshow(heatmap.T, origin='lower', aspect='auto',
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                  cmap=cm.viridis)
    
    # Add scatter points
    ax.scatter(x, y, c=z, cmap=cm.viridis, s=2, alpha=0.3)
   
    ax.set_title(f'{dataset_name}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=8)
    ax.set_ylabel('Dimension 2', fontsize=8)
    ax.tick_params(labelsize=7)
   
    return im

def create_3d_subplot(ax, embedds, kdps, dataset_name, inte_method="linear"):
    if embedds is None or len(embedds) == 0:
        return None
   
    x = embedds[:, 0]
    y = embedds[:, 1]
    z = kdps
   
    # Create grid for surface
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x, y), z, (XI, YI), method=inte_method)
   
    # Plot surface
    surf = ax.plot_surface(XI, YI, ZI, cmap=cm.viridis, alpha=0.8)
    
    # Add scatter points
    ax.scatter(x, y, z, c=z, cmap=cm.viridis, s=2, alpha=0.3)
   
    ax.set_title(f'{dataset_name}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=8)
    ax.set_ylabel('Dimension 2', fontsize=8)
    ax.set_zlabel('K-DPS Value', fontsize=8)
    ax.tick_params(labelsize=6)
   
    return surf

def create_contour_figure(model_names,inte_method="linear"):
    fig, axes = plt.subplots(1, len(model_names), figsize=(16, 4))
   
    if len(model_names) == 1:
        axes = [axes]
   
    contour_ims = []
   
    for i, model_name in enumerate(model_names):
        print(f"Processing - {model_name}")
        _, embedds, kdps = load_dataset_data(model_name=model_name)
       
        if embedds is not None:
            im = create_contour_subplot(axes[i], embedds, kdps, model_name,
                                        inte_method=inte_method)
            contour_ims.append(im)
    print(kdps)
    for k in kdps:
        if k<0:
            print(k)
            raise Exception
   
    # if contour_ims:
    #     cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    #     fig.colorbar(contour_ims[0], cax=cbar_ax, label='K-DPS Value')
   
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    if inte_method=="linear":
        plt.savefig('vm-kdps_contour_plots_linear.pdf',bbox_inches='tight')
    elif inte_method=="nearest":
        plt.savefig('vm-kdps_contour_plots_nearest.pdf',bbox_inches='tight')
    else:
        plt.savefig('vm-kdps_contour_plots.pdf',bbox_inches='tight')
    plt.close()
    print("Saved. Save to: kdps_contour_plots.pdf")

def create_heatmap_figure(model_names):
    fig, axes = plt.subplots(1, len(model_names), figsize=(16, 4))
   
    if len(model_names) == 1:
        axes = [axes]
   
    heatmap_ims = []
   
    for i, model_name in enumerate(model_names):
        print(f"Processing - {model_name}")
        _, embedds, kdps = load_dataset_data(model_name=model_name)
       
        if embedds is not None:
            im = create_heatmap_subplot(axes[i], embedds, kdps, model_name)
            heatmap_ims.append(im)
   
    if heatmap_ims:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(heatmap_ims[0], cax=cbar_ax, label='K-DPS Value')
   
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig('vm-kdps_heatmap_plots.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved. Save to: kdps_heatmap_plots.pdf")

def create_3d_figure(model_names,inte_method="linear"):
    fig = plt.figure(figsize=(20, 5))
   
    scatter_ims = []
   
    for i, model_name in enumerate(model_names):
        print(f"Processing - {model_name}")
        _, embedds, kdps = load_dataset_data(model_name=model_name)
       
        if embedds is not None:
            ax = fig.add_subplot(1, len(dataset_names), i+1, projection='3d')
            im = create_3d_subplot(ax, embedds, kdps, model_name,inte_method=inte_method)
            scatter_ims.append(im)
   
    if scatter_ims:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(scatter_ims[0], cax=cbar_ax, label='K-DPS Value')
   
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig('vm-kdps_3d_plots.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved. Save to: kdps_3d_plots.pdf")


if __name__=="__main__":
    dataset = "wikipedia_mini"
    models=[
        "meta-llama/Llama-3.2-1B",
        "Qwen/Qwen3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen3-32B",
        ]
    model_names=models
    create_contour_figure(model_names,inte_method="linear")
    create_contour_figure(model_names,inte_method="nearest")
    create_contour_figure(model_names,inte_method="cubic")
    create_heatmap_figure(model_names)
    create_3d_figure(model_names)








