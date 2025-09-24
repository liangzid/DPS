"""
======================================================================
DRAW_R_K --- 

    Author: Ann <xxxxxxxx@xxxxxxxxxx.xxxxx>
    Copyright © 2025, ANN, all rights reserved.
======================================================================
"""


# ------------------------ Code --------------------------------------

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

def compute_rk_statistics(
    json_file: str,
    K_step: int = 100,
    N_repeats: int = 20,
    min_K: int = 2,
    max_K: int = 20000
) -> Dict:
    """
    Compute statistics for R_K vs K from JSON data.
    
    Args:
    - json_file (str): Path to the JSON file.
    - K_step (int): Step size for K values.
    - N_repeats (int): Number of random subsets per K.
    - min_K (int): Minimum K value.
    - max_K (int): Maximum K value.
    
    Returns:
    - Dict: Statistics including 'Ks' and 'mean_rks' (mean R_K over prompts and repeats).
    """
    # Load data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    Ks = list(range(min_K, max_K + 1, K_step))
    n_Ks = len(Ks)
    valid_data = [d for d in data if 'error' not in d and len(d['generated_samples']) >= min_K]
    
    if not valid_data:
        return {'Ks': np.array(Ks), 'mean_rks': np.zeros(n_Ks)}
    
    # Collect all log_probs across prompts for averaging
    all_full_logps = []
    for prompt_data in valid_data:
        full_logps = np.array([s['log_prob'] for s in prompt_data['generated_samples']])
        if len(full_logps) >= min_K:
            all_full_logps.append(full_logps)
    
    if not all_full_logps:
        return {'Ks': np.array(Ks), 'mean_rks': np.zeros(n_Ks)}
    
    # For each K, compute R_K over all prompts and repeats
    rks_all = np.zeros((len(all_full_logps), n_Ks, N_repeats))
    
    for p_idx, full_logps in enumerate(all_full_logps):
        n_samples = len(full_logps)
        for i, K in enumerate(Ks):
            for r in range(N_repeats):
                if K >= n_samples:
                    subset_logps = full_logps.copy()
                else:
                    idx = np.random.choice(n_samples, K, replace=False)
                    subset_logps = full_logps[idx]
                
                if len(subset_logps) < 1:
                    rk = 0.0
                else:
                    rk = np.max(subset_logps) - np.min(subset_logps)
                rks_all[p_idx, i, r] = rk
    
    # Mean R_K over prompts and repeats
    mean_rks = np.mean(rks_all, axis=(0, 2))  # Mean over prompts and repeats: (n_Ks,)
    
    return {
        'Ks': np.array(Ks),
        'mean_rks': mean_rks
    }

def plot_rk_statistics(
    groups: Dict[str, List[Tuple[str, Dict]]],
    output_pdf: str
):
    """
    Plot 1xN figure for R_K vs K, one subplot per dataset.
    
    Args:
    - groups (Dict[str, List[Tuple[str, Dict]]]): {dataset: [(model, stats), ...]}
    - output_pdf (str): Path to save the PDF figure.
    """
    n_datasets = len(groups)
    fig, axs = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5))
    if n_datasets == 1:
        axs = [axs]
    
    for i, (dataset, model_stats_list) in enumerate(groups.items()):
        ax = axs[i]
        for model, stats in model_stats_list:
            ax.plot(stats['Ks'], stats['mean_rks'], label=model, linewidth=2)
        
        ax.set_xlabel('K')
        ax.set_ylabel('R_K')
        ax.set_title(f'Dataset: {dataset}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Compute and plot R_K statistics from multiple JSON files.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file with {'model_dataset': 'path'} dict.")
    parser.add_argument("--output_pdf", type=str, default="rk_statistics.pdf", help="Path to save the output PDF figure.")
    parser.add_argument("--K_step", type=int, default=100, help="Step size for K values.")
    parser.add_argument("--N_repeats", type=int, default=20, help="Number of repeats per K.")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        paths_dict = json.load(f)
    
    # Group by dataset: {dataset: [(model, path), ...]}
    groups = defaultdict(list)
    for key, path in paths_dict.items():
        # Assume key format: "model_dataset"
        parts = key.rsplit('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid key format: {key}. Expected 'model_dataset'.")
        model, dataset = parts
        groups[dataset].append((model, path))
    
    # Compute statistics for each model-dataset
    computed_groups = {}
    for dataset, model_path_list in groups.items():
        model_stats_list = []
        for model, path in model_path_list:
            stats = compute_rk_statistics(
                path,
                K_step=args.K_step,
                N_repeats=args.N_repeats,
                min_K=2,
                max_K=20000
            )
            model_stats_list.append((model, stats))
        computed_groups[dataset] = model_stats_list
    
    # Plot
    plot_rk_statistics(computed_groups, args.output_pdf)

if __name__ == "__main__":
    main()
