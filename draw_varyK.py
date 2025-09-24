"""
======================================================================
DRAW_VARYK --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2025, ZiLiang, all rights reserved.
    Created: 16 九月 2025
======================================================================
"""


# ------------------------ Code --------------------------------------


import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
from collections import OrderedDict
import pickle
import os

def load_json_data(json_file: str) -> List[Dict]:
    """
    Load the JSON file containing prompt data.
    
    Args:
    - json_file (str): Path to the JSON file.
    
    Returns:
    - List[Dict]: List of prompt data dictionaries.
    """
    with open(json_file, 'r', encoding='utf8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    return data

def compute_statistics(
    data: List[Dict],
    K_step: int = 100,
    N_repeats: int = 20,
    min_K: int = 2,
    max_K: int = 20000
) -> Dict:
    """
    Compute statistics for K-DPS vs ideal DPS (full set) across varying K.
    
    Args:
    - data (List[Dict]): Loaded JSON data.
    - K_step (int): Step size for K values.
    - N_repeats (int): Number of random subsets per K.
    - min_K (int): Minimum K value.
    - max_K (int): Maximum K value.
    
    Returns:
    - Dict: Statistics including Ks, phis_per_prompt, phi_fulls.
    """
    Ks = list(range(min_K, max_K + 1, K_step))
    phis_per_prompt = []
    phi_fulls = []
    
    valid_data = [d for d in data if 'error' not in d and len(d['generated_samples']) >= min_K]
    
    for prompt_data in valid_data:
        full_logps = np.array([s['log_prob'] for s in prompt_data['generated_samples']])
        sorted_full = np.sort(full_logps)[::-1]
        if len(sorted_full) < 2:
            continue
        delta_full = sorted_full[0] - sorted_full[1]
        phi_full = delta_full ** 2
        phi_fulls.append(phi_full)
        
        n_Ks = len(Ks)
        prompt_phis = np.zeros((n_Ks, N_repeats))
        
        for i, K in enumerate(Ks):
            n_samples = len(full_logps)
            for r in range(N_repeats):
                if K >= n_samples:
                    subset_logps = full_logps.copy()
                else:
                    idx = np.random.choice(n_samples, K, replace=False)
                    subset_logps = full_logps[idx]
                
                sorted_sub = np.sort(subset_logps)[::-1]
                if len(sorted_sub) < 2:
                    delta = 0.0
                else:
                    delta = sorted_sub[0] - sorted_sub[1]
                prompt_phis[i, r] = delta ** 2
        
        phis_per_prompt.append(prompt_phis)
    
    return {
        'Ks': np.array(Ks),
        'phis_per_prompt': np.array(phis_per_prompt),
        'phi_fulls': np.array(phi_fulls)
    }

def save_statistics(stats: Dict, pickle_file: str):
    """
    Save computed statistics to a pickle file.
    
    Args:
    - stats (Dict): Computed statistics.
    - pickle_file (str): Path to save the pickle file.
    """
    with open(pickle_file, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Statistics saved to {pickle_file}")

def load_statistics(pickle_file: str) -> Dict:
    """
    Load statistics from a pickle file.
    
    Args:
    - pickle_file (str): Path to the pickle file.
    
    Returns:
    - Dict: Loaded statistics.
    """
    with open(pickle_file, 'rb') as f:
        stats = pickle.load(f)
    print(f"Statistics loaded from {pickle_file}")
    return stats

def plot_kdps(
    stats: Dict,
    output_pdf: str,
    n_prompts: int = None,
    high_k_ratio: float = 0.2
):
    """
    Plot 2x1 figure for K-DPS: full K range and high K range.
    
    Args:
    - stats (Dict): Output from compute_statistics.
    - output_pdf (str): Path to save the PDF figure.
    - n_prompts (int): Number of prompts to plot (default: all).
    - high_k_ratio (float): Fraction of K range to consider as 'high K'.
    """
    Ks = stats['Ks']
    phis = stats['phis_per_prompt']  # (n_prompts, n_Ks, n_repeats)
    n_prompts_total = phis.shape[0]
    if n_prompts is None:
        n_prompts = n_prompts_total
    
    mean_phis_per_prompt = np.mean(phis, axis=2)  # (n_prompts, n_Ks)
    
    # Create 2x1 figure
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 3))
    
    # Plot 1: Full K range
    for i in range(min(n_prompts, n_prompts_total)):
        ax1.plot(Ks, mean_phis_per_prompt[i], alpha=0.3, color='blue', linewidth=0.5)
    ax1.set_xlabel('K')
    ax1.set_ylabel('K-DPS')
    ax1.set_title('K-DPS vs K')
    ax1.grid(True, alpha=0.3)
    
    # # Plot 2: High K range (last high_k_ratio fraction)
    # high_k_start = int((1 - high_k_ratio) * len(Ks))
    # Ks_high = Ks[high_k_start:]
    # for i in range(min(n_prompts, n_prompts_total)):
    #     ax2.plot(Ks_high, mean_phis_per_prompt[i, high_k_start:], alpha=0.3, color='blue', linewidth=0.5)
    # ax2.set_xlabel('K')
    # ax2.set_ylabel('K-DPS')
    # ax2.set_title('K-DPS vs K (High K Range)')
    # ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"K-DPS figure saved to {output_pdf}")

def plot_abs_error(
    stats: Dict,
    output_pdf: str,
    n_prompts: int = None,
    high_k_ratio: float = 0.2
):
    """
    Plot 2x1 figure for Absolute Error: full K range and high K range.
    
    Args:
    - stats (Dict): Output from compute_statistics.
    - output_pdf (str): Path to save the PDF figure.
    - n_prompts (int): Number of prompts to plot (default: all).
    - high_k_ratio (float): Fraction of K range to consider as 'high K'.
    """
    Ks = stats['Ks']
    phis = stats['phis_per_prompt']  # (n_prompts, n_Ks, n_repeats)
    phi_fulls = stats['phi_fulls']
    n_prompts_total = phis.shape[0]
    if n_prompts is None:
        n_prompts = n_prompts_total
    
    mean_phis_per_prompt = np.mean(phis, axis=2)  # (n_prompts, n_Ks)
    abs_errors_per_prompt = np.abs(mean_phis_per_prompt - phi_fulls[:, np.newaxis])
    
    # Create 2x1 figure
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 3))
    
    # Plot 1: Full K range
    for i in range(min(n_prompts, n_prompts_total)):
        ax1.plot(Ks, abs_errors_per_prompt[i], alpha=0.3, color='blue', linewidth=0.5)
    ax1.set_xlabel('K')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Absolute Error vs K')
    ax1.grid(True, alpha=0.3)
    
    # # Plot 2: High K range (last high_k_ratio fraction)
    # high_k_start = int((1 - high_k_ratio) * len(Ks))
    # Ks_high = Ks[high_k_start:]
    # for i in range(min(n_prompts, n_prompts_total)):
    #     ax2.plot(Ks_high, abs_errors_per_prompt[i, high_k_start:], alpha=0.3, color='blue', linewidth=0.5)
    # ax2.set_xlabel('K')
    # ax2.set_ylabel('Absolute Error')
    # ax2.set_title('Absolute Error vs K (High K Range)')
    # ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Absolute Error figure saved to {output_pdf}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot K-DPS statistics from JSON data.")
    parser.add_argument("--json_file",
                        type=str, required=True,
                        help="Path to the input JSON file.")
    parser.add_argument("--pickle_file",
                        type=str,
                        default="k_dps_stats.pkl",
                        help="Path to save/load the statistics pickle file.")
    parser.add_argument("--kdps_pdf",
                        type=str,
                        default="varyk-kdps.pdf",
                        help="Path to save the K-DPS PDF figure.")
    parser.add_argument("--abs_error_pdf",
                        type=str,
                        default="varyk-error.pdf",
                        help="Path to save the Absolute Error PDF figure.")
    parser.add_argument("--K_step",
                        type=int, default=100,
                        help="Step size for K values.")
    parser.add_argument("--N_repeats",
                        type=int, default=5,
                        help="Number of repeats per K.")
    parser.add_argument("--n_prompts_plot",
                        type=int, default=None,
                        help="Number of prompts to plot thin lines for (default: all).")
    
    args = parser.parse_args()
    
    # Check if pickle file exists
    if os.path.exists(args.pickle_file):
        stats = load_statistics(args.pickle_file)
    else:
        # Load data and compute statistics
        data = load_json_data(args.json_file)
        stats = compute_statistics(
            data,
            K_step=args.K_step,
            N_repeats=args.N_repeats,
            min_K=10,
            max_K=20000
        )
        save_statistics(stats, args.pickle_file)
    
    # Plot K-DPS
    plot_kdps(
        stats,
        args.kdps_pdf,
        n_prompts=args.n_prompts_plot
    )
    
    # Plot Absolute Error
    plot_abs_error(
        stats,
        args.abs_error_pdf,
        n_prompts=args.n_prompts_plot
    )

if __name__ == "__main__":
    main()
