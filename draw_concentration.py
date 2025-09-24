"""
======================================================================
DRAW_CONCENTRATION --- 

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
from typing import Dict, List
from scipy.special import logsumexp
import pickle
import os
import math

def compute_concentration_stats(
    json_file: str,
    K_step: int = 100,
    N_repeats: int = 50,
    lambda_val: float = 0.1,
    min_K: int = 2,
    max_K: int = 20000
) -> Dict:
    """
    Compute concentration statistics from JSON data, excluding theoretical bounds.
    
    Args:
    - json_file (str): Path to the JSON file.
    - K_step (int): Step size for K values.
    - N_repeats (int): Number of random subsets per K.
    - lambda_val (float): Lambda value for the tail probability.
    - min_K (int): Minimum K value.
    - max_K (int): Maximum K value.
    
    Returns:
    - Dict: Statistics including 'Ks', 'empirical_probs', 'lambda_val'.
    """
    # Load data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    Ks = list(range(min_K, max_K + 1, K_step))
    n_Ks = len(Ks)
    valid_data = [d for d in data if 'error' not in d and len(d['generated_samples']) >= min_K]
    
    if not valid_data:
        return {'Ks': np.array(Ks), 'empirical_probs': np.zeros(n_Ks), 'lambda_val': lambda_val}
    
    # For each prompt, compute phi_full and logsumexp
    prompt_info = []
    for prompt_data in valid_data:
        full_logps = np.array([s['log_prob'] for s in prompt_data['generated_samples']])
        if len(full_logps) < 2:
            continue
        sorted_full = np.sort(full_logps)[-2:]
        phi_full = (sorted_full[0] - sorted_full[1]) ** 2
        logsumexp_val = logsumexp(full_logps)
        prompt_info.append({
            'phi_full': phi_full,
            'full_logps': full_logps
        })
    
    if not prompt_info:
        return {'Ks': np.array(Ks), 'empirical_probs': np.zeros(n_Ks), 'lambda_val': lambda_val}
    
    # Compute empirical probs averaged over prompts
    empirical_probs_all = np.zeros((len(prompt_info), n_Ks))
    
    for p_idx, info in enumerate(prompt_info):
        phi_full = info['phi_full']
        full_logps = info['full_logps']
        n_samples = len(full_logps)
        
        for i, K in enumerate(Ks):
            errors = []
            
            for r in range(N_repeats):
                if K >= n_samples:
                    subset_logps = full_logps.copy()
                else:
                    idx = np.random.choice(n_samples, K, replace=False)
                    subset_logps = full_logps[idx]
                
                if len(subset_logps) < 2:
                    continue
                
                sorted_sub = np.sort(subset_logps)[-2:]
                phi_k = (sorted_sub[0] - sorted_sub[1]) ** 2
                error = abs(phi_k - phi_full)
                errors.append(error)
            
            if errors:
                emp_prob = np.mean(np.array(errors) >= lambda_val)
                empirical_probs_all[p_idx, i] = emp_prob
    
    # Average over prompts
    empirical_probs = np.mean(empirical_probs_all, axis=0)
    
    return {
        'Ks': np.array(Ks),
        'empirical_probs': empirical_probs,
        'lambda_val': lambda_val
    }

def save_statistics(stats_list: List[Dict], pickle_file: str):
    """
    Save computed statistics to a pickle file.
    
    Args:
    - stats_list (List[Dict]): List of computed statistics.
    - pickle_file (str): Path to save the pickle file.
    """
    with open(pickle_file, 'wb') as f:
        pickle.dump(stats_list, f)
    print(f"Statistics saved to {pickle_file}")

def load_statistics(pickle_file: str) -> List[Dict]:
    """
    Load statistics from a pickle file.
    
    Args:
    - pickle_file (str): Path to the pickle file.
    
    Returns:
    - List[Dict]: Loaded statistics.
    """
    with open(pickle_file, 'rb') as f:
        stats_list = pickle.load(f)
    print(f"Statistics loaded from {pickle_file}")
    return stats_list

def plot_concentration(
    stats_list: List[Dict],
    output_pdf_full: str,
    output_pdf_high: str,
    high_k_ratio: float = 0.2
):
    """
    Plot the concentration empirical probabilities in two 2x4 grids: full K range and high K range.
    
    Args:
    - stats_list (List[Dict]): List of outputs from compute_concentration_stats for different lambda values.
    - output_pdf_full (str): Path to save the full K range PDF figure.
    - output_pdf_high (str): Path to save the high K range PDF figure.
    - high_k_ratio (float): Fraction of K range to consider as 'high K'.
    """
    # Plot for full K range
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, stats in enumerate(stats_list):
        # if idx==1:
        #     continue
        Ks = stats['Ks']
        empirical = stats['empirical_probs']
        lambda_val = stats['lambda_val']
        
        ax = axes[idx]
        # ax = axes[idx if idx<1 else idx-1]
        ax.plot(Ks, empirical, label='Empirical Pr(|error| ≥ λ)', linewidth=2, color='blue')
        ax.set_xlabel('K')
        ax.set_ylabel('Tail Probability')
        ax.set_title(f'λ = $e^{(idx+1)*2}$')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_pdf_full, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Full range concentration figure saved to {output_pdf_full}")
    
    # Plot for high K range
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, stats in enumerate(stats_list):
        # if idx==1:
        #     continue
        Ks = stats['Ks']
        empirical = stats['empirical_probs']
        lambda_val = stats['lambda_val']
        
        # High K range (last high_k_ratio fraction)
        high_k_start = int((1 - high_k_ratio) * len(Ks))
        Ks_high = Ks[high_k_start:]
        empirical_high = empirical[high_k_start:]
        
        ax = axes[idx]
        # ax = axes[idx if idx < 1 else idx + 1]  # Skip the second subplot
        ax.plot(Ks_high, empirical_high, label='Empirical Pr(|error| ≥ λ)', linewidth=2, color='blue')
        ax.set_xlabel('K')
        ax.set_ylabel('Tail Probability')
        ax.set_title(f'λ = {lambda_val} (High K Range)')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_pdf_high, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"High K range concentration figure saved to {output_pdf_high}")

def main():
    parser = argparse.ArgumentParser(description="Compute and plot concentration statistics from K-DPS JSON data.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--pickle_file", type=str, default="concentration_stats.pkl", help="Path to save/load the statistics pickle file.")
    parser.add_argument("--output_pdf_full", type=str, default="concentration_full.pdf", help="Path to save the full K range PDF figure.")
    parser.add_argument("--output_pdf_high", type=str, default="concentration_high_k.pdf", help="Path to save the high K range PDF figure.")
    parser.add_argument("--K_step", type=int, default=100, help="Step size for K values.")
    parser.add_argument("--N_repeats", type=int, default=50, help="Number of repeats per K.")
    parser.add_argument("--lambda_vals", type=float, nargs='+',
    # default=[math.exp(2), math.exp(4), math.exp(6), math.exp(8),
            # math.exp(10), math.exp(12), math.exp(14), math.exp(16)],
    default=[math.exp(2), math.exp(4), math.exp(6), math.exp(8),],
    help="List of lambda values for the tail probability.")
    
    args = parser.parse_args()
    
    # Check if pickle file exists
    if os.path.exists(args.pickle_file):
        stats_list = load_statistics(args.pickle_file)
    else:
        # Compute statistics for each lambda_val
        stats_list = []
        for lambda_val in args.lambda_vals:
            stats = compute_concentration_stats(
                args.json_file,
                K_step=args.K_step,
                N_repeats=args.N_repeats,
                lambda_val=lambda_val,
                min_K=2,
                max_K=20000
            )
            stats_list.append(stats)
        save_statistics(stats_list, args.pickle_file)
    
    plot_concentration(stats_list, args.output_pdf_full, args.output_pdf_high)

if __name__ == "__main__":
    main()
