# VSer: Vulnerability Scanning on LLMs

This repository contains the **official implementation** of the paper
**“Decision Potential Surface: A Theoretical and Practical Approximation of LLM's Decision Boundary”**.
We provide everything you need to (1) compute the *K-grained Decision Potential Surface* (K-DPS) on any open-weight LLM, (2) reproduce all empirical results, and (3) visualize the decision-boundary approximation.

---

## Quick Start

### 1. Environment

```bash
# Python ≥ 3.9, CUDA ≥ 11.8 recommended
pip install -r dps_re.txt
```

### 2. Reproduce All Experiments 
Step 1: Generate preprared data.

```sh
bash scripts/1.1.parallel_kdps_varyK.sh
bash scripts/1.2.parallel_kdps_different_dataset.sh
bash scripts/1.1.parallel_kdps_different_model.sh
```

Step 2: Reproduce experiments of Section 5.2

```sh
bash scripts/2.1.compute_draw_varyK.sh
```

Step 3: Reproduce experiments of Section 5.3 & Appendix

```sh
bash scripts/2.2.compute_draw_concentration.sh
```

Step 4: Reproduce experiments of Section 5.4

```sh
python 5.run_kdps_varyDataset.py
```

Step 5: Reproduce experiments on Appendix

```sh
python 5.1.run_kdps_varyModels.py
```

---

## Code Map

| File | Purpose | Core API |
|---|---|---|
| `KDPS.py` | Single-GPU K-DPS calculator | `compute_k_dps()` |
| `KDPS_parallel.py` | Multi-GPU (vLLM + Ray) calculator | `compute_k_dps()` |
| `draw_varyK.py` | Plot K-DPS & absolute error vs. K | `plot_kdps()`, `plot_abs_error()` |
| `draw_concentration.py` | Empirical tail-probability plots | `compute_concentration_stats()` |
| `draw_R_K.py` | R_K surrogate curves | `compute_rk_statistics()` |
| `5.run_kdps_varyDataset.py` | Cross-dataset contour heat-maps | `load_dataset_data()` |
| `5.1.run_kdps_varyModels.py` | Cross-model contour heat-maps | `load_dataset_data()` |
| `text2vector.py` | LLM-based sentence embeddings | `compute_text_embeddings_llm()` |
| `utils.py` | Text preprocessing utilities | `weighted_middle_path4edit_distance()` |

---

## Minimal Example

Compute K-DPS on your own prompts in **3 lines**:

```python
from KDPS import compute_k_dps
phi_values = compute_k_dps(["Hello world", "Climate change is"],
                           K=100, model_name="gpt2", max_new_tokens=20)
print(phi_values)          # [0.42, 1.87]  (example outputs)
```

---

## Output Structure

After running any pipeline script you will obtain

```
results/
├─ k_dps_output.json          # Raw prompt / samples / log-probs / $/Phi_K$
├─ *_stats.pkl                # Cached statistics for fast re-plotting
├─ varyk-kdps.pdf
├─ concentration_full.pdf
├─ kdps_contour_plots.pdf
└─ …
```
