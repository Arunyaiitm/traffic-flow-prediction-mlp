# Traffic Flow Prediction with MLP — Deliverable 3
### Comparison with Existing Studies and an Iterative-Pruning Improvement

**Authors:** Arunya (ZDA24B031) & Srijan Reddy Sankepally (ZDA24B007)
**Course:** Foundations of Machine Learning

---

## Overview

Deliverable 3 has two parts, building on our D1 (baseline models) and D2 (compression) work:

1. **Comparison with existing studies** — we compare our traffic-prediction results
   against three papers from our literature survey.
2. **An implemented improvement** — we replace D2's one-shot pruning with *iterative*
   (gradual) pruning, and evaluate it across three random seeds with error bars.

## Repository contents

| File | Description |
|------|-------------|
| `D3_Dataset_1.ipynb` | Huawei dataset — baseline MLP, one-shot pruning (D2) vs iterative pruning (D3), 3 seeds, error bars. |
| `D3_Dataset_2.ipynb` | Metro Interstate dataset — baseline, one-shot vs iterative pruning, plus Prune+Coreset one-shot vs iterative, 3 seeds. |
| `traffic-prediction-dataset.csv` | Huawei 6-cross traffic data (5-minute readings). |
| `Metro_Interstate_Traffic_Volume.csv` | Metro Interstate hourly traffic data. |
| `D3_Report.pdf` | Report — comparison with studies, the improvement, tables, figures, error bars. |

## Datasets

1. **Huawei** (Navarro-Espinoza paper) — 16,128 readings, 6 sensors, every 5 min. Predict
   the next 5 minutes from the past hour (4 lanes).
2. **Metro Interstate** (Das & MSTIM papers) — 48,204 hourly records with weather/holiday
   features (38,511 after cleaning). Predict the next hour from the past 24 hours.

## Method

- **Models / architecture (unchanged from D1/D2):** Huawei MLP 48-128-64-4; Metro MLP
  456-128-64-32-16-1. ReLU activations, Adam optimizer, 20 epochs, MinMax scaling,
  chronological splits.
- **Improvement — iterative pruning:** instead of pruning to 80% sparsity in one step
  (D2, one-shot), prune gradually (20% → 40% → 60% → 80%) with a fine-tune after each
  step so the surviving weights adapt.
- **Evaluation:** every configuration is run over seeds {1, 7, 13}; results are reported
  as mean ± standard deviation (error bars).

## How to run

The notebooks are designed for Google Colab.

1. Open a notebook in Colab.
2. Run the first cell to import dependencies (`numpy`, `pandas`, `matplotlib`,
   `scikit-learn`, `torch`).
3. When prompted, upload the corresponding CSV
   (`traffic-prediction-dataset.csv` for Dataset 1,
   `Metro_Interstate_Traffic_Volume.csv` for Dataset 2).
4. Run the remaining cells top to bottom. Training prints every epoch.

Note: the Metro notebook trains many models over three seeds and can take a while; keep
the Colab tab active.

## Results (summary)

**Comparison with existing studies**

| Dataset | Our best model | Our R² | Study | Their result |
|---|---|---|---|---|
| Huawei | MLP + pruning | 0.947 | Navarro-Espinoza | R² ≈ 0.93 |
| Metro | MLP + prune + coreset | 0.90 | Das / MSTIM | LSTM/GRU, CNN+attention (deep) |

**Improvement — one-shot vs iterative pruning (R², mean ± std over 3 seeds)**

| Method | Huawei | Metro |
|---|---|---|
| Baseline MLP | 0.9437 ± 0.0023 | 0.8826 ± 0.0192 |
| One-shot prune 80% (D2) | **0.9472 ± 0.0008** | 0.8928 ± 0.0069 |
| Iterative prune 80% (D3) | 0.9454 ± 0.0007 | 0.8761 ± 0.0065 |
| Prune + Coreset one-shot (D2) | — | **0.8998 ± 0.0038** |
| Prune + Coreset iterative (D3) | — | 0.8818 ± 0.0017 |

**Finding:** iterative pruning did not outperform one-shot pruning on these
small-to-medium networks; one-shot is retained. Iterative pruning's advantage applies
mainly to large, over-parameterized models. Our best overall model remains the
prune+coreset MLP (≈0.90 on Metro, 0.947 on Huawei) at roughly one-third the storage.

## Reference papers

1. Navarro-Espinoza et al. (2022), *Technologies* — Huawei intersection dataset.
2. Das (2023), arXiv:2303.12643 — Metro Interstate dataset (LSTM/GRU).
3. MSTIM (2025), arXiv:2504.13576 — Metro Interstate dataset (CNN + LSTM + attention).
