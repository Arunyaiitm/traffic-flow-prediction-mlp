# Traffic Flow Prediction with a Multilayer Perceptron
### Foundations of Machine Learning — Course Project (Deliverables 1–3)

**Authors:** Arunya (ZDA24B031) & Srijan Reddy Sankepally (ZDA24B007)

---

## Project overview

This project predicts short-term traffic flow and studies how a simple Multilayer
Perceptron (MLP) compares with classical machine-learning models and with published
deep-learning studies — and how far the MLP can be compressed without losing accuracy.
It spans three deliverables:

- **D1 — Baselines & EDA:** datasets, exploratory analysis, literature survey, and a
  first comparison of a classical model vs an MLP.
- **D2 — Compression:** classical baselines (incl. Random Forest) plus pruning,
  quantization, and coreset selection applied to the MLP, with multi-seed error bars.
- **D3 — Comparison & improvement:** comparison against existing studies, and an
  implemented improvement (iterative pruning) evaluated against one-shot pruning.

## Datasets

1. **Huawei intersection dataset** (Navarro-Espinoza paper) — 16,128 readings from six
   sensors every 5 minutes over 56 days. Four sensors model the four lanes of one
   intersection. Task: predict the next 5 minutes from the past hour.
2. **Metro Interstate Traffic Volume** (Das & MSTIM papers) — 48,204 hourly records with
   weather and holiday features (38,511 after cleaning). Task: predict the next hour from
   the past 24 hours.

## Repository contents

| File | Deliverable | Description |
|------|-------------|-------------|
| `D1_DATASET1.ipynb`, `D1_DATASET2.ipynb` | D1 | EDA, preprocessing, Linear Regression vs MLP. |
| `D2_DATASET1.ipynb`, `D2_DATASET2.ipynb` | D2 | Classical models + pruning/quantization/coreset on the MLP. |
| `D3_Dataset_1.ipynb`, `D3_Dataset_2.ipynb` | D3 | One-shot vs iterative pruning, 3 seeds, error bars. |
| `traffic-prediction-dataset.csv` | — | Huawei data. |
| `Metro_Interstate_Traffic_Volume.csv` | — | Metro Interstate data. |
| `D1_Report.pdf`, `D2_Report.pdf`, `D3_Report.pdf` | — | Reports for each stage. |
| `Literature_Survey.pdf` | D1 | Survey of the three reference papers. |

## Method (consistent across all deliverables)

- **MLP architecture (unchanged throughout):** Huawei 48-128-64-4; Metro
  456-128-64-32-16-1. ReLU activations, linear output.
- **Pipeline:** moving-average repair of sensor-failure zeros (Huawei); impossible-value
  and IQR-outlier removal + one-hot encoding (Metro); MinMax scaling fit on training data
  only; chronological train/test split (no leakage); sliding-window inputs.
- **Training:** Adam (and plain SGD for Huawei experiments), 20 epochs; seeds {1, 7, 13}
  for error bars.
- **Compression (D2):** L1 pruning (swept 5–80%), dynamic int8 quantization, and coreset
  (training-subset) selection — individually, in pairs, and all three.
- **Improvement (D3):** iterative pruning (gradual 20→40→60→80% with fine-tuning between
  steps) compared against one-shot pruning.
- **Metrics:** MAE, MAPE, RMSE, R².

## Key results

**D1 — classical vs MLP**

| Dataset | Model | R² |
|---|---|---|
| Huawei | Linear Regression | 0.946 |
| Huawei | MLP | 0.942–0.945 |
| Metro | MLP | 0.90 |

**D2 — compression (best findings)**

- Huawei: pruning to 80% kept R² at 0.947; int8 quantization gave a ~3× smaller model
  with no accuracy loss.
- Metro: compression *improved* accuracy (baseline 0.883 → prune+coreset 0.906) — the
  over-parameterized network benefits from pruning as regularization.
- Random Forest (3 seeds): R² 0.919 ± 0.001 — below the MLP and Linear Regression.

**D3 — comparison & improvement**

| Dataset | Our best model | Our R² | Study | Their result |
|---|---|---|---|---|
| Huawei | MLP + pruning | 0.947 | Navarro-Espinoza | R² ≈ 0.93 |
| Metro | MLP + prune + coreset | 0.90 | Das / MSTIM | deep models (LSTM/GRU, CNN+attn) |

Iterative pruning (D3) did **not** beat one-shot pruning on these small-to-medium
networks (Huawei 0.945 vs 0.947; Metro 0.88 vs 0.90), so one-shot is retained. The
analysis explains that iterative pruning helps mainly large, over-parameterized models.

## How to run

The notebooks are built for Google Colab.

1. Open a notebook in Colab and run the first cell to import dependencies (`numpy`,
   `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`).
2. Upload the matching CSV when prompted.
3. Run all cells top to bottom. Training prints every epoch.

A fixed set of random seeds is used for reproducibility.

## Reference papers

1. Navarro-Espinoza et al. (2022), *Technologies*, vol. 10(1):5 — Huawei dataset.
2. Das (2023), arXiv:2303.12643 — Metro Interstate dataset (LSTM/GRU).
3. MSTIM (2025), arXiv:2504.13576 — Metro Interstate dataset (CNN + LSTM + attention).

## Summary

A simple MLP predicts traffic flow about as accurately as classical models and as
published deep-learning studies, while being far cheaper to train and — through pruning,
quantization, and coreset selection — far smaller to store, with no meaningful loss of
accuracy.
