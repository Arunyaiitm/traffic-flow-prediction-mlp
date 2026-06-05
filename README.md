# Traffic Flow Prediction using MLP
### Foundations of Machine Learning — Deliverable 1

**Authors:** Arunya (ZDA24B031) & Srijan Reddy Sankepally (ZDA24B007)

---

## Overview

This project predicts short-term traffic flow and compares a classical machine-learning
model (Linear Regression) against a Multilayer Perceptron (MLP). We use two publicly
available datasets, each drawn from a research paper, and adapt the deep-model setups in
those papers to a feed-forward MLP.

## Repository contents

| File | Description |
|------|-------------|
| `D1_DATASET1.ipynb` | Huawei dataset — EDA, preprocessing, Linear Regression vs MLP (MLP trained with SGD and mini-batch gradient descent). |
| `D1_DATASET2.ipynb` | Metro Interstate dataset — EDA, cleaning, and a 4-layer MLP (Adam). |
| `traffic-prediction-dataset.csv` | Huawei 6-cross traffic data (5-minute readings). |
| `Metro_Interstate_Traffic_Volume.csv` | Metro Interstate hourly traffic data with weather/holiday features. |
| `Literature_Survey.pdf` | Summary of the three reference papers. |
| `D1_Report.pdf` | Full report — dataset analysis, EDA, methodology, and results. |

## Datasets

1. **Huawei dataset** (Navarro-Espinoza paper) — 16,128 rows, 6 sensors, one reading every
   5 minutes for 56 days. Task: predict the next 5 minutes from the past 1 hour, using
   4 crosses as 4 lanes of one intersection.
2. **Metro Interstate Traffic Volume** (Das & MSTIM papers) — 48,204 rows, hourly traffic
   on I-94 (2012–2018) with weather and holiday features. Task: predict the next hour from
   the past 24 hours.

## How to run

The notebooks are designed for Google Colab.

1. Open a notebook in Google Colab.
2. Run the first cell to import dependencies (`numpy`, `pandas`, `matplotlib`, `seaborn`,
   `scikit-learn`, `torch`).
3. When prompted, upload the corresponding CSV file
   (`traffic-prediction-dataset.csv` for Notebook 1,
   `Metro_Interstate_Traffic_Volume.csv` for Notebook 2).
4. Run the remaining cells top to bottom.

A fixed random seed (`RANDOM_STATE = 0`) is set for reproducibility.

## Methods

- **Preprocessing:** moving-average handling of sensor-failure zeros (Huawei); removal of
  impossible values and IQR outliers, plus categorical encoding (Metro). MinMax scaling fit
  on the training split only; chronological train/test split to avoid data leakage.
- **Windowing:** past readings are flattened into one input row to predict the next value
  (Huawei: 12 readings × 4 lanes = 48 inputs).
- **Models:** Linear Regression (classical baseline) and an MLP with ≥2 hidden layers
  (PyTorch). Metro MLP uses 4 hidden layers (128, 64, 32, 16), batch size 64, Adam optimizer,
  adapted from the Das paper.
- **Metrics:** MAE, RMSE, MAPE, R².

## Results

| Dataset | Model | R² | MAE | Training time |
|---|---|---|---|---|
| Huawei | Linear Regression | 0.946 | 10.7 | ~0.15 s |
| Huawei | MLP (SGD, batch = 1) | 0.942 | 11.2 | ~159 s |
| Huawei | MLP (mini-batch = 128) | 0.803 | 21.5 | ~1.8 s |
| Metro Interstate | MLP (4-layer, Adam) | 0.90 | 425 | ~28 s |

R² is comparable across datasets; MAE is not (different target scales — vehicles per 5 min
for Huawei vs per hour for Metro Interstate).

## Reference papers

1. Navarro-Espinoza et al. (2022), *Technologies* — Huawei intersection dataset.
2. Das (2023), arXiv — Metro Interstate dataset (LSTM/GRU).
3. MSTIM (2025), arXiv — Metro Interstate dataset (CNN + LSTM + attention).

## Future work

- Add a non-linear classical model (Random Forest) for a stronger comparison.
- Run experiments varying history length (6/12/24 h) and feature sets.
- Apply model compression techniques (pruning, quantization) to study the size/speed/accuracy trade-off.
