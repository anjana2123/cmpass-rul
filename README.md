# NASA C-MAPSS — Turbofan RUL Prediction

Mini-project for 24-788 Deep Learning. We train three models — LSTM, TCN, and PatchTST — to predict remaining useful life (RUL) of turbofan engines from sensor readings, then test how well each one generalizes across operating conditions.

---

## Setup

Python 3.9+. Install dependencies:

```bash
pip install -r requirements.txt
```

Get the data from the NASA Prognostics Repository:
```
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
```

Extract and drop these 12 files into `data/raw/`:
```
train_FD001.txt  test_FD001.txt  RUL_FD001.txt
train_FD002.txt  test_FD002.txt  RUL_FD002.txt
train_FD003.txt  test_FD003.txt  RUL_FD003.txt
train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

---

## Reproducing Results

To regenerate all figures and numbers from the report without retraining:

1. Make sure `data/raw/` has the 12 data files
2. Make sure `checkpoints/` has the saved `.pt` and `.pkl` files
3. Run all cells in `notebooks/06_reproduce_results.ipynb`

Runs in under 2 minutes on CPU.

---

## Training From Scratch

Run notebooks in this order:

```
01_eda.ipynb                    → saves scaler and feature list
02_lstm_baseline.ipynb          → saves lstm_best.pt
03_tcn.ipynb                    → saves tcn_best.pt
04_patchtst.ipynb               → saves patchtst_best.pt
05_cross_dataset_analysis.ipynb → loads all three, no training needed
```

Each training notebook takes under 5 minutes on CPU.

---

## Results

FD001 test set:

| Model    | RMSE  | PHM Score | Params  |
|----------|-------|-----------|---------|
| LSTM     | 14.82 | 368.4     | 209,985 |
| PatchTST | 15.09 | 401.2     | 129,601 |
| TCN      | 16.93 | 458.7     | 92,801  |

Cross-dataset generalization (all models trained on FD001 only):

| Subset | LSTM  | TCN       | PatchTST |
|--------|-------|-----------|----------|
| FD001  | 14.82 | 16.93     | 15.09    |
| FD002  | 51.31 | collapsed | 61.20    |
| FD003  | 48.75 | 85.35     | 27.82    |
| FD004  | 53.44 | collapsed | 63.94    |

TCN collapsed on FD002 and FD004 (6 operating conditions each). LSTM and PatchTST degraded gracefully.

---

## References

- Bai et al., TCN — arxiv.org/abs/1803.01271
- Nie et al., PatchTST — arxiv.org/abs/2211.14730
- Zheng et al., LSTM on C-MAPSS — arxiv.org/abs/1709.01073
- PHM scoring — doi.org/10.1016/j.ress.2017.11.021