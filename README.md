# NASA C-MAPSS — Turbofan RUL Prediction
### 24-788 Introduction to Deep Learning, Spring 2026

This is my final mini-project. I trained three sequence models — LSTM, TCN, and PatchTST — to predict how many cycles a turbofan engine has left before failure, using the NASA C-MAPSS sensor dataset. Beyond just comparing them on the standard FD001 test set, I also tested how well each model holds up when evaluated on operating conditions it was never trained on (FD002–FD004).

---

## Getting the Data

Download from NASA's Prognostics Data Repository:
```
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
```
Extract and put these 12 files into `data/raw/`:
```
train_FD001.txt  test_FD001.txt  RUL_FD001.txt
train_FD002.txt  test_FD002.txt  RUL_FD002.txt
train_FD003.txt  test_FD003.txt  RUL_FD003.txt
train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

---

## Setup

Python 3.9+. Just run:
```bash
pip install -r requirements.txt
```

---

## Reproducing Results (without retraining)

All saved checkpoints are already in `checkpoints/`. To regenerate every figure and number from the report:

1. Put the 12 data files in `data/raw/`
2. Open `notebooks/06_reproduce_results.ipynb`
3. Run all cells

Takes under 2 minutes on CPU.

---

## Training From Scratch

If you want to retrain everything, run the notebooks in order:

```
01_eda.ipynb                     — data exploration, saves scaler + feature list
02_lstm_baseline.ipynb           — trains LSTM, saves lstm_best.pt
03_tcn.ipynb                     — trains TCN, saves tcn_best.pt
04_patchtst.ipynb                — trains PatchTST, saves patchtst_best.pt
05_cross_dataset_analysis.ipynb  — cross-dataset eval, no training needed
06_reproduce_results.ipynb       — regenerates all figures and tables
```

Each training notebook runs in under 5 minutes on CPU.

---

## Results

**FD001 test set:**

| Model    | RMSE (cycles) | PHM Score | Params  |
|----------|--------------|-----------|---------|
| LSTM     | 14.82        | 368.4     | 209,985 |
| PatchTST | 15.09        | 401.2     | 129,601 |
| TCN      | 16.93        | 458.7     | 92,801  |

**Cross-dataset generalization (trained on FD001, evaluated zero-shot):**

| Subset | LSTM  | TCN       | PatchTST |
|--------|-------|-----------|----------|
| FD001  | 14.82 | 16.93     | 15.09    |
| FD002  | 51.31 | collapsed | 61.20    |
| FD003  | 48.75 | 85.35     | 27.82    |
| FD004  | 53.44 | collapsed | 63.94    |

The TCN completely broke down on FD002 and FD004 — both have 6 operating conditions, and the shift in absolute sensor values pushed the model into numerical instability. LSTM and PatchTST handled it much more gracefully. PatchTST's result on FD003 (27.82) was the most surprising finding — it actually beat the LSTM by a significant margin there.

---

## AI Usage

I used Claude (Anthropic) as a coding assistant during this project — mainly for debugging PyTorch implementations, structuring the training loop, and working through the TCN hyperparameter tuning. The experimental design, all decisions about what to run and why, the analysis of results, and the written report are entirely my own work.

---

## References

- Bai et al., TCN — arxiv.org/abs/1803.01271
- Nie et al., PatchTST — arxiv.org/abs/2211.14730
- Zheng et al., LSTM on C-MAPSS — arxiv.org/abs/1709.01073
- Saxena et al., C-MAPSS dataset + PHM score — PHM Conference 2008
