from __future__ import annotations
import numpy as np
import pandas as pd
from data_handler import data_rand_slice
from dictionary_training import train_dictionary
from LISTA import use_lista
# from k_fold_alpha_optimize import evaluate_alpha_kfold



def main() -> None:
    X = np.loadtxt("ecg_segments_100x4_60s.csv", delimiter=",", dtype=np.float32)
    n_ecg = 100
    n_slice = 4
    # print(X.shape)  # (400, 15360)
    slice_train_idx, slice_test_idx = data_rand_slice(X, n_ecg, n_slice)
    Xtr = X[slice_train_idx]
    Xts = X[slice_test_idx]

    # alpha_list = [1.0, 3.0, 10.0, 30.0]
    # kf_result = evaluate_alpha_kfold(Xtr, alpha_list)
    # print(kf_result)
    # best_row = kf_result.loc[kf_result["val_mse_mean"].idxmin()]
    # best_alpha = best_row["alpha"]
    # print("Best alpha:", best_alpha)
    # -----------------------------
    #    alpha  val_mse_mean  val_mse_std  avg_nonzero_mean  sparsity_mean
    # 0    1.0      0.041046     0.000710         17.905043       0.720234
    # 1    3.0      0.042707     0.000772          2.415829       0.962253
    # 2   10.0      0.049045     0.000887          1.005995       0.984281
    # 3   30.0      0.079097     0.000973          0.340681       0.994677
    # Best alpha: 1.0
    # -----------------------------

    W_d = train_dictionary(Xtr)
    use_lista(Xtr, Xts, W_d)

if __name__ == "__main__":
    main()
