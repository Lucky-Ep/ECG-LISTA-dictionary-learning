from __future__ import annotations
import torch
import numpy as np
import pandas as pd
from data_handler import data_rand_slice
from dictionary_train import train_dictionary
from LISTA import use_lista
from ConvD_dictionary_train import train_sporco_cdl_1d, sporco_dict_to_torch
from ConvD_LISTA import conv_ista, use_conv_lista
# from k_fold_alpha_optimize import evaluate_alpha_kfold



def main() -> None:
    X = np.loadtxt("ecg_segments_100x40_6s.csv", delimiter=",", dtype=np.float32)
    n_ecg = 20
    n_slice = 40
    # print(X.shape)  # (4000, 1536)
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

    # LISTA with normal dictionary
    # W_d = train_dictionary(Xtr) #(64, 1536)
    # use_lista(Xtr, Xts, W_d)

    # LISTA with convolutional dictionary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, stats_list = train_sporco_cdl_1d(Xtr) #(16, 1, 32)
    D = sporco_dict_to_torch(D)
    D = D.to(device)
    Xtr_torch = torch.tensor(Xtr, dtype=torch.float32, device=device).unsqueeze(1)
    Xts_torch = torch.tensor(Xts, dtype=torch.float32, device=device).unsqueeze(1)
    # conv_ista(Xtr_torch, D)
    use_conv_lista(Xtr_torch, Xts_torch, D)
    

if __name__ == "__main__":
    main()
