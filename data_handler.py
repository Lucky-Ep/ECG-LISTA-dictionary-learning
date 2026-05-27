from __future__ import annotations
import numpy as np

def data_rand_slice(X: np.ndarray, n_ecg: int, n_slice: int) -> tuple[np.ndarray, np.ndarray]:
    n_data = X.shape[0]
    len_data = X.shape[1]
    ecg_idx = np.linspace(0, n_data, num=n_ecg, endpoint=False).astype(int)
    np.random.seed(42)
    np.random.shuffle(ecg_idx)

    ecg_train_idx = ecg_idx[0:round(0.8*n_ecg)]
    slice_train_idx = np.concatenate([
    ecg_train_idx,
    ecg_train_idx + 1,
    ecg_train_idx + 2,
    ecg_train_idx + 3
    ])
    np.random.shuffle(slice_train_idx)

    ecg_test_idx = ecg_idx[round(0.8*n_ecg):n_ecg]
    slice_test_idx = np.concatenate([
    ecg_test_idx,
    ecg_test_idx + 1,
    ecg_test_idx + 2,
    ecg_test_idx + 3
    ])
    np.random.shuffle(slice_test_idx)

    return slice_train_idx, slice_test_idx
    