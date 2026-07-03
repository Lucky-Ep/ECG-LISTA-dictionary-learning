import numpy as np
import torch
import torch.nn.functional as F
from sporco.dictlrn import cbpdndl


def train_sporco_cdl_1d(
    X_train_np,
    num_atoms=16,
    filter_len=32,
    lmbda=0.1,
    max_main_iter=5,
    seed=42,
):
    """
    Train a 1D convolutional dictionary using SPORCO.

    Parameters
    ----------
    X_train_np : np.ndarray
        Training signals, shape (num_segments, signal_length)

    num_atoms : int
        Number of convolutional dictionary atoms.

    filter_len : int
        Length of each 1D atom.

    lmbda : float
        Sparse regularization parameter used by SPORCO.

    max_main_iter : int
        Number of dictionary learning iterations.

    Returns
    -------
    D_sporco : np.ndarray
        Learned dictionary, expected shape approximately (filter_len, num_atoms)

    stats : list
        SPORCO training iteration statistics.
    """
    np.random.seed(seed)

    # SPORCO 1D convention:
    # s: (signal_length, num_training_signals) (4000, 1536)
    s = X_train_np.T.astype(np.float32)
    s = np.ascontiguousarray(s)

    # Optional but strongly recommended:
    # remove DC and normalize each training signal
    s = s - np.mean(s, axis=0, keepdims=True)
    S_std = np.std(s, axis=0, keepdims=True) + 1e-8
    s = s / S_std

    # Initial dictionary:
    # D0: (filter_len, num_atoms)
    D0 = np.random.randn(filter_len, num_atoms).astype(np.float32)

    # Normalize initial atoms
    D0 = D0 / (np.linalg.norm(D0, axis=0, keepdims=True) + 1e-8)

    opt = cbpdndl.ConvBPDNDictLearn.Options({
        'Verbose': True,
        'MaxMainIter': max_main_iter,

        'CBPDN': {
            'MaxMainIter': 20,
            'RelStopTol': 1e-3,
        },

        'CCMOD': {
            'MaxMainIter': 20,
            'RelStopTol': 1e-3,
            'ZeroMean': True,
        }
    })

    # dimN=1 means 1D signal.
    # dimK=1 means one axis is used for independent training signals.
    learner = cbpdndl.ConvBPDNDictLearn(
        D0,
        s,
        lmbda,
        opt,
        xmethod='admm',
        dmethod='pgm',
        dimN=1,
        dimK=1,
    )

    D_sporco = learner.solve()
    D_sporco = learner.getdict()

    return D_sporco, learner.itstat


def sporco_dict_to_torch(D_sporco, device="cpu"):
    """
    D_sporco: (K, 1, 1, M)
    return D_torch: (M, 1, K)
    """
    import numpy as np
    import torch

    D_np = np.asarray(D_sporco).astype(np.float32)

    # (K, 1, 1, M) -> (K, M)
    D_np = D_np[:, 0, 0, :]

    # (K, M) -> (M, K)
    D_np = D_np.T

    # (M, K) -> (M, 1, K)
    D_torch = torch.tensor(D_np, dtype=torch.float32, device=device).unsqueeze(1)

    D_torch = D_torch / (
        torch.linalg.vector_norm(D_torch, dim=2, keepdim=True) + 1e-12
    )

    return D_torch