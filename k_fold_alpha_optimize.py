import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import MiniBatchDictionaryLearning


def evaluate_alpha_kfold(
    X_train_all,
    alpha_list,
    num_atoms=64,
    batch_size=16,
    max_iter=20,
    n_splits=3,
    random_state=42,
):
    results = []

    kf = KFold(n_splits=n_splits)

    for alpha in alpha_list:
        fold_mse_list = []
        fold_nonzero_list = []
        fold_sparsity_list = []
        print("Evaluating ")
        for fold, (fit_idx, val_idx) in enumerate(kf.split(X_train_all)):
            X_fit = X_train_all[fit_idx]
            X_val = X_train_all[val_idx]

            model = MiniBatchDictionaryLearning(
                n_components=num_atoms,
                alpha=alpha,
                batch_size=batch_size,
                max_iter=max_iter,
                fit_algorithm="cd",
                transform_algorithm="lasso_cd",
                transform_alpha=alpha,
                random_state=random_state,
                shuffle=True,
                n_jobs=-1,
            )
            print(f"alpha={alpha}, fold={fold}: start fit", flush=True)

            model.fit(X_fit)

            print(f"alpha={alpha}, fold={fold}: fit done", flush=True)

            D = model.components_

            print(f"alpha={alpha}, fold={fold}: transform val", flush=True)

            A_val = model.transform(X_val)

            print(f"alpha={alpha}, fold={fold}: transform val done", flush=True)

            X_val_recon = A_val @ D
            print(f"alpha={alpha}, fold={fold}: transform done", flush=True)
            
            val_mse = np.mean((X_val - X_val_recon) ** 2)
            avg_nonzero = np.mean(np.sum(np.abs(A_val) > 1e-6, axis=1))
            sparsity = np.mean(np.abs(A_val) < 1e-6)

            fold_mse_list.append(val_mse)
            fold_nonzero_list.append(avg_nonzero)
            fold_sparsity_list.append(sparsity)

        results.append({
            "alpha": alpha,
            "val_mse_mean": np.mean(fold_mse_list),
            "val_mse_std": np.std(fold_mse_list),
            "avg_nonzero_mean": np.mean(fold_nonzero_list),
            "sparsity_mean": np.mean(fold_sparsity_list),
        })

    results_df = pd.DataFrame(results)
    return results_df