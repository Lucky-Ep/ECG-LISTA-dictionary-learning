from sklearn.decomposition import MiniBatchDictionaryLearning
import numpy as np
import pandas as pd

def train_dictionary(
    Xtr,
    num_atoms=64,
    batch_size=16,
    alpha=1.0,
    max_iter=500,
    random_state=42,
) -> np.ndarray:

    dict_model = MiniBatchDictionaryLearning(
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

    A_train = dict_model.fit_transform(Xtr)
    D = dict_model.components_

    return D
