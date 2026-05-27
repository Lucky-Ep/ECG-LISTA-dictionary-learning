# ECG LISTA Sparse Coding Experiment

This project is an experimental pipeline for applying dictionary learning and LISTA (Learned ISTA) to synthetic ECG signals.

The current goal is to compare the convergence efficiency of traditional ISTA and a learned LISTA model. The project first learns a sparse dictionary from ECG signal segments, then uses ISTA-generated sparse codes as training targets for LISTA. Finally, it plots the MSE comparison between ISTA and LISTA under different iteration/layer counts.

> Current status: this is a research prototype. The project currently focuses on LISTA convergence comparison. A complete denoised ECG export or application-stage output has not been implemented yet and may be added after further feedback.

## Project Summary

The pipeline follows these steps:

1. Load ECG segment data from `ecg_segments_100x4_60s.csv` (Due to the large size of the file, it cannot be uploaded at present. Users are requested to generate new training data by themselves.)
2. Split the dataset by original ECG source to reduce data leakage between training and testing sets.
3. Train a sparse dictionary using `MiniBatchDictionaryLearning`.
4. Generate sparse-code targets with ISTA.
5. Train a LISTA model to approximate ISTA solutions more efficiently.
6. Compare ISTA and LISTA using MSE over different iteration/layer counts.
7. Display a log-scale MSE plot for convergence comparison.

## Repository Structure

```text
.
├── starter.py
├── data_handler.py
├── dictionary_training.py
├── k_fold_alpha_optimize.py
├── LISTA.py
├── ecg_segments_100x4_60s.csv
└── README.md
```

## File Descriptions

### `starter.py`

Main entry point of the project.

It loads the ECG segment dataset, splits the data into training and testing sets, trains the dictionary, and then calls the LISTA/ISTA comparison pipeline.

Expected input file:

```text
ecg_segments_100x4_60s.csv
```

The current dataset shape is expected to be:

```text
(400, 15360)
```

This corresponds to 100 generated ECG signals, with 4 segments extracted from each signal.

### `data_handler.py`

Handles train/test splitting.

The project uses 100 original ECG signals, each split into 4 segments. To reduce data leakage, the split is performed at the original ECG-signal level rather than randomly mixing all 400 segments directly.

This means that segments from the same generated ECG signal should stay within either the training set or the testing set.

### `dictionary_training.py`

Trains a sparse dictionary using scikit-learn's `MiniBatchDictionaryLearning`.

Default parameters include:

```text
num_atoms = 64
batch_size = 16
alpha = 1.0
max_iter = 500
random_state = 42
```

The trained dictionary is returned and then passed into the LISTA pipeline.

### `k_fold_alpha_optimize.py`

Optional helper script for selecting the dictionary learning regularization parameter `alpha`.

It evaluates different alpha values using K-fold validation and reports:

- validation MSE
- average number of nonzero sparse coefficients
- sparsity ratio

This part is optional and is currently commented out in `starter.py`.

### `LISTA.py`

Contains the ISTA baseline, dataset construction, LISTA model definition, training loop, and comparison plot.

The comparison tests different iteration/layer counts:

```text
T = 1, 3, 5, 7, 9, 11
```

For each value of `T`:

- ISTA is applied for `T` iterations.
- LISTA is trained with `T` layers.
- The final MSE values are plotted on a log-scale graph.

The LISTA model uses learnable parameters initialized from the ISTA update structure:

- encoder matrix `W_e`
- recurrent matrix `S`
- threshold parameter `theta`

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install numpy pandas scikit-learn torch matplotlib
```

Depending on your Python and PyTorch setup, you may prefer to install PyTorch from the official installation page:

```text
https://pytorch.org/get-started/locally/
```

## How to Run

Place the ECG dataset file in the project root directory:

```text
ecg_segments_100x4_60s.csv
```

Then run:

```bash
python starter.py
```

The script will:

1. load the ECG dataset,
2. split training and testing data,
3. train the dictionary,
4. generate ISTA targets,
5. train LISTA models,
6. compare LISTA and ISTA MSE,
7. display the convergence plot.

## Data Generation

The ECG data used in this project was generated from synthetic ECG signals. The data generation part was based on external MATLAB code rather than original code written in this repository.

If the external MATLAB code is included in this repository, make sure its original copyright notice, license file, and author information are preserved.

If the external MATLAB code does not have a clear open-source license, the safer choice is **not to upload the MATLAB source code directly**. Instead, link to the original source and cite the related paper or software package in this README.

A suitable attribution section is included below.

## External Code / Data Attribution

Synthetic ECG data generation was based on ECGSYN, a realistic ECG waveform generator.

ECGSYN source:

```text
https://physionet.org/content/ecgsyn/
```

MATLAB implementation:

```text
https://physionet.org/content/ecgsyn/1.0.0/Matlab/
```

Please cite the original ECGSYN publication when using this data generation method:

```text
P. E. McSharry, G. D. Clifford, L. Tarassenko, and L. A. Smith,
"A dynamical model for generating synthetic electrocardiogram signals,"
IEEE Transactions on Biomedical Engineering, vol. 50, no. 3, pp. 289-294, 2003.
DOI: 10.1109/TBME.2003.808805
```

## Important Note on Licensing

This repository contains my own Python implementation for dictionary learning and LISTA experiments.

The ECG data generation code is based on external MATLAB code. Before uploading external source code to GitHub, check its license carefully.

Recommended options:

1. If the external code has a license that allows redistribution, include the original license and attribution.
2. If the external code does not clearly allow redistribution, do not upload the code directly.
3. In that case, provide a link to the original source and explain that users should obtain the data generation code from the original project.

## Current Limitations

- The project currently compares LISTA and ISTA convergence efficiency.
- A complete denoising output pipeline has not been implemented yet.
- The current LISTA training target is generated by ISTA sparse codes.
- The output is mainly an MSE comparison plot rather than a finalized cleaned ECG signal.
- Further changes may be made after receiving feedback on the experimental direction.

## Possible Future Work

- Add a complete ECG denoising output pipeline.
- Export reconstructed or denoised ECG signals.
- Save comparison plots automatically.
- Add command line arguments for dataset path, dictionary size, alpha, and training epochs.
- Compare LISTA results against additional sparse coding or denoising baselines.
- Add more detailed evaluation metrics beyond MSE.

## Disclaimer

This project is for educational and research experimentation only. It is not a medical diagnostic tool and should not be used for clinical decision-making.
