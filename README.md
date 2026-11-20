# Efficient Paddy Leaf Disease Classification through Progressive Multiscale Network Compression

Official TensorFlow/Keras implementation of the compressed student model from the paper by
Kiriharan Tharmini, Thanikasalam Kokul, Amirthalingam Ramanan and Subha Fernando.

The student is obtained from a wavelet-guided EfficientNetB0 + FPN teacher in three compression stages, then trained with knowledge distillation (KD):

1. **Backbone truncation**: keep EfficientNetB0 stages C2–C5.
2. **FPN channel reduction**: 256 channels.
3. **Importance-guided branch removal**: the Haar-wavelet gating scores, averaged over the validation set (Eq. 4), rank the branches. The two lowest-ranked branches, **P2** and **P4int**, are removed, leaving P3, P4 and P5.
4. **Knowledge distillation** from the frozen teacher (Eq. 6).

The final student has **0.74 M parameters** and needs **1.17 GFLOPs**.

## Teacher network

The teacher is the Haar wavelet-guided dynamic Feature Pyramid Network (Haar-FPN) from Tharmini et al. (2026). Its architecture and training are described in that paper, and its code and weights are not included in this repository:

> K. Tharmini, T. Kokul, A. Ramanan, T. Suvanthini and S. Fernando, "Haar wavelet-guided dynamic feature pyramid network for an efficient paddy leaf disease classification," *Computers and Electronics in Agriculture*, vol. 250, 2026.

`train_kd.py` takes the trained teacher as a TensorFlow SavedModel folder. To create one from a Keras model, run `teacher.export("teacher_savedmodel")`.

## Repository layout

```
├── train_kd.py            # KD: teacher -> compressed student (main result)
├── train_student.py       # student without KD (ablations, five-branch student)
├── branch_importance.py   # branch importance scores, Eq. (4)
├── evaluate.py            # evaluate a trained student on a test folder
├── requirements.txt
└── src/
    ├── config.py          # hyper-parameters used in the paper
    ├── data.py            # dataset loading + Haar wavelet input
    ├── models.py          # student architecture
    ├── distillation.py    # KD loss and training loop
    └── utils.py           # metrics, FLOPs/params, callbacks, plots
```

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Data

Put each dataset in this layout, with one sub-folder per class:

```
data/
├── train/   # split 80/20 into training/validation by the scripts
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

Datasets used in the paper:

| Dataset | Images | Classes | Link |
|---|---|---|---|
| NP-LankaPaddy | 2,002 | 4 | https://doi.org/10.5281/zenodo.18334503 |
| UCI Rice Leaf Diseases | 120 | 3 | https://doi.org/10.24432/C5R013 |
| RiceLeafsv3 | 2,710 | 6 | https://doi.org/10.34740/KAGGLE/DSV/3726583 |
| Paddy Doctor | 16,225 | 13 | Petchiammal et al., CODS-COMAD 2023 |

For NP-LankaPaddy, use a 70/30 train/test split. 20% of the training set is held out for validation.

## Usage

**1. Knowledge distillation (final model)**

```bash
python train_kd.py \
    --train_dir data/train --test_dir data/test \
    --teacher_model teacher_savedmodel \
    --out_dir runs/kd
```

**2. Student without KD**

```bash
python train_student.py --train_dir data/train --test_dir data/test --out_dir runs/student
```

**3. Branch importance (Eq. 4)**: first train the five-branch student, then score its branches:

```bash
python train_student.py --train_dir data/train --test_dir data/test --drop --out_dir runs/student_5branch
python branch_importance.py --train_dir data/train --test_dir data/test \
    --weights runs/student_5branch/student.weights.h5
```

**4. Evaluation**

```bash
python evaluate.py --test_dir data/test --weights runs/kd/student_kd.weights.h5
```

Each run writes `result.json`, `history.json`, `curves.png`, `confusion_matrix.png` and `classification_report.json` to `--out_dir`.

## Default hyper-parameters

| Setting | Value |
|---|---|
| Input size | 224 × 224 |
| Optimiser | Adam |
| Learning rate | 1e-4 (fixed) |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 7 (validation cross-entropy) |
| Validation split | 20% of training set |
| KD temperature τ | 4 |
| KD weight λ | 0.5 |
| Student backbone / FPN channels | C2–C5 / 256 |
| Student branches | P3, P4, P5 (P2, P4int removed) |
| Student top-k gating | 2 |
| Seed | 42 |

You can override any of these from the command line (`--lr`, `--epochs`, `--patience`, `--temperature`, `--kd_lambda`, `--drop`, `--top_k`, ...). Run any script with `-h` to list the options.

## Results

| Dataset | Params (M) | GFLOPs | Accuracy (%) |
|---|---|---|---|
| UCI Rice Leaf Diseases | 0.74 | 1.17 | 97.22 |
| RiceLeafsv3 | 0.74 | 1.17 | 96.86 |
| Paddy Doctor | 0.74 | 1.17 | 96.70 |
| NP-LankaPaddy | 0.74 | 1.17 | 94.35 |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{tharmini2026efficient,
  title     = {Efficient Paddy Leaf Disease Classification through Progressive Multiscale Network Compression},
  author    = {Tharmini, Kiriharan and Kokul, Thanikasalam and Ramanan, Amirthalingam and Fernando, Subha},
  booktitle = {TODO},
  year      = {2026}
}

@article{tharmini2026haar,
  title   = {Haar wavelet-guided dynamic feature pyramid network for an efficient paddy leaf disease classification},
  author  = {Tharmini, Kiriharan and Kokul, Thanikasalam and Ramanan, Amirthalingam and Suvanthini, T. and Fernando, Subha},
  journal = {Computers and Electronics in Agriculture},
  volume  = {250},
  year    = {2026}
}
```
