"""Branch importance scores, Eq. (4): mean wavelet-guided gating score per branch
over the validation set, computed from a trained five-branch student.

Example:
    python branch_importance.py --train_dir data/train --test_dir data/test \
        --weights runs/student_5branch/student.weights.h5
"""

import argparse

import numpy as np
import tensorflow as tf

from src import config
from src.data import load_datasets
from src.models import build_student


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--weights", required=True, help="Weights of the five-branch student")
    p.add_argument("--top_k", type=int, default=config.STUDENT_TOP_K)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    _, val_ds, _, class_names = load_datasets(args.train_dir, args.test_dir, seed=args.seed)
    model = build_student(len(class_names), active=config.ALL_BRANCHES, k=args.top_k)
    model.load_weights(args.weights)

    gate = tf.keras.Model(model.inputs, model.get_layer("student_importance_scores").output)
    scores = np.concatenate([gate([img, wav], training=False).numpy() for (img, wav), _ in val_ds])
    importance = scores.mean(axis=0)

    print("Branch importance (lowest first = first to remove):")
    for i in np.argsort(importance):
        print(f"  {config.PAPER_NAMES[config.ALL_BRANCHES[i]]:6s} {importance[i]:.3f}")


if __name__ == "__main__":
    main()
