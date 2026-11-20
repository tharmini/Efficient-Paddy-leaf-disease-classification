"""Evaluate a trained student on a test folder.

Example:
    python evaluate.py --test_dir data/test --weights runs/kd/student_kd.weights.h5
"""

import argparse
import os

import tensorflow as tf

from src import config
from src.data import LearnableWaveletLayer, prepare_dataset
from src.models import build_student
from src.utils import evaluate, get_complexity


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--drop", nargs="*", default=[config.PAPER_NAMES[b] for b in config.DROP_BRANCHES])
    p.add_argument("--top_k", type=int, default=config.STUDENT_TOP_K)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    test = tf.keras.preprocessing.image_dataset_from_directory(
        args.test_dir, shuffle=False, labels="inferred", batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE, color_mode="rgb")
    class_names = test.class_names
    test = prepare_dataset(test, LearnableWaveletLayer())

    active = config.active_branches(config.to_code_names(args.drop))
    model = build_student(len(class_names), active=active, k=args.top_k)
    model.load_weights(args.weights)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    cx = get_complexity(model)
    res = evaluate(model, test, class_names, out_dir=args.out_dir)
    print(f"Accuracy {res['accuracy'] * 100:.2f}% ({res['correct']}/{res['total']}) | "
          f"macro F1 {res['macro_f1'] * 100:.2f}% | {cx['params_M']} M params | {cx['gflops']} GFLOPs")


if __name__ == "__main__":
    main()
