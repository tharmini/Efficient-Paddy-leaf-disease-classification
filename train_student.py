"""Train the compressed student without distillation (cross-entropy only).

Used for the ablations in Tables III-V. Pass `--drop` with no names to train
the five-branch student needed by branch_importance.py.

Example:
    python train_student.py --train_dir data/train --test_dir data/test --drop P2 P4int
"""

import argparse
import json
import os

import tensorflow as tf

from src import config
from src.data import load_datasets
from src.models import build_student
from src.utils import SaveBestWeights, evaluate, get_complexity, save_history


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--out_dir", default="runs/student")
    p.add_argument("--drop", nargs="*", default=[config.PAPER_NAMES[b] for b in config.DROP_BRANCHES])
    p.add_argument("--top_k", type=int, default=config.STUDENT_TOP_K)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--cache_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.train_dir, args.test_dir, args.batch_size, seed=args.seed, cache_dir=args.cache_dir)

    active = config.active_branches(config.to_code_names(args.drop))
    model = build_student(len(class_names), active=active, k=args.top_k)
    cx = get_complexity(model)
    print(f"Student branches: {[config.PAPER_NAMES[b] for b in active]} | "
          f"{cx['params_M']} M params | {cx['gflops']} GFLOPs")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    best_path = os.path.join(args.out_dir, "student.weights.h5")
    best_ckpt = SaveBestWeights(model, best_path, monitor="val_loss")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, verbose=1)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=[best_ckpt, early_stop], verbose=1)
    save_history(history, args.out_dir, loss_key="loss")

    model.load_weights(best_path)
    res = evaluate(model, test_ds, class_names, out_dir=args.out_dir,
                   title="Student - test confusion matrix")
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump({"retained_branches": [config.PAPER_NAMES[b] for b in active],
                   "best_val_epoch": best_ckpt.best_epoch, **cx,
                   **{f"test_{k}": v for k, v in res.items()}}, f, indent=2)
    print(f"Student: {res['correct']}/{res['total']} = {res['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
