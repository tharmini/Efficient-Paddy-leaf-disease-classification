"""Knowledge distillation: frozen teacher -> compressed student.

The teacher is the Haar wavelet-guided dynamic FPN of Tharmini et al. (2026),
"Haar wavelet-guided dynamic feature pyramid network for an efficient paddy
leaf disease classification", Computers and Electronics in Agriculture, 250.
It is not part of this repository. Export your trained teacher as a TensorFlow
SavedModel (in Keras: `teacher.export("teacher_savedmodel")`) and pass that
folder with --teacher_model.

Example:
    python train_kd.py --train_dir data/train --test_dir data/test \
        --teacher_model teacher_savedmodel --out_dir runs/kd
"""

import argparse
import json
import os
import time

import tensorflow as tf

from src import config
from src.data import load_datasets
from src.distillation import Distiller
from src.models import build_student
from src.utils import SaveBestWeights, evaluate, get_complexity, save_history


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_dir", required=True, help="Training images, one sub-folder per class")
    p.add_argument("--test_dir", required=True, help="Held-out test images, one sub-folder per class")
    p.add_argument("--teacher_model", required=True,
                   help="Trained teacher exported as a TensorFlow SavedModel folder, see Tharmini et al. (2026)")
    p.add_argument("--out_dir", default="runs/kd")
    p.add_argument("--drop", nargs="*", default=[config.PAPER_NAMES[b] for b in config.DROP_BRANCHES],
                   help="Student branches to remove (paper names: P2 P3 P4 P4int P5)")
    p.add_argument("--top_k", type=int, default=config.STUDENT_TOP_K)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--kd_lambda", type=float, default=config.KD_LAMBDA)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--cache_dir", default=None, help="Optional tf.data cache folder (speeds up epochs)")
    return p.parse_args()


def load_teacher(path):
    """Wrap an exported teacher SavedModel as a frozen two-input Keras model."""
    img = tf.keras.Input(shape=(*config.IMG_SIZE, 3))
    wav = tf.keras.Input(shape=config.WAVELET_SHAPE)
    out = tf.keras.layers.TFSMLayer(path, call_endpoint="serve", trainable=False)([img, wav])
    teacher = tf.keras.Model([img, wav], out, name="teacher")
    teacher.trainable = False
    return teacher


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.train_dir, args.test_dir, args.batch_size, seed=args.seed, cache_dir=args.cache_dir)
    num_classes = len(class_names)

    # Teacher (frozen), loaded from its exported SavedModel
    teacher = load_teacher(args.teacher_model)
    teacher_cx = get_complexity(teacher)
    teacher_res = evaluate(teacher, test_ds, class_names)
    print(f"Teacher: test acc {teacher_res['accuracy'] * 100:.2f}% | "
          f"{teacher_cx['params_M']} M params | {teacher_cx['gflops']} GFLOPs")

    # Student (same seed as the non-KD baseline so initial weights match)
    tf.keras.utils.set_random_seed(args.seed)
    active = config.active_branches(config.to_code_names(args.drop))
    student = build_student(num_classes, active=active, k=args.top_k)
    student_cx = get_complexity(student)
    print(f"Student branches: {[config.PAPER_NAMES[b] for b in active]} | "
          f"{student_cx['params_M']} M params | {student_cx['gflops']} GFLOPs")

    # Train
    distiller = Distiller(student, teacher, temperature=args.temperature, kd_lambda=args.kd_lambda)
    distiller.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))
    best_path = os.path.join(args.out_dir, "student_kd.weights.h5")
    best_ckpt = SaveBestWeights(student, best_path, monitor="val_student_loss")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_student_loss", mode="min",
                                                  patience=args.patience, verbose=1)
    start = time.time()
    history = distiller.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                            callbacks=[best_ckpt, early_stop], verbose=1)
    minutes = (time.time() - start) / 60
    save_history(history, args.out_dir, loss_key="student_loss")

    # Evaluate the best-validation student
    student.load_weights(best_path)
    res = evaluate(student, test_ds, class_names, out_dir=args.out_dir,
                   title="KD student - test confusion matrix")

    summary = {
        "retained_branches": [config.PAPER_NAMES[b] for b in active],
        "removed_branches": args.drop,
        "hyperparameters": {"epochs": args.epochs, "patience": args.patience, "lr": args.lr,
                            "batch_size": args.batch_size, "temperature": args.temperature,
                            "kd_lambda": args.kd_lambda, "top_k": args.top_k, "seed": args.seed},
        "epochs_run": len(history.history["total_loss"]),
        "best_val_epoch": best_ckpt.best_epoch,
        "train_minutes": round(minutes, 2),
        "teacher": {**teacher_cx, "test_accuracy": teacher_res["accuracy"]},
        "student": {**student_cx, **{f"test_{k}": v for k, v in res.items()}},
    }
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nKD student: {res['correct']}/{res['total']} = {res['accuracy'] * 100:.2f}% "
          f"(macro F1 {res['macro_f1'] * 100:.2f}%) | best epoch {best_ckpt.best_epoch}")
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
