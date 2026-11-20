"""Evaluation, complexity measurement, callbacks and plotting helpers."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


def get_complexity(model):
    """Parameter count and FLOPs for a single 1-image forward pass."""
    try:
        specs = [tf.TensorSpec([1, *inp.shape[1:]], inp.dtype) for inp in model.inputs]

        @tf.function
        def forward(img, wav):
            return model([img, wav], training=False)

        concrete = forward.get_concrete_function(*specs)
        _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
                        tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
                    .with_empty_output().build())
            prof = tf.compat.v1.profiler.profile(graph=graph, run_meta=tf.compat.v1.RunMetadata(),
                                                 cmd="op", options=opts)
        flops = int(prof.total_float_ops)
    except Exception as e:  # profiler is not available on every TF build
        print(f"FLOPs calculation failed: {e}")
        flops = None
    params = int(sum(np.prod(w.shape) for w in model.weights))
    return {"gflops": round(flops / 1e9, 4) if flops else None,
            "params": params, "params_M": round(params / 1e6, 4)}


def predict(model, dataset):
    """Return (y_true, y_pred) over a ((image, wavelet), label) dataset."""
    y_true, y_pred = [], []
    for (img, wav), labels in dataset:
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(model.predict([img, wav], verbose=0), axis=1))
    return np.array(y_true), np.array(y_pred)


def evaluate(model, dataset, class_names, out_dir=None, title="Test confusion matrix"):
    """Accuracy, per-class report and confusion matrix (optionally saved to out_dir)."""
    y_true, y_pred = predict(model, dataset)
    acc = float((y_true == y_pred).mean())
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    if out_dir:
        with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues",
                                                                    xticks_rotation=45)
        ax.set_title(title)
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return {"accuracy": acc, "correct": int((y_true == y_pred).sum()), "total": int(len(y_true)),
            "macro_f1": report["macro avg"]["f1-score"]}


class SaveBestWeights(tf.keras.callbacks.Callback):
    """Save `model` weights whenever the monitored validation loss improves."""

    def __init__(self, model_to_save, filepath, monitor):
        super().__init__()
        self.model_to_save, self.filepath, self.monitor = model_to_save, filepath, monitor
        self.best, self.best_epoch = np.inf, None

    def on_epoch_end(self, epoch, logs=None):
        cur = (logs or {}).get(self.monitor)
        if cur is not None and cur < self.best:
            self.best, self.best_epoch = cur, epoch + 1
            self.model_to_save.save_weights(self.filepath)
            print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {cur:.5f}; weights saved.")


def save_history(history, out_dir, loss_key):
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["accuracy"], label="Train")
    axes[0].plot(hist["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[1].plot(hist[loss_key], label="Train")
    axes[1].plot(hist[f"val_{loss_key}"], label="Validation")
    axes[1].set_title("Cross-entropy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=150)
    plt.close(fig)
