"""Dataset loading and the Haar wavelet input branch."""

import os

import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from . import config


class LearnableWaveletLayer(tf.keras.layers.Layer):
    """Single-level 2-D Haar DWT applied per RGB channel.

    Output: (batch, H/2, W/2, 12) = [LL, LH, HL, HH] for each of the 3 channels.
    The low-pass (LL) sub-band is scaled by `alpha` and the high-pass
    sub-bands (LH, HL, HH) by `beta`.

    Note: the transform runs inside `tf.py_function` in the input pipeline,
    so `alpha` and `beta` keep their initial value (1.0) during training.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.Variable(1.0, trainable=True, dtype=tf.float32)
        self.beta = tf.Variable(1.0, trainable=True, dtype=tf.float32)

    def _dwt_channel(self, channel):
        LL, (LH, HL, HH) = pywt.dwt2(channel, "haar")
        LL *= self.alpha
        LH *= self.beta
        HL *= self.beta
        HH *= self.beta
        return np.stack([LL, LH, HL, HH], axis=-1)

    def _dwt_batch(self, images):
        images = images.numpy() if hasattr(images, "numpy") else images
        batch = images.shape[0]
        channels = [
            np.stack([self._dwt_channel(images[j, :, :, c]) for j in range(batch)], axis=0)
            for c in range(3)
        ]
        return np.concatenate(channels, axis=-1).astype(np.float32)

    def call(self, inputs):
        out = tf.py_function(func=self._dwt_batch, inp=[inputs], Tout=tf.float32)
        out.set_shape([None, *config.WAVELET_SHAPE])
        return out


def prepare_dataset(ds, wavelet_layer, cache_path=None, shuffle=False, seed=config.SEED):
    ds = ds.map(lambda img, lbl: ((img, wavelet_layer(img)), lbl))
    if cache_path is not None:
        ds = ds.cache(cache_path)
    if shuffle:
        ds = ds.shuffle(100, seed=seed)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_datasets(train_dir, test_dir, batch_size=config.BATCH_SIZE,
                  validation_split=config.VALIDATION_SPLIT, seed=config.SEED, cache_dir=None):
    """Return (train, val, test, class_names). Each element is ((image, wavelet), label).

    `train_dir` is split into training/validation subsets; `test_dir` is held out.
    Both directories must contain one sub-folder per class.
    """
    common = dict(labels="inferred", batch_size=batch_size,
                  image_size=config.IMG_SIZE, color_mode="rgb")
    train = image_dataset_from_directory(train_dir, validation_split=validation_split,
                                         subset="training", seed=seed, shuffle=True, **common)
    val = image_dataset_from_directory(train_dir, validation_split=validation_split,
                                       subset="validation", seed=seed, shuffle=False, **common)
    test = image_dataset_from_directory(test_dir, shuffle=False, **common)
    class_names = train.class_names

    cache = (lambda name: None)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = (lambda name: os.path.join(cache_dir, name))

    wavelet_layer = LearnableWaveletLayer()
    train = prepare_dataset(train, wavelet_layer, cache("train"), shuffle=True, seed=seed)
    val = prepare_dataset(val, wavelet_layer, cache("val"))
    test = prepare_dataset(test, wavelet_layer, cache("test"))
    return train, val, test, class_names
