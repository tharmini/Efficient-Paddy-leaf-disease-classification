"""Compressed student architecture.

The model takes two inputs: the RGB image (224x224x3) and its Haar wavelet
decomposition (112x112x12). The wavelet branch produces a guidance vector
that scores the multiscale branches; the top-k branches are fused with
importance weights before classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from . import config


def build_wavelet_cnn(input_shape=config.WAVELET_SHAPE):
    """Lightweight CNN applied to the wavelet sub-bands."""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters in (24, 48, 96):
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
    return models.Model(inputs, x)


def conv_block(x, filters, kernel_size=3, block_name="conv_block"):
    """CB: separable convolution -> batch norm -> ReLU."""
    x = layers.SeparableConv2D(filters, kernel_size, padding="same", name=f"{block_name}_conv")(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn")(x)
    return layers.ReLU(name=f"{block_name}_relu")(x)


def _gated_head(branches, wavelet_inputs, channels, k, num_classes, score_name=None):
    """Wavelet-guided top-k branch selection, importance-weighted fusion and classifier."""
    wavelet_features = build_wavelet_cnn()(wavelet_inputs)
    gate_features = layers.Conv2D(channels, (1, 1), padding="same")(wavelet_features)

    pooled = [layers.GlobalAveragePooling2D()(b) for b in branches]
    stacked = layers.Lambda(lambda x: tf.stack(x, axis=1))(pooled)

    g = layers.GlobalAveragePooling2D()(gate_features)                    # guidance vector g
    scores = layers.Dense(len(branches), activation="softmax", name=score_name)(g)  # Eq. (2)

    top_scores, top_idx = layers.Lambda(lambda s: tf.nn.top_k(s, k=k))(scores)
    top_branches = layers.Lambda(
        lambda t: tf.gather(t[0], t[1], batch_dims=1), output_shape=(k, channels)
    )([stacked, top_idx])
    fused = layers.Lambda(
        lambda t: tf.reduce_sum(t[0] * tf.expand_dims(t[1], axis=-1), axis=1)
    )([top_branches, top_scores])

    x = layers.Dense(256, activation="relu")(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    return layers.Dense(num_classes, activation="softmax")(x)


def build_student(num_classes, active=None, input_shape=(*config.IMG_SIZE, 3),
                  wavelet_shape=config.WAVELET_SHAPE, k=config.STUDENT_TOP_K,
                  backbone_weights="imagenet"):
    """Compressed student.

    1) Backbone truncation: EfficientNetB0 features up to C5 (block4c_add).
    2) FPN channel reduction: 256 channels.
    3) Branch removal: only branches listed in `active` are built
       (default: P5, P4, P3 -- P2 and P4int removed).
    """
    active = list(active) if active is not None else config.active_branches()
    k = min(k, len(active))
    ch = config.STUDENT_CHANNELS

    base = EfficientNetB0(include_top=False, input_shape=input_shape, weights=backbone_weights)
    for layer in base.layers:
        layer.trainable = layer.name.startswith("block4")

    C2 = base.get_layer("block1a_project_bn").output
    C3 = base.get_layer("block2b_add").output
    C4 = base.get_layer("block3b_add").output
    C5 = base.get_layer("block4c_add").output

    branch = {}

    p5 = layers.Conv2D(ch, (1, 1), padding="same")(C5)
    P5_con = conv_block(p5, ch, 3, "s_blockfun")
    if "P5_con" in active:
        branch["P5_con"] = P5_con
    P5_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(P5_con)

    C4_1 = layers.Conv2D(ch, (1, 1), padding="same")(C4)
    if "P4_a" in active:
        branch["P4_a"] = conv_block(C4_1, ch, 3, "s_blockfhun")     # P4int
    P4_add = layers.Add()([P5_up, C4_1])
    if "P4_con" in active:
        branch["P4_con"] = conv_block(P4_add, ch, 3, "s_blockgn")

    if "P3_con" in active or "P2_con" in active:
        P4_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(P4_add)
        C3_1 = layers.Conv2D(ch, (1, 1), padding="same")(C3)
        P3_add = layers.Add()([P4_up, C3_1])
        if "P3_con" in active:
            branch["P3_con"] = conv_block(P3_add, ch, 3, "s_blockhn")

    if "P2_con" in active:
        P3_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(P3_add)
        C2_1 = layers.Conv2D(ch, (1, 1), padding="same")(C2)
        P2_add = layers.Add()([P3_up, C2_1])
        branch["P2_con"] = conv_block(P2_add, ch, 3, "s_blockp2")

    wavelet_inputs = layers.Input(shape=wavelet_shape)
    branches = [branch[b] for b in active]
    outputs = _gated_head(branches, wavelet_inputs, ch, k, num_classes,
                          score_name="student_importance_scores")
    return models.Model([base.input, wavelet_inputs], outputs, name="student")
