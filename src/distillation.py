"""Response-based knowledge distillation (Section III-C, Eq. (5)-(6))."""

import tensorflow as tf

from . import config


def soften(probs, temperature):
    """Temperature-scaled distribution from softmax probabilities.

    The models output probabilities, so logits are recovered as log(p)
    (up to a constant, which softmax ignores) before dividing by tau.
    """
    logits = tf.math.log(tf.clip_by_value(probs, 1e-7, 1.0))
    return tf.nn.softmax(logits / temperature, axis=1)


class Distiller(tf.keras.Model):
    """Trains `student` against labels and a frozen `teacher`.

    L_KD = (1 - lambda) * CE(q_S, y) + lambda * tau^2 * KL(q_T^tau || q_S^tau)
    """

    def __init__(self, student, teacher, temperature=config.TEMPERATURE, kd_lambda=config.KD_LAMBDA):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False
        self.temperature = temperature
        self.kd_lambda = kd_lambda
        self.ce_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.kl_fn = tf.keras.losses.KLDivergence()

        self.ce_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        self.total_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.ce_tracker, self.kl_tracker, self.total_tracker, self.acc_tracker]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def _kd_losses(self, y, s_pred, t_pred):
        ce = self.ce_fn(y, s_pred)
        kl = self.kl_fn(soften(t_pred, self.temperature), soften(s_pred, self.temperature))
        total = (1.0 - self.kd_lambda) * ce + self.kd_lambda * (self.temperature ** 2) * kl
        return ce, kl, total

    def _update_metrics(self, y, ce, kl, total, s_pred):
        self.ce_tracker.update_state(ce)
        self.kl_tracker.update_state(kl)
        self.total_tracker.update_state(total)
        self.acc_tracker.update_state(y, s_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        (img, wav), y = data
        t_pred = self.teacher([img, wav], training=False)
        with tf.GradientTape() as tape:
            s_pred = self.student([img, wav], training=True)
            ce, kl, total = self._kd_losses(y, s_pred, t_pred)
        grads = tape.gradient(total, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        return self._update_metrics(y, ce, kl, total, s_pred)

    def test_step(self, data):
        (img, wav), y = data
        t_pred = self.teacher([img, wav], training=False)
        s_pred = self.student([img, wav], training=False)
        ce, kl, total = self._kd_losses(y, s_pred, t_pred)
        return self._update_metrics(y, ce, kl, total, s_pred)
