"""
Focal Loss for SkinVision — E5 and E6b experiments.

Focal Loss (Lin et al., 2017) down-weights easy examples so training focuses
on hard misclassifications. Useful for class-imbalanced datasets like HAM10000.

  FL(p_t) = -(1 - p_t)^gamma * log(p_t)

Fully compatible with TF 2.20 / Keras 3.11 with serialization support.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="SkinVision")
class FocalLoss(tf.keras.losses.Loss):
    """
    Keras-compatible Focal Loss implementation supporting serialization.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        from_logits: bool = False,
        name: str = "focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Clip to prevent log(0)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # Apply label smoothing if configured
        if self.label_smoothing > 0.0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        # Cross-entropy term per class
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Focal weight: (1 - p_t)^gamma applied to probability
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)

        # Sum over classes, mean over batch
        return tf.reduce_mean(tf.reduce_sum(focal_weight * cross_entropy, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits,
        })
        return config


def focal_loss(gamma: float = 2.0, label_smoothing: float = 0.0, from_logits: bool = False):
    """
    Returns a focal loss callable compatible with model.compile(loss=...).

    Args:
        gamma:           Focusing parameter (default 2.0). 0 recovers standard cross-entropy.
        label_smoothing: Float in [0, 1]. When > 0, smooths one-hot targets.
        from_logits:     Whether y_pred contains raw logits or softmax probabilities.

    Returns:
        FocalLoss instance compatible with Keras model.compile() and model saving/loading.
    """
    return FocalLoss(gamma=gamma, label_smoothing=label_smoothing, from_logits=from_logits)


def get_loss_fn(config: dict):
    """
    Factory: returns the correct loss function based on training config.

    Reads from config["training"]:
        focal_loss      (bool,  default False)
        focal_gamma     (float, default 2.0)   — used when focal_loss=True
        label_smoothing (float, default 0.0)   — used for smoothed CE or combined focal

    Per experiment plan:
        - focal_loss=True  and label_smoothing=0.0  -> focal loss   (E5)
        - focal_loss=False and label_smoothing=0.1  -> smoothed CE  (E6)
        - focal_loss=True  and label_smoothing>0.0  -> combined     (E6b)
        - focal_loss=False and label_smoothing=0.0  -> standard CE  (E1/E2/E3/E4)

    Returns:
        A Keras-compatible loss function or the string "categorical_crossentropy".
    """
    if isinstance(config, dict) and "training" in config:
        train_cfg = config["training"] or {}
    elif isinstance(config, dict):
        train_cfg = config
    else:
        train_cfg = {}

    use_focal       = bool(train_cfg.get("focal_loss", False))
    focal_gamma     = float(train_cfg.get("focal_gamma", 2.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    if use_focal:
        return focal_loss(gamma=focal_gamma, label_smoothing=label_smoothing)

    if label_smoothing > 0.0:
        return tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        )

    return "categorical_crossentropy"
