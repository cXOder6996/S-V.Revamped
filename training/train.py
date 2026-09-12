"""
SkinVision Training Pipeline — refactored for E1-E8 experiments.

Changes from original:
  - build_model() now reads architecture + input_size from config (D3 fix)
  - Supports EfficientNetB0, B3, B4, EfficientNetV2S (E4)
  - Loss function driven by focal_loss / label_smoothing / focal_gamma (D4 fix)
  - lr_scheduler field selects ReduceLROnPlateau vs CosineDecayRestarts (E7)
  - unfreeze_layers applied as absolute int; backbone total layer count logged (D5)
  - ExperimentLogger called at end of every run (D7 fix)
  - Per-epoch val macro-F1 callback added alongside val_loss monitoring (D8 fix)
  - seed_everything() called at startup for reproducibility
  - Inference latency measured on a representative single image after training

Augmentation NOTE (D6): get_augmentations() still matches what was ACTUALLY
applied during E1, not what config.yaml documents. This is intentional — it
preserves E1 reproducibility. config.yaml augmentation fields are reserved for
future experiment variants and annotated accordingly.
"""

import json
import logging
import os
import sys
import time
import argparse

# Project root on path so training modules resolve
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

try:
    import albumentations as A
except ImportError:
    A = None

import sklearn

from training.focal_loss import get_loss_fn
from training.experiment_logger import ExperimentLogger, seed_everything
from training.evaluate import evaluate_model, measure_inference_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported backbone registry
# ---------------------------------------------------------------------------
_BACKBONES = {
    "EfficientNetB0": (tf.keras.applications.EfficientNetB0, None),
    "EfficientNetB3": (tf.keras.applications.EfficientNetB3, None),
    "EfficientNetB4": (tf.keras.applications.EfficientNetB4, None),
    "EfficientNetV2S": (tf.keras.applications.EfficientNetV2S, None),
}


def verify_environment():
    logger.info("=== Environment Verification ===")
    logger.info(f"TensorFlow: {tf.__version__}")
    try:
        logger.info(f"Keras: {tf.keras.__version__}")
    except AttributeError:
        pass
    logger.info(f"scikit-learn: {sklearn.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    logger.info(f"GPUs available: {len(gpus)} — {[g.name for g in gpus]}")
    logger.info("================================\n")


def parse_args():
    p = argparse.ArgumentParser(description="SkinVision Training Pipeline")
    p.add_argument("--data-dir",   required=True,               help="HAM10000 dataset root")
    p.add_argument("--config",     default="training/config.yaml")
    p.add_argument("--experiment", default=None,                help="Experiment ID (e.g. E2)")
    p.add_argument("--notes",      default="",                  help="Free-text run notes")
    args = p.parse_args()
    if args.data_dir:
        args.data_dir = args.data_dir.strip("\"' ")
    if args.config:
        args.config = args.config.strip("\"' ")
    if args.experiment:
        args.experiment = args.experiment.strip("\"' ")
    if args.notes:
        args.notes = args.notes.strip("\"' ")
    return args


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_splits(metadata_path: str, n_splits: int = 7, seed: int = 42,
                   test_fold: int = 0, val_fold: int = 1):
    df = pd.read_csv(metadata_path)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df["dx"], df["lesion_id"])):
        df.loc[val_idx, "fold"] = fold

    test_df  = df[df["fold"] == test_fold].copy()
    val_df   = df[df["fold"] == val_fold].copy()
    train_df = df[df["fold"].isin([f for f in range(n_splits)
                                    if f not in (test_fold, val_fold)])].copy()

    assert test_fold not in train_df["fold"].values, f"Test fold ({test_fold}) must never be in training data!"
    assert test_fold not in val_df["fold"].values, f"Test fold ({test_fold}) must never be in validation data!"

    logger.info(f"Split — Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_augmentations():
    """
    Returns the augmentation pipeline that was ACTUALLY applied during E1.
    Preserved verbatim for reproducibility. Do NOT change for E2–E4 experiments.
    (augmentation is an independent experimental variable not yet scheduled)
    """
    if A is None:
        raise ImportError("albumentations is required for data augmentation. Install it via pip install albumentations==1.4.21")
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
    ])


class SkinVisionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, data_dir, batch_size=32, target_size=(224, 224),
                 augment=False, class_mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.df           = df.reset_index(drop=True)
        self.data_dir     = data_dir
        self.batch_size   = batch_size
        self.target_size  = target_size
        self.augment      = augment
        self.class_mapping = class_mapping
        self.image_ids    = self.df["image_id"].values
        self.labels       = self.df["dx"].map(self.class_mapping).values
        self.augmentor    = get_augmentations() if self.augment else None
        self.indexes      = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        if len(self.df) == 0:
            return 0
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(batch_idx)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        n_samples = len(batch_indexes)
        X = np.empty((n_samples, *self.target_size, 3), dtype=np.float32)
        y = np.empty(n_samples, dtype=int)

        for i, idx in enumerate(batch_indexes):
            img_id = self.image_ids[idx]
            candidates = [
                os.path.join(self.data_dir, f"{img_id}.jpg"),
                os.path.join(self.data_dir, "images", f"{img_id}.jpg"),
                os.path.join(self.data_dir, "HAM10000_images_part_1", f"{img_id}.jpg"),
                os.path.join(self.data_dir, "HAM10000_images_part_2", f"{img_id}.jpg"),
            ]
            path = next((p for p in candidates if os.path.exists(p)), None)
            if path is None:
                raise FileNotFoundError(f"Image not found for ID {img_id}")

            if cv2 is not None:
                img = cv2.imread(path)
                if img is None:
                    from PIL import Image
                    try:
                        pil_img = Image.open(path).convert("RGB")
                        img = np.array(pil_img)
                    except Exception as e:
                        logger.warning(f"Corrupt or unreadable image {path}: {e}. Using zero array fallback.")
                        img = np.zeros((*self.target_size, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.augmentor:
                    img = self.augmentor(image=img)["image"]
                img = cv2.resize(img, self.target_size)
            else:
                from PIL import Image
                try:
                    pil_img = Image.open(path).convert("RGB")
                except Exception as e:
                    logger.warning(f"Corrupt or unreadable image {path}: {e}. Using zero array fallback.")
                    pil_img = Image.new("RGB", self.target_size, (0, 0, 0))
                if self.augmentor:
                    img_np = np.array(pil_img)
                    img_np = self.augmentor(image=img_np)["image"]
                    pil_img = Image.fromarray(img_np)
                pil_img = pil_img.resize(self.target_size, Image.Resampling.BILINEAR)
                img = np.array(pil_img)

            X[i] = img.astype(np.float32)   # raw [0, 255] — no normalisation
            y[i] = self.labels[idx]

        return X, tf.keras.utils.to_categorical(y, num_classes=len(self.class_mapping))


class ValMacroF1Callback(tf.keras.callbacks.Callback):
    """Computes and logs validation macro-F1 at the end of each epoch."""

    def __init__(self, val_gen, classes):
        super().__init__()
        self.val_gen = val_gen
        self.classes = classes

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import f1_score
        if len(self.val_gen) == 0:
            return
        preds, trues = [], []
        for i in range(len(self.val_gen)):
            X, y = self.val_gen[i]
            p = self.model.predict(X, verbose=0)
            preds.append(np.argmax(p, axis=1))
            trues.append(np.argmax(y, axis=1))
        if not preds:
            return
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if logs is not None:
            logs["val_macro_f1"] = macro_f1
        logger.info(f"  Epoch {epoch + 1} — val_macro_f1: {macro_f1:.4f}")


def build_model(architecture: str, input_size: tuple, num_classes: int) -> tf.keras.Model:
    """
    Builds a transfer-learning model using the specified backbone.

    Args:
        architecture: One of the keys in _BACKBONES.
        input_size:   (H, W) tuple — must match backbone's native resolution for
                      best performance, but any size >= 32 is accepted.
        num_classes:  Number of output classes.

    Returns:
        A compiled-ready Keras Model with frozen backbone.
    """
    if architecture not in _BACKBONES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Supported: {list(_BACKBONES.keys())}"
        )

    backbone_cls, _ = _BACKBONES[architecture]

    # All supported backbones include an internal Rescaling layer and expect
    # raw [0, 255] float input. preprocess_input() is a no-op in TF 2.20.
    base = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(input_size[0], input_size[1], 3),
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)

    # Log backbone layer count for unfreeze-fraction bookkeeping (D5)
    logger.info(f"[build_model] {architecture} backbone layers: {len(base.layers)}")
    return model, len(base.layers)


def get_lr_callbacks(train_cfg: dict, checkpoint_path: str):
    """Returns EarlyStopping, scheduler, and ModelCheckpoint callbacks."""
    cb_cfg      = train_cfg.get("callbacks", {})
    es_patience = cb_cfg.get("early_stopping", {}).get("patience", 7)
    rlr_cfg     = cb_cfg.get("reduce_lr", {})
    scheduler   = train_cfg.get("lr_scheduler", "reduce_lr")

    early_stop = EarlyStopping(
        monitor="val_loss", patience=es_patience, restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True
    )

    if scheduler == "cosine":
        # CosineDecayRestarts is attached directly to optimizer; no separate plateau callback
        logger.info("[LR] CosineDecayRestarts scheduler active on optimizer")
        return [early_stop, checkpoint]
    else:
        # Default: ReduceLROnPlateau (used in E1–E6)
        lr_cb = ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(rlr_cfg.get("factor", 0.5)),
            patience=int(rlr_cfg.get("patience", 5)),
            min_lr=float(rlr_cfg.get("min_lr", 1e-7)),
        )
        logger.info("[LR] ReduceLROnPlateau callback active")
        return [early_stop, lr_cb, checkpoint]


def apply_fine_tuning(model, phase: dict, base_layer_count: int):
    """
    Unfreezes the top N layers of the nested backbone.
    BatchNorm layers are kept frozen throughout (standard practice).

    Args:
        model:            The full Keras Model.
        phase:            A fine_tuning phase config dict.
        base_layer_count: Total layers in the backbone (for logging fraction).
    """
    unfreeze_n = int(phase.get("unfreeze_layers", 20))
    fraction   = round(unfreeze_n / base_layer_count * 100, 1) if base_layer_count else "?"

    # Locate the nested backbone sub-model
    nested = next((l for l in model.layers if hasattr(l, "layers")), None)

    if unfreeze_n <= 0:
        # Keep entire backbone frozen
        if nested is not None:
            nested.trainable = False
            for lyr in nested.layers:
                lyr.trainable = False
        else:
            for lyr in model.layers[:-3]:
                lyr.trainable = False
            for lyr in model.layers[-3:]:
                lyr.trainable = True
    else:
        if nested is not None:
            nested.trainable = True
            for lyr in nested.layers[:-unfreeze_n]:
                lyr.trainable = False
            for lyr in nested.layers[-unfreeze_n:]:
                if not isinstance(lyr, tf.keras.layers.BatchNormalization):
                    lyr.trainable = True
        else:
            backbone_layers = model.layers[:-3]
            for lyr in backbone_layers[:-unfreeze_n]:
                lyr.trainable = False
            for lyr in backbone_layers[-unfreeze_n:]:
                if not isinstance(lyr, tf.keras.layers.BatchNormalization):
                    lyr.trainable = True
            for lyr in model.layers[-3:]:
                lyr.trainable = True

    if nested is not None:
        trainable_count = sum(1 for l in nested.layers if l.trainable)
    else:
        trainable_count = sum(1 for l in model.layers[:-3] if l.trainable)

    logger.info(
        f"[fine_tuning] Unfroze top {unfreeze_n} backbone layers "
        f"({fraction}% of {base_layer_count}). "
        f"Total trainable backbone layers: {trainable_count}"
    )
    return unfreeze_n


def measure_inference_latency(model, input_size: tuple, n_warmup: int = 5, n_runs: int = 30) -> dict:
    """
    Measures single-image inference latency on a random input.
    Returns {"p50": float, "p95": float} in milliseconds.
    """
    dummy = np.random.rand(1, input_size[0], input_size[1], 3).astype(np.float32) * 255
    for _ in range(n_warmup):
        model.predict(dummy, verbose=0)
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "p50": round(float(np.percentile(latencies, 50)), 2),
        "p95": round(float(np.percentile(latencies, 95)), 2),
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    experiment_id = args.experiment or config.get("experiment_id", "E1")
    config["experiment_id"] = experiment_id

    train_cfg = config.get("training", {})
    data_cfg  = config.get("data", {})
    model_cfg = config.get("model", {})
    aug_cfg   = config.get("augmentation", {})

    seed_val = int(data_cfg.get("seed", 42))
    seed_everything(seed_val)

    verify_environment()

    # ── Architecture & resolution from config (D3 fix) ────────────────────────
    architecture = model_cfg.get("architecture", "EfficientNetB0")
    input_size   = tuple(model_cfg.get("input_size", [224, 224]))
    num_classes  = int(model_cfg.get("num_classes", 7))
    logger.info(f"[config] experiment_id={experiment_id}  architecture={architecture}  input_size={input_size}")

    classes       = sorted(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
    class_mapping = {c: i for i, c in enumerate(classes)}

    metadata_path = os.path.join(args.data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    train_df, val_df, test_df = prepare_splits(
        metadata_path,
        n_splits=data_cfg.get("n_splits", 7),
        seed=seed_val,
        test_fold=data_cfg.get("test_fold", 0),
        val_fold=data_cfg.get("val_fold", 1),
    )

    assert 0 not in train_df["fold"].values, "Test fold 0 must never be in training set!"
    assert 0 not in val_df["fold"].values, "Test fold 0 must never be in validation set!"

    batch_size      = int(train_cfg.get("batch_size", 32))
    augment_enabled = aug_cfg.get("enabled", False)

    train_gen = SkinVisionDataGenerator(
        train_df, args.data_dir, batch_size=batch_size,
        target_size=input_size, augment=augment_enabled, class_mapping=class_mapping,
    )
    val_gen = SkinVisionDataGenerator(
        val_df, args.data_dir, batch_size=batch_size,
        target_size=input_size, augment=False, class_mapping=class_mapping,
    )

    # ── Class weights ──────────────────────────────────────────────────────────
    class_weight_dict = None
    if train_cfg.get("class_weights", False):
        train_labels = train_df["dx"].map(class_mapping).values
        weights      = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        class_weight_dict = {int(k): float(v) for k, v in enumerate(weights)}
        logger.info(f"[class_weights] {class_weight_dict}")

    # ── Loss function (D4 fix) ────────────────────────────────────────────────
    loss_fn = get_loss_fn(config)
    logger.info(f"[loss] {loss_fn}")

    # ── Build model ───────────────────────────────────────────────────────────
    model, backbone_layer_count = build_model(architecture, input_size, num_classes)

    # ── Output directories ────────────────────────────────────────────────────
    out_dir         = os.path.join("experiments", experiment_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join(out_dir, f"{experiment_id}_best.keras")
    canonical_path  = os.path.join("models", "best_model.keras")

    val_f1_cb = ValMacroF1Callback(val_gen, classes)
    callbacks = [val_f1_cb] + get_lr_callbacks(train_cfg, checkpoint_path)

    frozen_epochs = int(train_cfg.get("frozen_epochs", 15))
    initial_lr    = float(train_cfg.get("initial_lr", 1e-3))

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    scheduler = train_cfg.get("lr_scheduler", "reduce_lr")
    steps_per_epoch = max(1, len(train_gen))

    if scheduler == "cosine":
        cosine_cfg = train_cfg.get("cosine_decay") or train_cfg.get("callbacks", {}).get("cosine_decay", {})
        t0    = int(cosine_cfg.get("t_0", 10))
        t_mul = float(cosine_cfg.get("t_mul", 2.0))
        alpha = float(cosine_cfg.get("alpha", 1e-6))
        lr_schedule_p1 = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=steps_per_epoch * t0,
            t_mul=t_mul,
            alpha=alpha,
        )
        opt_p1 = Adam(learning_rate=lr_schedule_p1)
        logger.info(f"[LR] Phase 1 CosineDecayRestarts: T_0={t0}, T_mul={t_mul}, alpha={alpha}")
    else:
        opt_p1 = Adam(learning_rate=initial_lr)

    logger.info(f"=== Phase 1: Frozen backbone — {frozen_epochs} epochs @ lr={initial_lr} ===")
    model.compile(optimizer=opt_p1, loss=loss_fn, metrics=["accuracy"])

    history_total = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
    fit_start = time.time()

    h1 = model.fit(
        train_gen, validation_data=val_gen, epochs=frozen_epochs,
        callbacks=callbacks, class_weight=class_weight_dict,
    )
    for k in history_total:
        if k in h1.history:
            history_total[k].extend([float(v) for v in h1.history[k]])

    # ── Phase 2+: fine-tuning ─────────────────────────────────────────────────
    ft_cfg   = train_cfg.get("fine_tuning", {})
    last_unfreeze = 0
    if ft_cfg.get("enabled", False):
        for p_idx, phase in enumerate(ft_cfg.get("phases", [])):
            phase_lr     = float(phase.get("lr", 1e-4))
            phase_epochs = int(phase.get("epochs", 10))
            last_unfreeze = apply_fine_tuning(model, phase, backbone_layer_count)

            logger.info(f"=== Fine-Tune Phase {p_idx+1}: {phase_epochs} epochs @ lr={phase_lr} ===")
            if scheduler == "cosine":
                lr_schedule_ft = tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=phase_lr,
                    first_decay_steps=steps_per_epoch * t0,
                    t_mul=t_mul,
                    alpha=alpha,
                )
                phase_opt = Adam(learning_rate=lr_schedule_ft)
                logger.info(f"[LR] Phase {p_idx+1} CosineDecayRestarts: T_0={t0}, T_mul={t_mul}, alpha={alpha}")
            else:
                phase_opt = Adam(learning_rate=phase_lr)

            model.compile(optimizer=phase_opt, loss=loss_fn, metrics=["accuracy"])

            h_ft = model.fit(
                train_gen, validation_data=val_gen, epochs=phase_epochs,
                callbacks=callbacks, class_weight=class_weight_dict,
            )
            for k in history_total:
                if k in h_ft.history:
                    history_total[k].extend([float(v) for v in h_ft.history[k]])

    training_time_seconds = time.time() - fit_start

    # ── Promote best checkpoint & restore weights for evaluation (D8) ────────
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading best checkpoint weights from {checkpoint_path} for final evaluation and promotion...")
        model.load_weights(checkpoint_path)
        # Re-compile with standard categorical_crossentropy before saving canonical_path
        # so src/inference.py (and Streamlit app) can load without requiring custom loss registration.
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.save(canonical_path)
        logger.info(f"Promoted best model -> {canonical_path}")

    # ── Save training history ─────────────────────────────────────────────────
    history_path = os.path.join(out_dir, f"{experiment_id}_history.json")
    with open(history_path, "w") as f:
        json.dump(history_total, f, indent=2)
    # Also keep canonical copy for Streamlit
    with open(os.path.join("models", "training_history.json"), "w") as f:
        json.dump(history_total, f, indent=2)

    # ── Validation evaluation ────────────────────────────────────────────────
    logger.info("=== Validation Evaluation ===")
    val_preds, val_true = [], []
    for i in range(len(val_gen)):
        X, y = val_gen[i]
        val_preds.append(model.predict(X, verbose=0))
        val_true.append(np.argmax(y, axis=1))

    val_probs  = np.vstack(val_preds)
    val_labels = np.concatenate(val_true)

    try:
        from training.evaluate import evaluate_model
    except ImportError:
        from evaluate import evaluate_model

    eval_out = os.path.join(out_dir, f"{experiment_id}_eval.json")
    results  = evaluate_model(val_labels, val_probs, classes, output_path=eval_out)
    # Canonical copy for Streamlit Model Info page
    import shutil
    shutil.copyfile(eval_out, os.path.join("models", "evaluation_results.json"))

    prim     = results.get("primary_metrics", results)
    sec      = results.get("secondary_metrics", results)
    macro_f1 = prim.get("macro_f1", results.get("macro_f1", 0.0))
    bal_acc  = prim.get("balanced_accuracy", results.get("balanced_accuracy", 0.0))
    acc      = sec.get("overall_accuracy", results.get("accuracy", 0.0))
    mel_rec  = prim.get("classification_report", {}).get("mel", {}).get("recall", "N/A")

    logger.info(f"  Macro F1:        {macro_f1:.4f}")
    logger.info(f"  Balanced Acc:    {bal_acc:.4f}")
    logger.info(f"  Overall Acc:     {acc:.4f}")
    logger.info(f"  mel recall:      {mel_rec}")

    # ── Inference latency ─────────────────────────────────────────────────────
    logger.info("Measuring inference latency...")
    latency_ms = measure_inference_latency(model, input_size)
    logger.info(f"  Latency P50: {latency_ms['p50']}ms  P95: {latency_ms['p95']}ms")

    # ── Experiment log ────────────────────────────────────────────────────────
    exp_log = ExperimentLogger(experiment_id, out_dir="experiments")
    record_path = exp_log.log(
        config_snapshot=config,
        eval_results=results,
        training_time_seconds=training_time_seconds,
        inference_latency_ms=latency_ms,
        model_path=os.path.abspath(canonical_path),
        backbone_total_layers=backbone_layer_count,
        notes=args.notes,
    )
    logger.info(f"Experiment record: {record_path}")
    logger.info("Training pipeline completed.")


if __name__ == "__main__":
    main()
