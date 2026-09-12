"""
Offline Test-Time Augmentation (TTA) evaluator — E8 only.

Evaluates a frozen, already-selected model on the validation set using TTA.
NOT integrated into the Streamlit app until E8 passes acceptance criteria:
  - macro-F1 gain >= 0.010 vs single-pass
  - TTA latency < 3x single-pass latency (P50)

Usage:
    python training/tta_eval.py \
        --model models/best_model.keras \
        --data-dir /path/to/ham10000 \
        --config training/config.yaml \
        --n-augmentations 4 \
        --output experiments/E8/tta_results.json
"""

import argparse
import json
import logging
import os
import sys
import time

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Offline TTA evaluation for SkinVision")
    p.add_argument("--model",           required=True,  help="Path to .keras model file")
    p.add_argument("--data-dir",        required=True,  help="HAM10000 dataset directory")
    p.add_argument("--config",          default="training/config.yaml")
    p.add_argument("--n-augmentations", type=int, default=4,
                   help="TTA variants: 1=original, 2=+hflip, 3=+vflip, 4=+rot90")
    p.add_argument("--output",          required=True,  help="Output JSON path")
    args = p.parse_args()
    if args.model:
        args.model = args.model.strip("\"' ")
    if args.data_dir:
        args.data_dir = args.data_dir.strip("\"' ")
    if args.config:
        args.config = args.config.strip("\"' ")
    if args.output:
        args.output = args.output.strip("\"' ")
    return args


def tta_variants(image: np.ndarray, n: int) -> list:
    """
    Returns up to n geometric variants of a uint8 HWC RGB image.
    Order: original, horizontal flip, vertical flip, 90 clockwise rotation.
    """
    n = max(1, n)
    variants = [image]
    if n >= 2:
        variants.append(cv2.flip(image, 1) if cv2 is not None else np.ascontiguousarray(np.fliplr(image)))
    if n >= 3:
        variants.append(cv2.flip(image, 0) if cv2 is not None else np.ascontiguousarray(np.flipud(image)))
    if n >= 4:
        variants.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) if cv2 is not None else np.ascontiguousarray(np.rot90(image, 3)))
    return variants[:n]


def measure_inference_latency(model, inp: np.ndarray, n_warmup: int = 3, n_measure: int = 20) -> dict:
    """
    Measures model inference latency over n_measure runs after n_warmup warm-up passes.

    Returns:
        dict with keys "p50" and "p95" (milliseconds, float).
    """
    # Warm-up
    for _ in range(n_warmup):
        model.predict(inp, verbose=0)

    latencies = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        model.predict(inp, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "p50": round(float(np.percentile(latencies, 50)), 2),
        "p95": round(float(np.percentile(latencies, 95)), 2),
    }


def run_tta_eval(model, val_gen, n_augmentations: int, input_size: tuple):
    """
    Runs both single-pass and TTA inference over the full validation generator.

    Returns:
        single_preds (N, C), tta_preds (N, C), true_labels (N,),
        single_lat_ms (list), tta_lat_ms (list)
    """
    single_preds_list, tta_preds_list, true_list = [], [], []
    single_latencies, tta_latencies = [], []

    total_batches = len(val_gen)
    for batch_idx in range(total_batches):
        if batch_idx % 10 == 0:
            logger.info(f"  Processing batch {batch_idx}/{total_batches}")

        X_batch, y_batch = val_gen[batch_idx]   # X: (B, H, W, 3) float32 [0,255]
        true_list.append(np.argmax(y_batch, axis=1))

        for img_float in X_batch:
            img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
            inp       = np.expand_dims(img_float, 0)

            # Single-pass
            t0 = time.perf_counter()
            sp = model.predict(inp, verbose=0)[0]
            single_latencies.append((time.perf_counter() - t0) * 1000)
            single_preds_list.append(sp)

            # TTA
            t0       = time.perf_counter()
            variants = tta_variants(img_uint8, n_augmentations)
            aug_imgs = []
            for v in variants:
                if cv2 is not None:
                    v_resized = cv2.resize(v, input_size).astype(np.float32)
                else:
                    from PIL import Image
                    v_resized = np.array(Image.fromarray(v).resize(input_size, Image.Resampling.BILINEAR), dtype=np.float32)
                aug_imgs.append(v_resized)
            aug_stack = np.stack(aug_imgs)
            aug_preds = model.predict(aug_stack, verbose=0)
            tta_pred  = np.mean(aug_preds, axis=0)
            tta_latencies.append((time.perf_counter() - t0) * 1000)
            tta_preds_list.append(tta_pred)

    return (
        np.array(single_preds_list) if single_preds_list else np.empty((0, 7)),
        np.array(tta_preds_list) if tta_preds_list else np.empty((0, 7)),
        np.concatenate(true_list) if true_list else np.array([], dtype=int),
        single_latencies,
        tta_latencies,
    )


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Add project root to path so imports resolve correctly on Colab
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import tensorflow as tf
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    from training.evaluate import evaluate_model
    from training.train import SkinVisionDataGenerator

    logger.info(f"Loading model from {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    data_cfg       = config.get("data", {})
    input_size_cfg = config.get("model", {}).get("input_size", [224, 224])
    input_size     = tuple(input_size_cfg)

    metadata_path = os.path.join(args.data_dir, "HAM10000_metadata.csv")
    df = pd.read_csv(metadata_path)

    sgkf = StratifiedGroupKFold(
        n_splits=data_cfg.get("n_splits", 7),
        shuffle=True,
        random_state=data_cfg.get("seed", 42),
    )
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df["dx"], df["lesion_id"])):
        df.loc[val_idx, "fold"] = fold

    test_fold = int(data_cfg.get("test_fold", 0))
    val_fold  = int(data_cfg.get("val_fold", 1))
    assert val_fold != test_fold, f"Validation fold ({val_fold}) cannot be the test fold ({test_fold})!"
    val_df = df[df["fold"] == val_fold].copy()
    assert 0 not in val_df["fold"].values, "Test set (fold 0) must never be loaded or evaluated in tta_eval!"
    classes      = sorted(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
    class_mapping = {c: i for i, c in enumerate(classes)}

    val_gen = SkinVisionDataGenerator(
        val_df, args.data_dir,
        batch_size=config.get("training", {}).get("batch_size", 32),
        target_size=input_size,
        augment=False,
        class_mapping=class_mapping,
    )

    logger.info(f"Running TTA with n_augmentations={args.n_augmentations}")
    single_preds, tta_preds, true_labels, single_lat, tta_lat = run_tta_eval(
        model, val_gen, args.n_augmentations, input_size
    )

    single_metrics = evaluate_model(true_labels, single_preds, classes)
    tta_metrics    = evaluate_model(true_labels, tta_preds,    classes)

    def pct(lat, p):
        return round(float(np.percentile(lat, p)), 2) if len(lat) > 0 else 0.0

    single_p50, single_p95 = pct(single_lat, 50), pct(single_lat, 95)
    tta_p50,    tta_p95    = pct(tta_lat, 50),    pct(tta_lat, 95)

    single_f1 = single_metrics["macro_f1"]
    tta_f1    = tta_metrics["macro_f1"]
    f1_gain   = round(tta_f1 - single_f1, 4)
    lat_ratio = round(tta_p50 / single_p50, 2) if single_p50 > 0 else None

    def mel_recall(metrics_dict):
        return (metrics_dict.get("primary_metrics", {})
                            .get("classification_report", {})
                            .get("mel", {}).get("recall", None))

    # E8 acceptance: gain >= 0.010 AND ratio < 3.0
    e8_accepted = (f1_gain >= 0.010) and (lat_ratio is not None and lat_ratio < 3.0)

    results = {
        "n_augmentations": args.n_augmentations,
        "single_pass": {
            "macro_f1":       single_f1,
            "balanced_acc":   single_metrics["balanced_accuracy"],
            "mel_recall":     mel_recall(single_metrics),
            "latency_p50_ms": single_p50,
            "latency_p95_ms": single_p95,
        },
        "tta": {
            "macro_f1":       tta_f1,
            "balanced_acc":   tta_metrics["balanced_accuracy"],
            "mel_recall":     mel_recall(tta_metrics),
            "latency_p50_ms": tta_p50,
            "latency_p95_ms": tta_p95,
        },
        "comparison": {
            "macro_f1_gain":     f1_gain,
            "latency_ratio_p50": lat_ratio,
        },
        "e8_acceptance": {
            "criterion_f1_gain_ge_0.010":   f1_gain >= 0.010,
            "criterion_latency_ratio_lt_3x": lat_ratio is not None and lat_ratio < 3.0,
            "accepted_for_production":       e8_accepted,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    logger.info(f"Results written to {args.output}")
    logger.info(f"  Single macro-F1={single_f1:.4f}  TTA macro-F1={tta_f1:.4f}  gain={f1_gain:+.4f}")
    logger.info(f"  Single P50={single_p50}ms  TTA P50={tta_p50}ms  ratio={lat_ratio}x")
    logger.info(f"  E8 accepted for production: {e8_accepted}")


if __name__ == "__main__":
    main()
