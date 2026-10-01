"""
Offline Test-Time Augmentation (TTA) evaluator — E8 only.

Evaluates a frozen, already-selected model on the held-out test set using TTA.
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

        # X_batch: (B, H, W, 3)
        # Single-pass prediction
        t0 = time.perf_counter()
        sp_batch = model(X_batch, training=False).numpy()
        single_latencies.append((time.perf_counter() - t0) * 1000 / len(X_batch))
        single_preds_list.extend(sp_batch)

        # TTA prediction
        t0 = time.perf_counter()
        aug_batch = []
        for img_float in X_batch:
            img_uint8 = np.clip(img_float, 0, 255).astype(np.uint8)
            variants = tta_variants(img_uint8, n_augmentations)
            for v in variants:
                if cv2 is not None:
                    aug_batch.append(cv2.resize(v, input_size).astype(np.float32))
                else:
                    from PIL import Image
                    aug_batch.append(np.array(Image.fromarray(v).resize(input_size, Image.Resampling.BILINEAR), dtype=np.float32))
        
        aug_stack = np.stack(aug_batch) # Shape: (B * n_augmentations, H, W, 3)
        
        # We can predict on the whole stack at once, or if it's too large, in chunks
        aug_preds = model(aug_stack, training=False).numpy() # (B * n_augmentations, 7)
        
        # Reshape to (B, n_augmentations, 7) and mean across augmentations
        aug_preds = aug_preds.reshape(len(X_batch), n_augmentations, -1)
        tta_batch = np.mean(aug_preds, axis=1)
        
        tta_latencies.append((time.perf_counter() - t0) * 1000 / len(X_batch))
        tta_preds_list.extend(tta_batch)

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
    test_df = df[df["fold"] == test_fold].copy()
    
    classes      = sorted(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
    class_mapping = {c: i for i, c in enumerate(classes)}

    test_gen = SkinVisionDataGenerator(
        test_df, args.data_dir,
        batch_size=config.get("training", {}).get("batch_size", 32),
        target_size=input_size,
        augment=False,
        class_mapping=class_mapping,
    )

    logger.info(f"Running FINAL TEST EVALUATION on fold {test_fold} with TTA n={args.n_augmentations}")
    _, tta_preds, true_labels, _, _ = run_tta_eval(
        model, test_gen, args.n_augmentations, input_size
    )

    # Evaluate the TTA predictions
    tta_metrics = evaluate_model(true_labels, tta_preds, classes, output_path=args.output)
    
    logger.info(f"Test Results written to {args.output}")
    logger.info(f"  Accuracy: {tta_metrics['accuracy']:.4f}")
    logger.info(f"  Macro-F1: {tta_metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted-F1: {tta_metrics['primary_metrics']['weighted_f1']:.4f}")
    logger.info(f"  Balanced Accuracy: {tta_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Melanoma Recall: {tta_metrics['primary_metrics']['classification_report']['mel']['recall']:.4f}")
    logger.info(f"  DF Recall: {tta_metrics['primary_metrics']['classification_report']['df']['recall']:.4f}")
    logger.info(f"  ECE: {tta_metrics['calibration']['expected_calibration_error']:.4f}")

if __name__ == "__main__":
    main()
