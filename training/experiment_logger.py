"""
Experiment Logger for SkinVision.

Writes a self-contained JSON record for every training run and appends
a one-row summary to experiments/results_table.csv.

Typical usage from train.py:
    from training.experiment_logger import ExperimentLogger, seed_everything
    exp_log = ExperimentLogger("E2", out_dir="experiments")
    exp_log.log(config, eval_results, training_secs, latency_ms, model_path)
"""

import csv
import json
import logging
import os
import platform
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# E1 baseline macro-F1 — locked reference, never changes.
_E1_MACRO_F1 = 0.6170

_CSV_COLUMNS = [
    "experiment_id", "timestamp_utc", "architecture", "input_size",
    "loss_fn", "label_smoothing", "focal_gamma", "unfreeze_layers",
    "backbone_total_layers", "unfreeze_fraction_pct", "frozen_epochs",
    "finetune_epochs", "initial_lr", "finetune_lr", "lr_scheduler",
    "class_weights", "batch_size", "seed", "split_method", "n_splits",
    "test_fold", "val_fold", "augmentation_note",
    "val_macro_f1", "val_balanced_acc", "val_overall_acc", "val_weighted_f1",
    "mel_recall", "mel_f1", "df_recall", "ece",
    "model_size_mb", "training_time_min",
    "inference_latency_p50_ms", "inference_latency_p95_ms",
    "vs_e1_macro_f1_delta", "accepted", "notes",
]


class ExperimentLogger:
    """Logs a single experiment configuration and results."""

    def __init__(self, experiment_id: str, out_dir: str = "experiments"):
        self.experiment_id = experiment_id
        self.out_dir = out_dir
        self.exp_dir = os.path.join(out_dir, experiment_id)
        os.makedirs(self.exp_dir, exist_ok=True)

    def log(
        self,
        config_snapshot: dict,
        eval_results: dict,
        training_time_seconds: float,
        inference_latency_ms: dict,
        model_path: str,
        backbone_total_layers: int = None,
        notes: str = "",
    ) -> str:
        """
        Persist experiment record as JSON and update results_table.csv.

        Args:
            config_snapshot:       Full config dict used for this run.
            eval_results:          Output of evaluate_model().
            training_time_seconds: Wall-clock seconds for the entire fit().
            inference_latency_ms:  {"p50": float, "p95": float} in ms.
            model_path:            Absolute path to the saved .keras file.
            backbone_total_layers: Total layer count of backbone (for fraction).
            notes:                 Optional free-text notes.

        Returns:
            Path to the written JSON record file.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        eval_results = eval_results or {}
        config_snapshot = config_snapshot or {}

        # --- Extract metrics -------------------------------------------------
        prim = eval_results.get("primary_metrics") or eval_results
        sec  = eval_results.get("secondary_metrics") or eval_results
        clf  = (prim.get("classification_report") or {}) if isinstance(prim, dict) else {}

        macro_f1    = float(prim.get("macro_f1",        eval_results.get("macro_f1", 0.0))) if isinstance(prim, dict) else 0.0
        bal_acc     = float(prim.get("balanced_accuracy", eval_results.get("balanced_accuracy", 0.0))) if isinstance(prim, dict) else 0.0
        overall_acc = float(sec.get("overall_accuracy", eval_results.get("accuracy", 0.0))) if isinstance(sec, dict) else 0.0
        weighted_f1 = float(prim.get("weighted_f1", 0.0)) if isinstance(prim, dict) else 0.0
        mel_recall  = float((clf.get("mel") or {}).get("recall",   0.0)) if isinstance(clf, dict) else 0.0
        mel_f1      = float((clf.get("mel") or {}).get("f1-score", 0.0)) if isinstance(clf, dict) else 0.0
        df_recall   = float((clf.get("df")  or {}).get("recall",   0.0)) if isinstance(clf, dict) else 0.0
        cal         = eval_results.get("calibration") or {}
        ece         = float(cal.get("expected_calibration_error", 0.0)) if isinstance(cal, dict) else 0.0

        # --- Model file size -------------------------------------------------
        model_size_mb = 0.0
        if model_path and os.path.exists(model_path):
            model_size_mb = round(os.path.getsize(model_path) / (1024 ** 2), 2)

        if not inference_latency_ms:
            inference_latency_ms = {}

        # --- Config helpers --------------------------------------------------
        train_cfg = config_snapshot.get("training", {})
        ft_cfg    = train_cfg.get("fine_tuning", {})
        phases    = ft_cfg.get("phases", [])
        model_cfg = config_snapshot.get("model", {})
        data_cfg  = config_snapshot.get("data", {})
        aug_cfg   = config_snapshot.get("augmentation", {})

        ft_enabled = bool(ft_cfg.get("enabled", False))
        if ft_enabled:
            if phases:
                unfreeze_layers = int(phases[-1].get("unfreeze_layers", 0))
                finetune_epochs = sum(int(p.get("epochs", 0)) for p in phases)
                finetune_lr     = float(phases[-1].get("lr", 1e-4))
            else:
                unfreeze_layers = int(train_cfg.get("unfreeze_layers", 0))
                finetune_epochs = int(train_cfg.get("finetune_epochs", 0))
                finetune_lr     = float(train_cfg.get("finetune_lr", 1e-4))
        else:
            unfreeze_layers = 0
            finetune_epochs = 0
            finetune_lr     = 0.0

        unfreeze_fraction_pct = None
        if backbone_total_layers and backbone_total_layers > 0:
            unfreeze_fraction_pct = round(unfreeze_layers / backbone_total_layers * 100, 1)

        architecture    = model_cfg.get("architecture", "EfficientNetB0")
        input_size_cfg  = model_cfg.get("input_size", [224, 224])
        initial_lr      = float(train_cfg.get("initial_lr", 1e-3))
        frozen_epochs   = int(train_cfg.get("frozen_epochs", 15))
        batch_size      = int(train_cfg.get("batch_size", 32))
        seed_val        = int(data_cfg.get("seed", 42))
        class_weights   = bool(train_cfg.get("class_weights", False))
        label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
        focal_loss_flag = bool(train_cfg.get("focal_loss", False))
        focal_gamma     = float(train_cfg.get("focal_gamma", 2.0)) if focal_loss_flag else 0.0
        lr_scheduler    = train_cfg.get("lr_scheduler", "reduce_lr")

        if focal_loss_flag:
            loss_fn_str = f"focal(gamma={focal_gamma})"
        elif label_smoothing > 0.0:
            loss_fn_str = f"categorical_crossentropy(label_smoothing={label_smoothing})"
        else:
            loss_fn_str = "categorical_crossentropy"

        # --- Assemble JSON record --------------------------------------------
        record: dict[str, Any] = {
            "experiment_id":         self.experiment_id,
            "timestamp_utc":         timestamp,
            "python_version":        platform.python_version(),
            "platform":              platform.platform(),
            "config":                config_snapshot,
            "backbone_total_layers": backbone_total_layers,
            "unfreeze_fraction_pct": unfreeze_fraction_pct,
            "metrics": {
                "val_macro_f1":         macro_f1,
                "val_balanced_acc":     bal_acc,
                "val_overall_acc":      overall_acc,
                "val_weighted_f1":      weighted_f1,
                "mel_recall":           mel_recall,
                "mel_f1":               mel_f1,
                "df_recall":            df_recall,
                "ece":                  ece,
                "vs_e1_macro_f1_delta": round(macro_f1 - _E1_MACRO_F1, 4),
            },
            "resources": {
                "training_time_seconds":    round(training_time_seconds, 1),
                "training_time_min":        round(training_time_seconds / 60, 2),
                "model_size_mb":            model_size_mb,
                "model_path":               model_path,
                "inference_latency_p50_ms": inference_latency_ms.get("p50"),
                "inference_latency_p95_ms": inference_latency_ms.get("p95"),
            },
            "notes":     notes,
            "full_eval": eval_results,
        }

        json_path = os.path.join(self.exp_dir, f"{self.experiment_id}_record.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        logger.info(f"[ExperimentLogger] Wrote JSON record: {json_path}")

        # --- Update CSV ------------------------------------------------------
        csv_path   = os.path.join(self.out_dir, "results_table.csv")
        write_hdr  = True
        needs_leading_newline = False

        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                with open(csv_path, "rb") as f_bin:
                    f_bin.seek(-1, os.SEEK_END)
                    if f_bin.read(1) not in (b"\n", b"\r"):
                        needs_leading_newline = True
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f_txt:
                    first_line = f_txt.readline()
                    if first_line.strip().startswith("experiment_id"):
                        write_hdr = False
            except Exception as e:
                logger.warning(f"[ExperimentLogger] Error checking existing CSV header/newline: {e}")

        lat_p50 = inference_latency_ms.get("p50")
        lat_p95 = inference_latency_ms.get("p95")

        # Sanitize notes so multiline text does not split the single summary row in CSV
        clean_notes = notes.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip() if notes else ""

        row = {
            "experiment_id":           self.experiment_id,
            "timestamp_utc":           timestamp,
            "architecture":            architecture,
            "input_size":              f"{input_size_cfg[0]}x{input_size_cfg[1]}",
            "loss_fn":                 loss_fn_str,
            "label_smoothing":         label_smoothing,
            "focal_gamma":             focal_gamma,
            "unfreeze_layers":         unfreeze_layers,
            "backbone_total_layers":   backbone_total_layers if backbone_total_layers is not None else "",
            "unfreeze_fraction_pct":   unfreeze_fraction_pct if unfreeze_fraction_pct is not None else "",
            "frozen_epochs":           frozen_epochs,
            "finetune_epochs":         finetune_epochs,
            "initial_lr":              initial_lr,
            "finetune_lr":             finetune_lr,
            "lr_scheduler":            lr_scheduler,
            "class_weights":           class_weights,
            "batch_size":              batch_size,
            "seed":                    seed_val,
            "split_method":            data_cfg.get("split_method", "StratifiedGroupKFold"),
            "n_splits":                data_cfg.get("n_splits", 7),
            "test_fold":               data_cfg.get("test_fold", 0),
            "val_fold":                data_cfg.get("val_fold", 1),
            "augmentation_note":       "enabled" if aug_cfg.get("enabled", False) else "disabled",
            "val_macro_f1":            macro_f1,
            "val_balanced_acc":        bal_acc,
            "val_overall_acc":         overall_acc,
            "val_weighted_f1":         weighted_f1,
            "mel_recall":              mel_recall,
            "mel_f1":                  mel_f1,
            "df_recall":               df_recall,
            "ece":                     ece,
            "model_size_mb":           model_size_mb,
            "training_time_min":       round(training_time_seconds / 60, 2),
            "inference_latency_p50_ms": lat_p50 if lat_p50 is not None else "",
            "inference_latency_p95_ms": lat_p95 if lat_p95 is not None else "",
            "vs_e1_macro_f1_delta":    round(macro_f1 - _E1_MACRO_F1, 4),
            "accepted":                "",
            "notes":                   clean_notes,
        }

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            if needs_leading_newline:
                csvfile.write("\n")
            writer = csv.DictWriter(csvfile, fieldnames=_CSV_COLUMNS)
            if write_hdr:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"[ExperimentLogger] Updated results table: {csv_path}")
        return json_path


def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow for reproducibility."""
    import random
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"[ExperimentLogger] Global seed set to {seed}")
