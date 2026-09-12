import numpy as np
import json
import os
import time
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    accuracy_score,
    confusion_matrix
)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    """
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = y_pred == y_true

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # for the first bin include 0
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)


def measure_inference_latency(model, input_size=(224, 224), n_warmup: int = 5, n_runs: int = 30) -> dict:
    """
    Measures single-image inference latency on a dummy input.
    Returns {"p50": float, "p95": float} in milliseconds.
    """
    dummy = np.random.rand(1, input_size[0], input_size[1], 3).astype(np.float32) * 255.0
    for _ in range(n_warmup):
        model.predict(dummy, verbose=0)
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50": round(float(np.percentile(latencies, 50)), 2),
        "p95": round(float(np.percentile(latencies, 95)), 2),
    }


def append_experiment_record(
    experiment_id: str,
    config_snapshot: dict,
    eval_results: dict,
    training_time_seconds: float,
    inference_latency_ms: dict,
    model_path: str,
    backbone_total_layers: int = None,
    notes: str = "",
    out_dir: str = "experiments",
) -> str:
    """
    Hook to log experiment results to JSON and append to experiments/results_table.csv
    without duplication across scripts.
    """
    try:
        from training.experiment_logger import ExperimentLogger
    except ImportError:
        from experiment_logger import ExperimentLogger

    logger = ExperimentLogger(experiment_id, out_dir=out_dir)
    return logger.log(
        config_snapshot=config_snapshot,
        eval_results=eval_results,
        training_time_seconds=training_time_seconds,
        inference_latency_ms=inference_latency_ms,
        model_path=model_path,
        backbone_total_layers=backbone_total_layers,
        notes=notes,
    )


def evaluate_model(
    true_labels,
    predicted_probs,
    class_names,
    output_path=None,
    experiment_id=None,
    config=None,
    training_time_seconds=0.0,
    inference_latency_ms=None,
    model_path=None,
    backbone_total_layers=None,
    notes="",
    out_dir="experiments",
):
    """
    Evaluates model predictions and computes metrics.
    
    Args:
        true_labels: numpy array of true integer labels
        predicted_probs: numpy array of predicted probabilities of shape (N, num_classes)
        class_names: list of class names
        output_path: Optional path to save evaluation results as JSON
        experiment_id: Optional experiment identifier to trigger logging
        config: Optional config dictionary for experiment logger hook
        training_time_seconds: Wall-clock training duration
        inference_latency_ms: Dict with p50 and p95 latencies
        model_path: Path to model checkpoint file
        backbone_total_layers: Total layer count in backbone
        notes: Free-text notes for experiment record
        out_dir: Root directory for experiments artifacts
        
    Returns:
        Dictionary of computed metrics
    """
    y_pred = np.argmax(predicted_probs, axis=1)
    labels = list(range(len(class_names)))
    
    # Primary metrics
    bal_acc = balanced_accuracy_score(true_labels, y_pred)
    macro_f1 = f1_score(true_labels, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, y_pred, average='weighted', zero_division=0)
    
    class_report = classification_report(
        true_labels, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    
    # Secondary metrics
    acc = accuracy_score(true_labels, y_pred)
    conf_matrix_abs = confusion_matrix(true_labels, y_pred, labels=labels).tolist()
    conf_matrix_norm = confusion_matrix(true_labels, y_pred, labels=labels, normalize='true').tolist()
    
    # Calibration analysis (ECE)
    ece = expected_calibration_error(true_labels, predicted_probs)
    
    results = {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "accuracy": float(acc),
        "primary_metrics": {
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "classification_report": class_report
        },
        "secondary_metrics": {
            "overall_accuracy": float(acc),
            "confusion_matrix_absolute": conf_matrix_abs,
            "confusion_matrix_normalized": conf_matrix_norm
        },
        "calibration": {
            "expected_calibration_error": ece
        }
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    if experiment_id and config is not None:
        append_experiment_record(
            experiment_id=experiment_id,
            config_snapshot=config,
            eval_results=results,
            training_time_seconds=training_time_seconds,
            inference_latency_ms=inference_latency_ms or {},
            model_path=model_path or "",
            backbone_total_layers=backbone_total_layers,
            notes=notes,
            out_dir=out_dir,
        )

    return results
