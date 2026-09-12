"""
Postprocessing module for SkinVision.
Handles uncertainty estimation and prediction formatting.
"""

import numpy as np

try:
    from config import CLASS_KEYS, CLASS_INFO, NUM_CLASSES
    from src.inference import get_top_k
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import CLASS_KEYS, CLASS_INFO, NUM_CLASSES
    from src.inference import get_top_k


def compute_uncertainty(probabilities: np.ndarray) -> dict:
    """
    Compute uncertainty metrics from prediction probabilities.
    
    Args:
        probabilities: 1D array of shape (NUM_CLASSES,)
        
    Returns:
        Dictionary containing entropy, max_entropy, normalized_entropy, 
        margin (top-1 minus top-2), and max_probability.
    """
    eps = 1e-7
    # Entropy: -sum(p * log(p + eps))
    entropy = -np.sum(probabilities * np.log(probabilities + eps))
    max_entropy = np.log(NUM_CLASSES)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Margin: difference between top-1 and top-2 probabilities
    sorted_probs = np.sort(probabilities)[::-1]
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    
    max_probability = sorted_probs[0]
    
    return {
        'entropy': float(entropy),
        'max_entropy': float(max_entropy),
        'normalized_entropy': float(normalized_entropy),
        'margin': float(margin),
        'max_probability': float(max_probability)
    }


def should_abstain(probabilities: np.ndarray, calibration_threshold: float = None) -> tuple[bool, str]:
    """
    Determine if the model should abstain from making a prediction due to high uncertainty.
    
    Note: The calibration_threshold should ideally be derived from validation 
    calibration analysis, rather than relying on hardcoded defaults.
    
    Args:
        probabilities: 1D array of shape (NUM_CLASSES,)
        calibration_threshold: Optional threshold derived from evaluation results.
        
    Returns:
        Tuple of (should_abstain: bool, reason: str)
    """
    uncertainty = compute_uncertainty(probabilities)
    
    if calibration_threshold is not None:
        # If we have a calibrated threshold for uncertainty/entropy
        if uncertainty['normalized_entropy'] > calibration_threshold:
            return True, f"Normalized entropy ({uncertainty['normalized_entropy']:.3f}) exceeds calibrated threshold ({calibration_threshold:.3f})"
        return False, ""
    
    # Fallback to conservative uncalibrated defaults
    warning_prefix = "[WARNING: Using uncalibrated default thresholds] "
    
    if uncertainty['normalized_entropy'] > 0.7:
        return True, f"{warning_prefix}High uncertainty: normalized entropy ({uncertainty['normalized_entropy']:.3f}) > 0.7"
        
    if uncertainty['max_probability'] < 0.3:
        return True, f"{warning_prefix}Low confidence: max probability ({uncertainty['max_probability']:.3f}) < 0.3"
        
    return False, ""


def format_prediction_result(probabilities: np.ndarray) -> dict:
    """
    Format prediction probabilities into a structured result dictionary.
    
    Args:
        probabilities: 1D array of shape (NUM_CLASSES,)
        
    Returns:
        Dictionary containing top_prediction, top_k, uncertainty, 
        abstain_recommended, and abstain_reason.
    """
    top_k = get_top_k(probabilities, k=3)
    uncertainty = compute_uncertainty(probabilities)
    abstain_recommended, abstain_reason = should_abstain(probabilities)
    
    top_prediction = top_k[0]
    
    return {
        'top_prediction': top_prediction,
        'top_k': top_k,
        'uncertainty': uncertainty,
        'abstain_recommended': abstain_recommended,
        'abstain_reason': abstain_reason
    }
