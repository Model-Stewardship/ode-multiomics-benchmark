"""Evaluation metrics for MOTIF and UDE pipeline comparison.

Implements recovery metrics, classification metrics, and robustness testing
for comparing method performance on synthetic inflammation data.
"""

import numpy as np
from typing import Dict, Tuple, List, Callable, Optional
from scipy.stats import spearmanr
import json
from pathlib import Path

# Try to import sklearn, but provide fallbacks
try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        confusion_matrix,
        roc_curve,
        auc,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_recovery_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    variable_name: str = "variable",
) -> Dict[str, float]:
    """
    Compute recovery metrics comparing predicted to ground truth values.

    Metrics include:
    - R²: Coefficient of determination
    - RMSE: Root mean squared error
    - Spearman r: Rank correlation coefficient and p-value

    Args:
        pred: Predicted values (shape: (N,) or (N, T))
        true: Ground truth values (same shape as pred)
        variable_name: Name of variable for reporting

    Returns:
        Dictionary with keys: 'r2', 'rmse', 'spearman_r', 'spearman_pval'
    """
    # Flatten if needed
    pred_flat = np.asarray(pred).flatten()
    true_flat = np.asarray(true).flatten()

    # Remove NaN/inf values
    valid_idx = np.isfinite(pred_flat) & np.isfinite(true_flat)
    pred_valid = pred_flat[valid_idx]
    true_valid = true_flat[valid_idx]

    if len(pred_valid) == 0:
        return {
            'r2': np.nan,
            'rmse': np.nan,
            'spearman_r': np.nan,
            'spearman_pval': np.nan,
        }

    # R² = 1 - (SS_res / SS_tot)
    ss_res = np.sum((true_valid - pred_valid) ** 2)
    ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    # RMSE
    rmse = np.sqrt(np.mean((true_valid - pred_valid) ** 2))

    # Spearman correlation
    spearman_r, spearman_pval = spearmanr(true_valid, pred_valid)

    return {
        'r2': float(r2),
        'rmse': float(rmse),
        'spearman_r': float(spearman_r),
        'spearman_pval': float(spearman_pval),
        'n_samples': len(pred_valid),
        'variable': variable_name,
    }


def compute_classification_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute classification metrics for outcome prediction.

    Supports binary and multi-class classification.

    Args:
        y_pred: Predicted class labels or probabilities
        y_true: Ground truth class labels
        class_names: Names of classes (for reporting)

    Returns:
        Dictionary with AUROC, F1, confusion matrix, and per-class metrics
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # Determine number of classes
    n_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f'class_{i}' for i in range(n_classes)]

    results = {
        'n_samples': len(y_true),
        'n_classes': n_classes,
        'class_names': class_names,
    }

    if not HAS_SKLEARN:
        # Provide basic metrics without sklearn
        # Compute accuracy
        accuracy = np.mean(y_pred == y_true)
        results['accuracy'] = float(accuracy)
        return results

    # Handle multi-class
    if n_classes > 2:
        # Macro-averaged metrics
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_macro'] = float(f1_macro)

        # AUROC (one-vs-rest for multi-class)
        try:
            if y_pred.ndim > 1:  # Probabilities
                auroc_macro = roc_auc_score(
                    y_true, y_pred, multi_class='ovr', average='macro', zero_division=0
                )
            else:  # Labels only
                auroc_macro = np.nan
            results['auroc_macro'] = float(auroc_macro)
        except Exception as e:
            results['auroc_macro'] = np.nan

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        results['confusion_matrix'] = cm.tolist()

        # Per-class metrics
        for i, name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)

            f1_binary = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            results[f'f1_{name}'] = float(f1_binary)

    else:
        # Binary classification
        f1_binary = f1_score(y_true, y_pred, zero_division=0)
        results['f1'] = float(f1_binary)

        # AUROC
        try:
            auroc = roc_auc_score(y_true, y_pred)
            results['auroc'] = float(auroc)
        except Exception:
            results['auroc'] = np.nan

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()

    # Add accuracy (works for both binary and multi-class)
    from sklearn.metrics import accuracy_score
    results['accuracy'] = float(accuracy_score(y_true, y_pred))

    return results


def compare_pipelines(
    motif_results: Dict,
    ude_results: Dict,
    variable_names: List[str] = None,
) -> Dict:
    """
    Create head-to-head comparison of MOTIF and UDE pipelines.

    Args:
        motif_results: Results dict from MOTIF pipeline with recovery metrics
        ude_results: Results dict from UDE pipeline
        variable_names: Variables to compare (default: ['P', 'D', 'h'])

    Returns:
        Comparison dict with side-by-side metrics
    """
    if variable_names is None:
        variable_names = ['P', 'D', 'h']

    comparison = {
        'timestamp': str(np.datetime64('now')),
        'pipelines': {
            'motif': {},
            'ude': {},
        },
        'comparison': {},
    }

    # Extract MOTIF metrics
    if 'recovery_metrics' in motif_results:
        for var in variable_names:
            key = f'proxy_{var.lower()}'
            if key in motif_results['recovery_metrics']:
                comparison['pipelines']['motif'][var] = motif_results['recovery_metrics'][key]

    if 'classification_metrics' in motif_results:
        comparison['pipelines']['motif']['classification'] = motif_results['classification_metrics']

    # Extract UDE metrics
    if 'recovery_metrics' in ude_results:
        for var in variable_names:
            if var in ude_results['recovery_metrics']:
                comparison['pipelines']['ude'][var] = ude_results['recovery_metrics'][var]

    if 'classification_metrics' in ude_results:
        comparison['pipelines']['ude']['classification'] = ude_results['classification_metrics']

    # Side-by-side comparison for shared metrics
    comparison['comparison'] = {
        'recovery': {},
        'classification': {},
    }

    for var in variable_names:
        motif_r2 = comparison['pipelines']['motif'].get(var, {}).get('r2', np.nan)
        ude_r2 = comparison['pipelines']['ude'].get(var, {}).get('r2', np.nan)

        comparison['comparison']['recovery'][var] = {
            'motif_r2': motif_r2,
            'ude_r2': ude_r2,
            'motif_better': motif_r2 > ude_r2 if np.isfinite(motif_r2) and np.isfinite(ude_r2) else None,
        }

    # Classification comparison
    motif_auroc = (
        comparison['pipelines']['motif'].get('classification', {}).get('auroc_macro')
        or comparison['pipelines']['motif'].get('classification', {}).get('auroc')
    )
    ude_auroc = (
        comparison['pipelines']['ude'].get('classification', {}).get('auroc_macro')
        or comparison['pipelines']['ude'].get('classification', {}).get('auroc')
    )

    if np.isfinite(motif_auroc) and np.isfinite(ude_auroc):
        comparison['comparison']['classification']['auroc'] = {
            'motif': motif_auroc,
            'ude': ude_auroc,
            'motif_better': motif_auroc > ude_auroc,
        }

    return comparison


def run_robustness_experiment(
    experiment_fn: Callable,
    param_name: str,
    param_values: List,
    n_replicates: int = 5,
    seed_base: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run robustness experiment varying one parameter.

    Args:
        experiment_fn: Function that takes (param_value, seed) and returns results
        param_name: Name of parameter being varied
        param_values: List of parameter values to test
        n_replicates: Number of random replicates per parameter value
        seed_base: Base random seed
        verbose: Print progress messages

    Returns:
        Dictionary with results aggregated by parameter value
    """
    results = {
        'param_name': param_name,
        'param_values': param_values,
        'n_replicates': n_replicates,
        'results_by_param': {},
    }

    for param_idx, param_val in enumerate(param_values):
        if verbose:
            print(f"Testing {param_name}={param_val}...")

        param_results = {
            'param_value': param_val,
            'replicates': [],
        }

        for rep_idx in range(n_replicates):
            seed = seed_base + param_idx * 1000 + rep_idx
            if verbose:
                print(f"  Replicate {rep_idx + 1}/{n_replicates} (seed={seed})")

            try:
                rep_result = experiment_fn(param_val, seed)
                param_results['replicates'].append(rep_result)
            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                param_results['replicates'].append({'error': str(e)})

        # Aggregate results
        param_results['summary'] = aggregate_replicate_results(param_results['replicates'])
        results['results_by_param'][str(param_val)] = param_results

    return results


def aggregate_replicate_results(replicates: List[Dict]) -> Dict:
    """
    Aggregate results across replicates (compute mean ± SD).

    Args:
        replicates: List of result dicts from individual replicates

    Returns:
        Summary dict with mean and std for each metric
    """
    # Filter out error results
    valid_replicates = [r for r in replicates if 'error' not in r]

    if not valid_replicates:
        return {'error': 'No valid replicates'}

    summary = {}

    # Collect all metric keys
    all_keys = set()
    for rep in valid_replicates:
        all_keys.update(flatten_dict_keys(rep))

    # Aggregate numeric metrics
    for key in all_keys:
        values = []
        for rep in valid_replicates:
            val = get_nested_value(rep, key)
            if val is not None and np.isfinite(val):
                values.append(val)

        if values:
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))
            summary[f'{key}_n'] = len(values)

    return summary


def flatten_dict_keys(d: Dict, prefix: str = '') -> set:
    """Recursively flatten dict keys for aggregation."""
    keys = set()
    for k, v in d.items():
        full_key = f'{prefix}_{k}' if prefix else k
        if isinstance(v, dict):
            keys.update(flatten_dict_keys(v, full_key))
        elif isinstance(v, (int, float, np.number)):
            keys.add(full_key)
    return keys


def get_nested_value(d: Dict, key: str):
    """Get value from nested dict using dot-separated key."""
    parts = key.split('_')
    current = d
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current if isinstance(current, (int, float, np.number)) else None


def save_results(results: Dict, output_path: str):
    """Save results dict to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python for JSON serialization
    results_serializable = convert_numpy_to_native(results)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)


def convert_numpy_to_native(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj


if __name__ == '__main__':
    print("Testing evaluation metrics...")

    # Test recovery metrics
    true_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred_vals = np.array([1.1, 2.1, 2.9, 4.0, 5.1])

    recovery_metrics = compute_recovery_metrics(pred_vals, true_vals, 'test_var')
    print("\nRecovery Metrics Example:")
    for key, val in recovery_metrics.items():
        print(f"  {key}: {val}")

    # Test classification metrics
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 0, 2])

    clf_metrics = compute_classification_metrics(y_pred, y_true, ['resolution', 'chronic', 'death'])
    print("\nClassification Metrics Example:")
    for key, val in clf_metrics.items():
        if key not in ['confusion_matrix', 'class_names']:
            print(f"  {key}: {val}")

    # Test robustness experiment
    def dummy_experiment(param_val, seed):
        np.random.seed(seed)
        return {
            'param_value': param_val,
            'metric1': np.random.uniform(0.5, 0.95),
            'metric2': np.random.uniform(0.1, 0.3),
        }

    robustness_results = run_robustness_experiment(
        dummy_experiment,
        'noise_level',
        [0.05, 0.10, 0.20],
        n_replicates=3,
        verbose=False,
    )

    print("\nRobustness Experiment Example:")
    for param_val, res in robustness_results['results_by_param'].items():
        print(f"  {param_val}: {res['summary']}")

    print("\nStep 4: Evaluation metrics implementation complete.")
