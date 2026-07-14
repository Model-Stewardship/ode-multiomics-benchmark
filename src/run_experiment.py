"""Main experiment orchestration runner."""

import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import numpy as np

from src.experiment_utils import ExperimentConfig
from src.synthetic_data import generate_cohort
from src.motif_pipeline import run_motif_pipeline
from src.ude_pipeline import run_ude_pipeline


def run_single_replicate(config: ExperimentConfig, replicate_id: int,
                        verbose: bool = True) -> Tuple[Dict, Dict, Dict]:
    """
    Run a single replicate of the experiment.

    Args:
        config: ExperimentConfig object
        replicate_id: Replicate number (0-indexed)
        verbose: Print progress

    Returns:
        Tuple of (cohort, motif_results, ude_results)
    """
    if verbose:
        print(f"\n  Replicate {replicate_id + 1}/{config.n_replicates}...")

    # Generate synthetic cohort
    seed = config.random_seed + replicate_id
    base = config.n_patients // 3
    remainder = config.n_patients % 3
    cohort = generate_cohort(
        N_resolution=base + (1 if remainder > 0 else 0),
        N_chronic=base + (1 if remainder > 1 else 0),
        N_death=base,
        seed=seed,
        verbose=False
    )

    # Run MOTIF pipeline
    motif_config = {
        'n_restarts': 3,
    }
    motif_results = run_motif_pipeline(cohort, config=motif_config, verbose=verbose)

    # Run UDE pipeline
    ude_config = {
        'nn_hidden_dim': config.ude.hidden_dim,
        'n_epochs': config.ude.n_epochs,
        'lr': config.ude.lr,
        'batch_size': config.ude.batch_size,
        'max_grad_norm': config.ude.max_grad_norm,
        'random_seed': seed,
        'sindy_degree': config.sindy.degree,
        'sindy_threshold': config.sindy.threshold,
    }
    ude_results = run_ude_pipeline(cohort, config=ude_config, verbose=verbose)

    return cohort, motif_results, ude_results


def aggregate_results(replicates: list) -> Dict:
    """
    Aggregate results across replicates.

    Args:
        replicates: List of (cohort, motif_results, ude_results) tuples

    Returns:
        Aggregated metrics dict
    """
    metrics = {
        'n_replicates': len(replicates),
        'motif_recovery': {},
        'ude_recovery': {},
        'motif_classification': {},
        'ude_classification': {},
    }

    # Aggregate recovery metrics if available
    for var in ['P', 'D', 'h']:
        motif_r2_values = []
        ude_r2_values = []

        for cohort, motif_res, ude_res in replicates:
            if 'recovery_metrics' in motif_res and var in motif_res['recovery_metrics']:
                r2 = motif_res['recovery_metrics'][var].get('r2', None)
                if r2 is not None:
                    motif_r2_values.append(r2)

            if 'recovery_metrics' in ude_res and var in ude_res['recovery_metrics']:
                r2 = ude_res['recovery_metrics'][var].get('r2', None)
                if r2 is not None:
                    ude_r2_values.append(r2)

        if motif_r2_values:
            metrics['motif_recovery'][var] = {
                'mean_r2': float(np.mean(motif_r2_values)),
                'std_r2': float(np.std(motif_r2_values)),
            }

        if ude_r2_values:
            metrics['ude_recovery'][var] = {
                'mean_r2': float(np.mean(ude_r2_values)),
                'std_r2': float(np.std(ude_r2_values)),
            }

    # Aggregate classification metrics
    for key in ['with_proxies', 'without_proxies']:
        motif_aurocs = []
        ude_aurocs = []

        for cohort, motif_res, ude_res in replicates:
            if 'classification_results' in motif_res and key in motif_res['classification_results']:
                clf = motif_res['classification_results'][key]
                auroc = clf.get('auroc', clf.get('accuracy', None))
                if auroc is not None:
                    motif_aurocs.append(auroc)

            if 'classification_results' in ude_res and key in ude_res['classification_results']:
                clf = ude_res['classification_results'][key]
                auroc = clf.get('auroc', clf.get('accuracy', None))
                if auroc is not None:
                    ude_aurocs.append(auroc)

        if motif_aurocs:
            metrics['motif_classification'][key] = {
                'mean_auroc': float(np.mean(motif_aurocs)),
                'std_auroc': float(np.std(motif_aurocs)),
            }

        if ude_aurocs:
            metrics['ude_classification'][key] = {
                'mean_auroc': float(np.mean(ude_aurocs)),
                'std_auroc': float(np.std(ude_aurocs)),
            }

    return metrics


def run_experiment(config: ExperimentConfig, verbose: bool = True) -> str:
    """
    Run complete experiment with all replicates.

    Args:
        config: ExperimentConfig object
        verbose: Print progress

    Returns:
        Path to output directory
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.output_dir) / f"{config.experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Running experiment: {config.experiment_name}")
        print(f"Output directory: {output_dir}")

    # Run replicates
    replicates = []
    for rep_id in range(config.n_replicates):
        cohort, motif_res, ude_res = run_single_replicate(config, rep_id, verbose=verbose)
        replicates.append((cohort, motif_res, ude_res))

    # Save individual results
    for rep_id, (cohort, motif_res, ude_res) in enumerate(replicates):
        rep_dir = output_dir / f"replicate_{rep_id}"
        rep_dir.mkdir(exist_ok=True)

        with open(rep_dir / 'cohort.pkl', 'wb') as f:
            pickle.dump(cohort, f)

        with open(rep_dir / 'motif_results.pkl', 'wb') as f:
            pickle.dump(motif_res, f)

        with open(rep_dir / 'ude_results.pkl', 'wb') as f:
            pickle.dump(ude_res, f)

    # Save aggregated metrics
    metrics = {
        'config': config.to_dict(),
        'timestamp': timestamp,
    }
    metrics.update(aggregate_results(replicates))

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"Experiment complete. Results saved to {output_dir}")

    return str(output_dir)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Run ODE-Multiomics benchmark experiment')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--verbose', action='store_true', help='Print progress messages')

    args = parser.parse_args()

    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    # Run experiment
    try:
        output_dir = run_experiment(config, verbose=args.verbose)
        print(f"Experiment complete: {output_dir}")
        return 0
    except Exception as e:
        print(f"Experiment failed: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
