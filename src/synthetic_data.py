"""Synthetic patient cohort generation from Reynolds ODE model.

Generates virtual patients with parameter heterogeneity, measurement noise,
and stratified outcome distribution across three inflammation phenotypes.
"""

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Handle imports from package or direct script execution
try:
    from src.reynolds_ode import REYNOLDS_PARAMS, solve_reynolds, get_outcome
except ImportError:
    from reynolds_ode import REYNOLDS_PARAMS, solve_reynolds, get_outcome


# Cohort stratification: initial pathogen load ranges and counts
COHORT_SPEC = {
    'resolution': {'N': 200, 'P0_range': (0.5, 3.5)},
    'chronic':    {'N': 150, 'P0_range': (4.0, 9.0)},
    'death':      {'N': 150, 'P0_range': (11.0, 20.0)},
}

# Observation timepoints (hours)
OBS_TIMEPOINTS = np.array([0, 6, 12, 24, 48, 72], dtype=float)

# Observation variables (P, D, h are latent)
OBS_VARIABLES = ['Nstar', 'CA', 'f']


def generate_patient(
    P0: float,
    params_base: Dict = None,
    seed: int = None,
    obs_noise_sigma: float = 0.10,
    param_noise_sigma: float = 0.15,
) -> Dict:
    """
    Generate a single synthetic patient.

    Args:
        P0: Initial pathogen load
        params_base: Baseline parameters (uses REYNOLDS_PARAMS if None)
        seed: Random seed for reproducibility
        obs_noise_sigma: Observation noise coefficient of variation
        param_noise_sigma: Parameter noise log-SD for lognormal perturbation

    Returns:
        Patient dict with observed, latent, and ground truth data
    """
    if seed is not None:
        np.random.seed(seed)

    if params_base is None:
        params_base = REYNOLDS_PARAMS.copy()

    # Apply per-patient parameter perturbation (±15% lognormal)
    patient_params = {}
    for key, val in params_base.items():
        log_noise = np.random.normal(0, param_noise_sigma)
        patient_params[key] = val * np.exp(log_noise)

    # Initial conditions
    x0 = np.array([P0, 0.0, 0.0, 0.0, 0.0])

    # Solve ODE at high resolution for ground truth
    t_fine = np.linspace(0, 72, 1000)
    try:
        sol_fine = solve_reynolds(patient_params, x0, t_fine, method='Radau')
    except ValueError as e:
        raise ValueError(f"ODE integration failed for P0={P0}: {e}")

    # Determine outcome from ground truth
    outcome = get_outcome(sol_fine['f'][-1])

    # Extract values at observation timepoints
    t_obs = OBS_TIMEPOINTS.copy()
    sol_obs = solve_reynolds(patient_params, x0, t_obs, method='Radau')

    # Add measurement noise to observations
    # Proportional Gaussian noise: observed = true * (1 + N(0, sigma))
    obs_Nstar = sol_obs['Nstar'] * (1.0 + np.random.normal(0, obs_noise_sigma, len(t_obs)))
    obs_CA = sol_obs['CA'] * (1.0 + np.random.normal(0, obs_noise_sigma, len(t_obs)))
    obs_f = sol_obs['f'] * (1.0 + np.random.normal(0, obs_noise_sigma, len(t_obs)))

    # Clip to non-negative and valid range
    obs_Nstar = np.clip(obs_Nstar, 0, None)
    obs_CA = np.clip(obs_CA, 0, None)
    obs_f = np.clip(obs_f, 0, 1.0)

    # Extract latent ground truth at observation timepoints
    true_P = sol_obs['P']
    true_D = sol_obs['D']
    true_h = sol_obs['h']

    return {
        'id': None,  # Will be set by generate_cohort
        'P0': P0,
        'outcome': outcome,
        'params': patient_params,
        # Observed (noisy, 6 timepoints)
        'obs_Nstar': obs_Nstar,
        'obs_CA': obs_CA,
        'obs_f': obs_f,
        'obs_t': t_obs,
        # Ground truth at observation timepoints
        'true_P': true_P,
        'true_D': true_D,
        'true_h': true_h,
        # Full high-resolution ground truth trajectory
        'true_traj': sol_fine.copy(),  # Full solution dict
        'true_t': sol_fine['t'],
    }


def generate_cohort(
    N_resolution: int = 200,
    N_chronic: int = 150,
    N_death: int = 150,
    seed: int = 42,
    obs_noise_sigma: float = 0.10,
    param_noise_sigma: float = 0.15,
    verbose: bool = True,
) -> List[Dict]:
    """
    Generate a stratified cohort of synthetic patients.

    Args:
        N_resolution: Number of resolution-outcome patients
        N_chronic: Number of chronic-outcome patients
        N_death: Number of death-outcome patients
        seed: Random seed for reproducibility
        obs_noise_sigma: Observation noise coefficient of variation
        param_noise_sigma: Parameter noise log-SD
        verbose: Print progress messages

    Returns:
        List of patient dicts, stratified by outcome
    """
    np.random.seed(seed)

    cohort = []
    patient_id = 0

    outcome_specs = [
        ('resolution', N_resolution, COHORT_SPEC['resolution']['P0_range']),
        ('chronic', N_chronic, COHORT_SPEC['chronic']['P0_range']),
        ('death', N_death, COHORT_SPEC['death']['P0_range']),
    ]

    for outcome_name, N, (P0_min, P0_max) in outcome_specs:
        if verbose:
            print(f"Generating {N} {outcome_name} patients (P0 ~ U({P0_min}, {P0_max}))...")

        iterator = tqdm(range(N), disable=not verbose)
        for i in iterator:
            # Sample P0 uniformly from outcome-specific range
            P0 = np.random.uniform(P0_min, P0_max)

            # Generate patient with unique seed
            patient_seed = seed + patient_id
            try:
                patient = generate_patient(
                    P0,
                    params_base=REYNOLDS_PARAMS.copy(),
                    seed=patient_seed,
                    obs_noise_sigma=obs_noise_sigma,
                    param_noise_sigma=param_noise_sigma,
                )
                patient['id'] = patient_id
                cohort.append(patient)
                patient_id += 1
            except ValueError as e:
                if verbose:
                    print(f"  Warning: Failed to generate patient {patient_id}: {e}")
                continue

    if verbose:
        print(f"\nSuccessfully generated {len(cohort)} patients")
        print(f"Outcome distribution:")
        outcomes = [p['outcome'] for p in cohort]
        for outcome in ['resolution', 'chronic', 'death']:
            count = sum(1 for o in outcomes if o == outcome)
            pct = 100 * count / len(cohort)
            print(f"  {outcome:12s}: {count:3d} ({pct:5.1f}%)")

    return cohort


def save_cohort(
    cohort: List[Dict],
    output_dir: str = 'results',
    name: str = 'synthetic_cohort'
) -> Tuple[str, str]:
    """
    Save cohort to pickle and CSV summary.

    Args:
        cohort: List of patient dicts
        output_dir: Output directory
        name: Base name for output files

    Returns:
        Tuple of (pickle_path, csv_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full data as pickle
    pickle_path = output_dir / f'{name}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(cohort, f)

    # Save summary as CSV
    csv_path = output_dir / f'{name}_summary.csv'
    summary_rows = []
    for patient in cohort:
        summary_rows.append({
            'patient_id': patient['id'],
            'P0': patient['P0'],
            'outcome': patient['outcome'],
            'obs_Nstar_mean': np.mean(patient['obs_Nstar']),
            'obs_CA_mean': np.mean(patient['obs_CA']),
            'obs_f_mean': np.mean(patient['obs_f']),
            'obs_f_final': patient['obs_f'][-1],
            'true_P_final': patient['true_P'][-1],
            'true_D_final': patient['true_D'][-1],
            'true_h_final': patient['true_h'][-1],
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(csv_path, index=False)

    return str(pickle_path), str(csv_path)


def load_cohort(cohort_path: str) -> List[Dict]:
    """
    Load cohort from pickle file.

    Args:
        cohort_path: Path to pickle file

    Returns:
        List of patient dicts
    """
    with open(cohort_path, 'rb') as f:
        cohort = pickle.load(f)
    return cohort


def plot_example_trajectories(
    cohort: List[Dict],
    ax: Optional[plt.Axes] = None,
    title: str = 'Example Patient Trajectories'
) -> plt.Axes:
    """
    Plot one example patient from each outcome category.

    Args:
        cohort: Patient cohort list
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    colors_outcome = {
        'resolution': '#2ECC71',
        'chronic': '#F39C12',
        'death': '#E74C3C',
    }

    # Find one example from each outcome
    for outcome in ['resolution', 'chronic', 'death']:
        patient = next((p for p in cohort if p['outcome'] == outcome), None)
        if patient is None:
            continue

        sol = patient['true_traj']
        color = colors_outcome[outcome]

        # Plot observed variables as points
        ax.plot(patient['obs_t'], patient['obs_Nstar'], 'o', color=color,
                markersize=5, alpha=0.6, label=f'N* obs ({outcome})')
        ax.plot(patient['obs_t'], patient['obs_f'], 's', color=color,
                markersize=5, alpha=0.6, label=f'f obs ({outcome})')

        # Plot ground truth as lines
        ax.plot(sol['t'], sol['Nstar'], '-', color=color, linewidth=2, alpha=0.8)
        ax.plot(sol['t'], sol['f'], '--', color=color, linewidth=2, alpha=0.8)

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('State Variable Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    return ax


def plot_p0_distribution(
    cohort: List[Dict],
    ax: Optional[plt.Axes] = None,
    title: str = 'Initial Pathogen Load Distribution'
) -> plt.Axes:
    """
    Plot distribution of P0 values by outcome.

    Args:
        cohort: Patient cohort list
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors_outcome = {
        'resolution': '#2ECC71',
        'chronic': '#F39C12',
        'death': '#E74C3C',
    }

    for outcome in ['resolution', 'chronic', 'death']:
        P0_values = [p['P0'] for p in cohort if p['outcome'] == outcome]
        ax.hist(P0_values, bins=20, alpha=0.6, label=outcome,
                color=colors_outcome[outcome], edgecolor='black')

    ax.set_xlabel('Initial Pathogen Load (P₀)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    return ax


if __name__ == '__main__':
    print("Generating synthetic patient cohort...")
    print(f"Total cohort size: {sum(v['N'] for v in COHORT_SPEC.values())}\n")

    cohort = generate_cohort(
        N_resolution=COHORT_SPEC['resolution']['N'],
        N_chronic=COHORT_SPEC['chronic']['N'],
        N_death=COHORT_SPEC['death']['N'],
        seed=42,
        verbose=True
    )

    # Save cohort
    print("\nSaving cohort...")
    pkl_path, csv_path = save_cohort(cohort, output_dir='results', name='synthetic_cohort_baseline')
    print(f"  Pickle: {pkl_path}")
    print(f"  Summary CSV: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_p0_distribution(cohort, axes[0])
    plot_example_trajectories(cohort, axes[1])

    plt.tight_layout()
    plt.savefig('synthetic_data_test.png', dpi=150, bbox_inches='tight')
    print("  Saved: synthetic_data_test.png")

    print("\nStep 3: Synthetic patient generation complete.")
