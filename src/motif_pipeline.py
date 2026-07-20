"""MOTIF Pipeline: ODE-based proxy expansion and correlation analysis.

Reference: Funk, Bangs, Paterson (2025-26) MOTIF approach.
Forward ODE simulation → synthetic state variable proxies → correlation analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from scipy.stats import spearmanr
from scipy.integrate import trapezoid
import warnings
from tqdm import tqdm
import time

# Try sklearn imports with fallback
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from src.reynolds_ode import REYNOLDS_PARAMS, solve_reynolds, get_outcome
    from src.evaluation import compute_recovery_metrics, compute_classification_metrics
except ImportError:
    from reynolds_ode import REYNOLDS_PARAMS, solve_reynolds, get_outcome
    from evaluation import compute_recovery_metrics, compute_classification_metrics

warnings.filterwarnings('ignore')


# Parameters to optimize during calibration
# Removed mu_p: pathogen decay is implicit in logistic growth (paper doesn't have explicit mu_p term)
CALIBRATION_PARAMS = ['k_pg', 's_nr', 's_c']

# Parameters to hold fixed
FIXED_PARAMS = set(REYNOLDS_PARAMS.keys()) - set(CALIBRATION_PARAMS)


def calibrate_patient(
    patient: Dict,
    params_init: Optional[Dict] = None,
    n_restarts: int = 3,
    obs_timepoints: np.ndarray = None,
    verbose: bool = False,
) -> Tuple[Dict, float, float]:
    """
    Calibrate ODE parameters to a single patient's observed data.

    Optimizes a subset of parameters (k_pg, s_nr, s_c) + P0
    using L-BFGS-B to fit observed N*, CA, f trajectories.

    Args:
        patient: Patient dict with obs_Nstar, obs_CA, obs_f, obs_t
        params_init: Initial parameters (uses REYNOLDS_PARAMS if None)
        n_restarts: Number of optimization restarts
        obs_timepoints: Observation timepoints (from patient if None)
        verbose: Print optimization messages

    Returns:
        Tuple of (fitted_params, fitted_P0, fit_quality_R2)
    """
    if params_init is None:
        params_init = REYNOLDS_PARAMS.copy()

    if obs_timepoints is None:
        obs_timepoints = patient['obs_t']

    # Observed data
    obs_Nstar = patient['obs_Nstar']
    obs_CA = patient['obs_CA']
    obs_f = patient['obs_f']

    best_result = None
    best_residual = np.inf

    for restart in range(n_restarts):
        # Initialize parameters with some randomness
        if restart > 0:
            log_scale = 0.1 * np.random.randn(len(CALIBRATION_PARAMS))
            x0 = np.concatenate([
                np.log(np.array([params_init[p] for p in CALIBRATION_PARAMS]) * np.exp(log_scale)),
                np.array([np.log(patient['P0'])]),  # Also optimize P0
            ])
        else:
            x0 = np.concatenate([
                np.log(np.array([params_init[p] for p in CALIBRATION_PARAMS])),
                np.array([np.log(patient['P0'])]),
            ])

        def residual_fn(log_params):
            """Objective function: sum of squared residuals."""
            try:
                # Decode parameters
                calib_params = dict(zip(CALIBRATION_PARAMS, np.exp(log_params[:-1])))
                P0_fit = np.exp(log_params[-1])

                # Merge with fixed parameters
                params_fit = {**params_init}
                params_fit.update(calib_params)

                # Simulate with fast tolerances (optimization loop doesn't need tight precision)
                # Use LSODA instead of RK45: LSODA auto-detects stiffness and switches to BDF
                # for implicit solve, avoiding the micro-step trap near bistability boundaries
                x0_fit = np.array([P0_fit, 0.0, 0.0, 0.0, 0.0])
                sol = solve_reynolds(
                    params_fit, x0_fit, obs_timepoints,
                    method='LSODA',
                    rtol=1e-4, atol=1e-6, dense_output=False
                )

                # Residuals
                res_Nstar = (obs_Nstar - sol['Nstar']) ** 2
                res_CA = (obs_CA - sol['CA']) ** 2
                res_f = (obs_f - sol['f']) ** 2

                residual = np.sum(res_Nstar + res_CA + res_f)

                if not np.isfinite(residual):
                    return 1e10

                return residual

            except Exception as e:
                return 1e10

        # Optimize with timeout callback to prevent hanging on pathological parameters
        deadline = time.monotonic() + 60  # 60s max per patient per restart

        def timeout_callback(xk):
            if time.monotonic() > deadline:
                raise StopIteration("calibration timeout")

        try:
            result = minimize(
                residual_fn,
                x0,
                method='L-BFGS-B',
                options={'ftol': 1e-6, 'maxiter': 500},
                callback=timeout_callback,
            )

            if result.fun < best_residual:
                best_residual = result.fun
                best_result = result
        except (Exception, StopIteration) as e:
            if verbose:
                print(f"  Restart {restart} failed: {e}")
            continue

    if best_result is None:
        # Calibration failed; return initial parameters
        return params_init.copy(), patient['P0'], 0.0

    # Decode best result
    log_params_best = best_result.x
    fitted_calib = dict(zip(CALIBRATION_PARAMS, np.exp(log_params_best[:-1])))
    fitted_P0 = np.exp(log_params_best[-1])

    fitted_params = {**params_init}
    fitted_params.update(fitted_calib)

    # Compute fit quality (R²) with moderate precision (final evaluation)
    x0_fit = np.array([fitted_P0, 0.0, 0.0, 0.0, 0.0])
    sol_fit = solve_reynolds(
        fitted_params, x0_fit, obs_timepoints,
        method='LSODA',
        rtol=1e-8, atol=1e-10, max_step=0.5, dense_output=False
    )

    # R² for combined observed variables
    obs_all = np.concatenate([patient['obs_Nstar'], patient['obs_CA'], patient['obs_f']])
    pred_all = np.concatenate([sol_fit['Nstar'], sol_fit['CA'], sol_fit['f']])

    ss_res = np.sum((obs_all - pred_all) ** 2)
    ss_tot = np.sum((obs_all - np.mean(obs_all)) ** 2)
    r2_quality = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return fitted_params, fitted_P0, r2_quality


def generate_motif_proxies(
    patient: Dict,
    fitted_params: Dict,
    fitted_P0: float,
    t_eval: np.ndarray = None,
) -> Dict:
    """
    Generate MOTIF proxy trajectories for latent variables.

    Args:
        patient: Patient dict
        fitted_params: Calibrated parameters
        fitted_P0: Calibrated initial pathogen load
        t_eval: Time points for proxy evaluation (high resolution)

    Returns:
        Dict with proxy trajectories and summary statistics
    """
    if t_eval is None:
        t_eval = np.linspace(0, 72, 200)

    # Simulate with fitted parameters (fast tolerances OK for proxy generation)
    x0 = np.array([fitted_P0, 0.0, 0.0, 0.0, 0.0])
    try:
        sol = solve_reynolds(
            fitted_params, x0, t_eval,
            method='RK45',
            rtol=1e-4, atol=1e-6, max_step=2.0, dense_output=False
        )
    except Exception:
        return {}

    # Extract proxies at observation timepoints
    obs_t = patient['obs_t']
    idx_obs = [np.argmin(np.abs(t_eval - t)) for t in obs_t]

    proxies = {
        'proxy_P': sol['P'][idx_obs],
        'proxy_D': sol['D'][idx_obs],
        'proxy_h': sol['h'][idx_obs],
        # Summary statistics
        'proxy_P_auc': trapezoid(sol['P'], t_eval),
        'proxy_P_peak': np.max(sol['P']),
        'proxy_P_final': sol['P'][-1],
        'proxy_D_auc': trapezoid(sol['D'], t_eval),
        'proxy_D_peak': np.max(sol['D']),
        'proxy_h_min': np.min(sol['h']),
        'proxy_h_final': sol['h'][-1],
    }

    return proxies


def extract_features(
    patient: Dict,
    proxies: Optional[Dict] = None,
) -> Dict:
    """
    Extract feature vector from patient data and optionally proxies.

    Features: AUC, peak, final value for each observed/proxy variable.

    Args:
        patient: Patient dict with observed data
        proxies: Proxy dict (if None, only observed features extracted)

    Returns:
        Dictionary of features
    """
    features = {}

    # Observed features
    obs_t = patient['obs_t']

    # N* features
    features['obs_Nstar_auc'] = trapezoid(patient['obs_Nstar'], obs_t)
    features['obs_Nstar_peak'] = np.max(patient['obs_Nstar'])
    features['obs_Nstar_final'] = patient['obs_Nstar'][-1]

    # CA features
    features['obs_CA_auc'] = trapezoid(patient['obs_CA'], obs_t)
    features['obs_CA_peak'] = np.max(patient['obs_CA'])
    features['obs_CA_final'] = patient['obs_CA'][-1]

    # f features
    features['obs_f_auc'] = trapezoid(patient['obs_f'], obs_t)
    features['obs_f_peak'] = np.max(patient['obs_f'])
    features['obs_f_final'] = patient['obs_f'][-1]

    # Proxy features (if available)
    if proxies is not None:
        for key, val in proxies.items():
            if isinstance(val, (int, float, np.number)):
                features[key] = val

    return features


def motif_correlation_analysis(
    patients_with_proxies: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Compute correlation matrix between observed and proxy features.

    Args:
        patients_with_proxies: List of patient dicts with proxies field

    Returns:
        Tuple of (correlation_matrix, pvalue_matrix, obs_feature_names, proxy_feature_names)
    """
    obs_features = []
    proxy_features = []

    for patient in patients_with_proxies:
        if 'features' in patient and 'proxies' in patient:
            obs_feat = [
                patient['features'].get(k)
                for k in ['obs_Nstar_auc', 'obs_CA_auc', 'obs_f_auc',
                          'obs_Nstar_peak', 'obs_CA_peak', 'obs_f_peak']
            ]
            proxy_feat = [
                patient['proxies'].get(k)
                for k in ['proxy_P_auc', 'proxy_D_auc', 'proxy_h_min',
                          'proxy_P_peak', 'proxy_D_peak', 'proxy_h_final']
            ]

            if all(v is not None for v in obs_feat) and all(v is not None for v in proxy_feat):
                obs_features.append(obs_feat)
                proxy_features.append(proxy_feat)

    if not obs_features:
        return np.array([]), np.array([]), [], []

    obs_features = np.array(obs_features)
    proxy_features = np.array(proxy_features)

    # Correlation matrix
    n_obs = obs_features.shape[1]
    n_proxy = proxy_features.shape[1]
    corr_matrix = np.zeros((n_obs, n_proxy))
    pval_matrix = np.zeros((n_obs, n_proxy))

    for i in range(n_obs):
        for j in range(n_proxy):
            r, p = spearmanr(obs_features[:, i], proxy_features[:, j])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

    obs_names = ['Nstar_auc', 'CA_auc', 'f_auc', 'Nstar_peak', 'CA_peak', 'f_peak']
    proxy_names = ['P_auc', 'D_auc', 'h_min', 'P_peak', 'D_peak', 'h_final']

    return corr_matrix, pval_matrix, obs_names, proxy_names


def motif_classify_outcomes(
    train_patients: List[Dict],
    test_patients: List[Dict],
    use_proxies: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Train logistic regression classifier on MOTIF features.

    Args:
        train_patients: Training patient list (with features and proxies)
        test_patients: Test patient list
        use_proxies: Include proxy features (True) or only observed (False)
        verbose: Print progress

    Returns:
        Classification results dict with AUROC, F1, confusion matrix
    """
    if not HAS_SKLEARN:
        # Return basic dummy results if sklearn not available
        return {
            'error': 'sklearn not available',
            'accuracy': 0.0,
            'n_test': 0,
        }

    # Extract training features
    X_train = []
    y_train = []

    for patient in train_patients:
        if 'features' in patient:
            features = patient['features']
            obs_feat = [
                features.get('obs_Nstar_auc', 0),
                features.get('obs_CA_auc', 0),
                features.get('obs_f_auc', 0),
                features.get('obs_Nstar_peak', 0),
                features.get('obs_CA_peak', 0),
                features.get('obs_f_peak', 0),
            ]

            if use_proxies and 'proxies' in patient:
                proxy_feat = [
                    patient['proxies'].get('proxy_P_auc', 0),
                    patient['proxies'].get('proxy_D_auc', 0),
                    patient['proxies'].get('proxy_h_min', 0),
                    patient['proxies'].get('proxy_P_peak', 0),
                    patient['proxies'].get('proxy_D_peak', 0),
                    patient['proxies'].get('proxy_h_final', 0),
                ]
                obs_feat.extend(proxy_feat)

            X_train.append(obs_feat)

            # Encode outcome as integer
            outcome_map = {'resolution': 0, 'chronic': 1, 'death': 2}
            y_train.append(outcome_map.get(patient['outcome'], 0))

    if not X_train:
        return {'error': 'No training samples'}

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train classifier
    try:
        clf = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)
    except Exception as e:
        return {'error': f'Training failed: {e}'}

    # Evaluate on test set
    X_test = []
    y_test = []

    for patient in test_patients:
        if 'features' in patient:
            features = patient['features']
            obs_feat = [
                features.get('obs_Nstar_auc', 0),
                features.get('obs_CA_auc', 0),
                features.get('obs_f_auc', 0),
                features.get('obs_Nstar_peak', 0),
                features.get('obs_CA_peak', 0),
                features.get('obs_f_peak', 0),
            ]

            if use_proxies and 'proxies' in patient:
                proxy_feat = [
                    patient['proxies'].get('proxy_P_auc', 0),
                    patient['proxies'].get('proxy_D_auc', 0),
                    patient['proxies'].get('proxy_h_min', 0),
                    patient['proxies'].get('proxy_P_peak', 0),
                    patient['proxies'].get('proxy_D_peak', 0),
                    patient['proxies'].get('proxy_h_final', 0),
                ]
                obs_feat.extend(proxy_feat)

            X_test.append(obs_feat)
            outcome_map = {'resolution': 0, 'chronic': 1, 'death': 2}
            y_test.append(outcome_map.get(patient['outcome'], 0))

    if not X_test:
        return {'error': 'No test samples'}

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)

    # Metrics
    try:
        auroc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    except Exception:
        auroc = np.nan

    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    return {
        'auroc': float(auroc),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'n_test': len(y_test),
        'accuracy': float(np.mean(y_pred == y_test)),
    }


def run_motif_pipeline(
    cohort: List[Dict],
    config: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run complete MOTIF pipeline on patient cohort.

    Args:
        cohort: List of patient dicts
        config: Configuration dict (optional)
        verbose: Print progress

    Returns:
        Results dict with all pipeline outputs
    """
    if config is None:
        config = {
            'n_calibration_restarts': 3,
            'train_test_split': 0.8,
            'random_seed': 42,
        }

    np.random.seed(config.get('random_seed', 42))

    results = {
        'config': config,
        'n_patients': len(cohort),
        'calibration_results': {},
        'proxy_results': {},
        'classification_results': {},
    }

    if verbose:
        print(f"Running MOTIF pipeline on {len(cohort)} patients...")

    # Step 1: Calibration (parallelized across all available cores)
    if verbose:
        print("  Step 1: Parameter calibration...")

    def calibrate_one_patient(idx_patient_tuple):
        """Helper function for parallel calibration."""
        idx, patient = idx_patient_tuple
        try:
            fitted_params, fitted_P0, fit_quality = calibrate_patient(
                patient,
                params_init=REYNOLDS_PARAMS.copy(),
                n_restarts=config.get('n_restarts', 1),  # Default to 1 restart (fast)
                verbose=False,
            )
            patient['fitted_params'] = fitted_params
            patient['fitted_P0'] = fitted_P0
            patient['fit_quality'] = fit_quality
            return patient
        except Exception as e:
            if verbose:
                print(f"    Patient {idx} calibration failed: {e}")
            return None

    # Run calibration serially to avoid scipy.optimize + OpenBLAS crashes on Windows
    # with multiprocessing. joblib.Parallel with threads causes deadlock, processes cause segfaults.
    # Serial is stable and acceptable for reasonable patient counts (< 1000).
    results_list = []
    for idx, patient in tqdm(
        enumerate(cohort),
        total=len(cohort),
        desc='Calibrating patients',
        disable=not verbose
    ):
        result = calibrate_one_patient((idx, patient))
        results_list.append(result)

    # Filter out failed calibrations
    calibrated_patients = [p for p in results_list if p is not None]

    results['n_calibrated'] = len(calibrated_patients)

    # Step 2: Proxy generation
    if verbose:
        print("  Step 2: Proxy generation...")

    for patient in tqdm(calibrated_patients, disable=not verbose):
        try:
            proxies = generate_motif_proxies(
                patient,
                patient['fitted_params'],
                patient['fitted_P0'],
            )
            patient['proxies'] = proxies

            # Extract features
            patient['features'] = extract_features(patient, proxies)
        except Exception as e:
            if verbose:
                print(f"    Proxy generation failed: {e}")

    # Step 3: Correlation analysis
    if verbose:
        print("  Step 3: Correlation analysis...")

    corr_matrix, pval_matrix, obs_names, proxy_names = motif_correlation_analysis(
        calibrated_patients
    )

    if corr_matrix.size > 0:
        results['correlation_analysis'] = {
            'correlation_matrix': corr_matrix.tolist(),
            'pvalue_matrix': pval_matrix.tolist(),
            'observed_features': obs_names,
            'proxy_features': proxy_names,
        }

    # Step 4: Outcome classification
    if verbose:
        print("  Step 4: Outcome classification...")

    n_train = int(len(calibrated_patients) * config.get('train_test_split', 0.8))
    train_patients = calibrated_patients[:n_train]
    test_patients = calibrated_patients[n_train:]

    clf_results_with_proxy = motif_classify_outcomes(
        train_patients, test_patients, use_proxies=True, verbose=False
    )
    clf_results_without_proxy = motif_classify_outcomes(
        train_patients, test_patients, use_proxies=False, verbose=False
    )

    results['classification_results'] = {
        'with_proxies': clf_results_with_proxy,
        'without_proxies': clf_results_without_proxy,
    }

    # Step 5: Recovery metrics
    if verbose:
        print("  Step 5: Computing recovery metrics...")

    recovery_metrics = {}
    for var in ['P', 'D', 'h']:
        true_vals = []
        pred_vals = []

        for p in calibrated_patients:
            if 'proxies' in p and f'proxy_{var}' in p['proxies']:
                proxy_vals = p['proxies'][f'proxy_{var}']
                if hasattr(proxy_vals, '__len__'):
                    true_vals.extend(p[f'true_{var}'])
                    pred_vals.extend(proxy_vals)

        if len(pred_vals) > 0 and len(true_vals) == len(pred_vals):
            metrics = compute_recovery_metrics(np.array(pred_vals), np.array(true_vals), f'proxy_{var}')
            recovery_metrics[f'proxy_{var}'] = metrics

    results['recovery_metrics'] = recovery_metrics

    if verbose:
        print("  MOTIF pipeline complete.")

    return results


if __name__ == '__main__':
    print("Testing MOTIF pipeline implementation...")

    # Load a small cohort for testing
    try:
        from src.synthetic_data import generate_cohort
    except ImportError:
        from synthetic_data import generate_cohort

    cohort_small = generate_cohort(N_resolution=10, N_chronic=5, N_death=5, seed=42, verbose=False)

    print(f"\nGenerating MOTIF on {len(cohort_small)} test patients...")

    results = run_motif_pipeline(cohort_small, verbose=True)

    print("\nMOTIF Pipeline Results:")
    print(f"  Calibrated: {results['n_calibrated']}/{results['n_patients']}")

    if 'recovery_metrics' in results:
        print("  Recovery metrics:")
        for var, metrics in results['recovery_metrics'].items():
            print(f"    {var}: R² = {metrics['r2']:.3f}")

    if 'classification_results' in results:
        clf_with = results['classification_results'].get('with_proxies', {})
        clf_without = results['classification_results'].get('without_proxies', {})
        print(f"  Classification AUROC (with proxies): {clf_with.get('auroc', np.nan):.3f}")
        print(f"  Classification AUROC (without proxies): {clf_without.get('auroc', np.nan):.3f}")

    print("\nStep 5: MOTIF pipeline implementation complete.")
