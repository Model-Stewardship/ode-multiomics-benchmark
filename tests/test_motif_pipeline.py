"""Tests for MOTIF pipeline implementation."""

import numpy as np
import unittest
from src.motif_pipeline import (
    calibrate_patient,
    generate_motif_proxies,
    extract_features,
    motif_correlation_analysis,
    motif_classify_outcomes,
    run_motif_pipeline,
    CALIBRATION_PARAMS,
)
from src.synthetic_data import generate_patient, generate_cohort
from src.reynolds_ode import REYNOLDS_PARAMS


class TestPatientCalibration(unittest.TestCase):
    """Test parameter calibration for individual patients."""

    def test_calibrate_patient_returns_tuple(self):
        """Verify calibration returns correct tuple structure."""
        patient = generate_patient(P0=5.0, seed=42)

        fitted_params, fitted_P0, fit_quality = calibrate_patient(patient, n_restarts=1)

        self.assertIsInstance(fitted_params, dict)
        self.assertIsInstance(fitted_P0, (float, np.floating))
        self.assertIsInstance(fit_quality, (float, np.floating))

    def test_calibrate_patient_contains_params(self):
        """Verify fitted parameters dict has required keys."""
        patient = generate_patient(P0=5.0, seed=42)

        fitted_params, _, _ = calibrate_patient(patient, n_restarts=1)

        # Should have all Reynolds parameters
        for param in REYNOLDS_PARAMS.keys():
            self.assertIn(param, fitted_params)

    def test_calibrate_patient_P0_positive(self):
        """Verify fitted P0 is positive."""
        patient = generate_patient(P0=5.0, seed=42)

        _, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)

        self.assertGreater(fitted_P0, 0)
        self.assertLess(fitted_P0, 50)

    def test_calibrate_patient_fit_quality_valid(self):
        """Verify fit quality R² is in valid range."""
        patient = generate_patient(P0=5.0, seed=42)

        _, _, fit_quality = calibrate_patient(patient, n_restarts=1)

        self.assertGreaterEqual(fit_quality, -10)  # Can be negative (worse than mean)
        self.assertLessEqual(fit_quality, 1.0)

    def test_calibrate_patient_multiple_restarts(self):
        """Verify multiple restarts improve fit."""
        patient = generate_patient(P0=5.0, seed=42)

        _, _, fit_1 = calibrate_patient(patient, n_restarts=1)
        _, _, fit_3 = calibrate_patient(patient, n_restarts=3)

        # More restarts should give better or equal fit
        self.assertGreaterEqual(fit_3, fit_1 - 0.01)


class TestProxyGeneration(unittest.TestCase):
    """Test MOTIF proxy generation."""

    def test_generate_proxies_returns_dict(self):
        """Verify proxy generation returns dict."""
        patient = generate_patient(P0=5.0, seed=42)
        fitted_params, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)

        proxies = generate_motif_proxies(patient, fitted_params, fitted_P0)

        self.assertIsInstance(proxies, dict)

    def test_generate_proxies_contains_expected_keys(self):
        """Verify proxy dict has expected fields."""
        patient = generate_patient(P0=5.0, seed=42)
        fitted_params, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)

        proxies = generate_motif_proxies(patient, fitted_params, fitted_P0)

        expected_keys = {'proxy_P', 'proxy_D', 'proxy_h', 'proxy_P_auc', 'proxy_D_auc'}
        self.assertTrue(expected_keys.issubset(set(proxies.keys())))

    def test_proxy_values_finite(self):
        """Verify proxy values are finite."""
        patient = generate_patient(P0=5.0, seed=42)
        fitted_params, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)

        proxies = generate_motif_proxies(patient, fitted_params, fitted_P0)

        for key, val in proxies.items():
            if isinstance(val, (int, float, np.number)):
                self.assertTrue(np.isfinite(val), f"{key} is not finite")

    def test_proxy_array_shapes(self):
        """Verify proxy array shapes match observation timepoints."""
        patient = generate_patient(P0=5.0, seed=42)
        fitted_params, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)

        proxies = generate_motif_proxies(patient, fitted_params, fitted_P0)

        n_obs = len(patient['obs_t'])
        self.assertEqual(len(proxies['proxy_P']), n_obs)
        self.assertEqual(len(proxies['proxy_D']), n_obs)
        self.assertEqual(len(proxies['proxy_h']), n_obs)


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction from patient data."""

    def test_extract_features_returns_dict(self):
        """Verify feature extraction returns dict."""
        patient = generate_patient(P0=5.0, seed=42)

        features = extract_features(patient)

        self.assertIsInstance(features, dict)

    def test_extract_features_contains_observed(self):
        """Verify observed features are extracted."""
        patient = generate_patient(P0=5.0, seed=42)

        features = extract_features(patient)

        self.assertIn('obs_Nstar_auc', features)
        self.assertIn('obs_CA_auc', features)
        self.assertIn('obs_f_auc', features)

    def test_extract_features_with_proxies(self):
        """Verify proxy features are added when proxies provided."""
        patient = generate_patient(P0=5.0, seed=42)
        fitted_params, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)
        proxies = generate_motif_proxies(patient, fitted_params, fitted_P0)

        features_without = extract_features(patient)
        features_with = extract_features(patient, proxies)

        # With proxies should have more features
        self.assertLess(len(features_without), len(features_with))
        self.assertIn('proxy_P_auc', features_with)

    def test_feature_values_finite(self):
        """Verify all feature values are finite."""
        patient = generate_patient(P0=5.0, seed=42)
        fitted_params, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)
        proxies = generate_motif_proxies(patient, fitted_params, fitted_P0)

        features = extract_features(patient, proxies)

        for key, val in features.items():
            self.assertTrue(np.isfinite(val), f"Feature {key} is not finite: {val}")


class TestCorrelationAnalysis(unittest.TestCase):
    """Test correlation analysis between observed and proxy features."""

    def test_correlation_analysis_returns_arrays(self):
        """Verify correlation analysis returns expected arrays."""
        cohort = generate_cohort(N_resolution=5, N_chronic=3, N_death=2, seed=42, verbose=False)

        # Calibrate and generate proxies
        for patient in cohort:
            fitted_params, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)
            patient['fitted_params'] = fitted_params
            patient['fitted_P0'] = fitted_P0
            proxies = generate_motif_proxies(patient, fitted_params, fitted_P0)
            patient['proxies'] = proxies
            patient['features'] = extract_features(patient, proxies)

        corr_matrix, pval_matrix, obs_names, proxy_names = motif_correlation_analysis(cohort)

        self.assertEqual(corr_matrix.shape, (6, 6))
        self.assertEqual(pval_matrix.shape, (6, 6))
        self.assertEqual(len(obs_names), 6)
        self.assertEqual(len(proxy_names), 6)

    def test_correlation_values_valid(self):
        """Verify correlation values are in [-1, 1]."""
        cohort = generate_cohort(N_resolution=5, N_chronic=3, N_death=2, seed=42, verbose=False)

        for patient in cohort:
            fitted_params, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)
            patient['fitted_params'] = fitted_params
            patient['fitted_P0'] = fitted_P0
            proxies = generate_motif_proxies(patient, fitted_params, fitted_P0)
            patient['proxies'] = proxies
            patient['features'] = extract_features(patient, proxies)

        corr_matrix, _, _, _ = motif_correlation_analysis(cohort)

        self.assertTrue(np.all(np.abs(corr_matrix) <= 1.0))


class TestOutcomeClassification(unittest.TestCase):
    """Test outcome classification using MOTIF features."""

    def test_classify_outcomes_returns_dict(self):
        """Verify classification returns dict with expected keys."""
        cohort = generate_cohort(N_resolution=10, N_chronic=5, N_death=5, seed=42, verbose=False)

        # Prepare cohort
        for patient in cohort:
            fitted_params, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)
            patient['fitted_params'] = fitted_params
            patient['fitted_P0'] = fitted_P0
            proxies = generate_motif_proxies(patient, fitted_params, fitted_P0)
            patient['proxies'] = proxies
            patient['features'] = extract_features(patient, proxies)

        train_patients = cohort[:12]
        test_patients = cohort[12:]

        results = motif_classify_outcomes(train_patients, test_patients, use_proxies=True)

        self.assertIn('auroc', results)
        self.assertIn('f1', results)
        self.assertIn('confusion_matrix', results)
        self.assertIn('accuracy', results)

    def test_classify_with_vs_without_proxies(self):
        """Verify classification works with and without proxies."""
        cohort = generate_cohort(N_resolution=10, N_chronic=5, N_death=5, seed=42, verbose=False)

        for patient in cohort:
            fitted_params, fitted_P0, _ = calibrate_patient(patient, n_restarts=1)
            patient['fitted_params'] = fitted_params
            patient['fitted_P0'] = fitted_P0
            proxies = generate_motif_proxies(patient, fitted_params, fitted_P0)
            patient['proxies'] = proxies
            patient['features'] = extract_features(patient, proxies)

        train_patients = cohort[:12]
        test_patients = cohort[12:]

        results_with = motif_classify_outcomes(train_patients, test_patients, use_proxies=True)
        results_without = motif_classify_outcomes(train_patients, test_patients, use_proxies=False)

        self.assertIsNotNone(results_with['auroc'])
        self.assertIsNotNone(results_without['auroc'])
        self.assertGreater(results_with['auroc'], 0)
        self.assertGreater(results_without['auroc'], 0)


class TestMOTIFPipeline(unittest.TestCase):
    """Test full MOTIF pipeline."""

    def test_run_motif_pipeline_returns_results(self):
        """Verify pipeline returns results dict."""
        cohort = generate_cohort(N_resolution=10, N_chronic=5, N_death=5, seed=42, verbose=False)

        results = run_motif_pipeline(cohort, verbose=False)

        self.assertIsInstance(results, dict)
        self.assertIn('config', results)
        self.assertIn('n_patients', results)
        self.assertIn('n_calibrated', results)

    def test_run_motif_pipeline_completes(self):
        """Verify pipeline completes without errors."""
        cohort = generate_cohort(N_resolution=10, N_chronic=5, N_death=5, seed=42, verbose=False)

        results = run_motif_pipeline(cohort, verbose=False)

        self.assertGreater(results['n_calibrated'], 0)
        self.assertLessEqual(results['n_calibrated'], results['n_patients'])

    def test_run_motif_pipeline_recovery_metrics(self):
        """Verify recovery metrics are computed."""
        cohort = generate_cohort(N_resolution=10, N_chronic=5, N_death=5, seed=42, verbose=False)

        results = run_motif_pipeline(cohort, verbose=False)

        if 'recovery_metrics' in results and results['recovery_metrics']:
            for var, metrics in results['recovery_metrics'].items():
                self.assertIn('r2', metrics)
                self.assertIn('rmse', metrics)

    def test_run_motif_pipeline_classification(self):
        """Verify classification results are included."""
        cohort = generate_cohort(N_resolution=10, N_chronic=5, N_death=5, seed=42, verbose=False)

        results = run_motif_pipeline(cohort, verbose=False)

        self.assertIn('classification_results', results)
        self.assertIn('with_proxies', results['classification_results'])
        self.assertIn('without_proxies', results['classification_results'])

    def test_run_motif_pipeline_correlation(self):
        """Verify correlation analysis is included."""
        cohort = generate_cohort(N_resolution=10, N_chronic=5, N_death=5, seed=42, verbose=False)

        results = run_motif_pipeline(cohort, verbose=False)

        if 'correlation_analysis' in results:
            self.assertIn('correlation_matrix', results['correlation_analysis'])
            self.assertIn('observed_features', results['correlation_analysis'])
            self.assertIn('proxy_features', results['correlation_analysis'])


class TestMOTIFBaseline(unittest.TestCase):
    """Test MOTIF pipeline on realistic cohort."""

    def test_motif_on_50_patient_cohort(self):
        """Run MOTIF on 50-patient cohort and verify basic quality."""
        cohort = generate_cohort(N_resolution=20, N_chronic=15, N_death=15, seed=42, verbose=False)

        results = run_motif_pipeline(cohort, verbose=False)

        # Should calibrate most patients
        self.assertGreater(results['n_calibrated'], 40)

        # Should have recovery metrics
        self.assertIn('recovery_metrics', results)
        if results['recovery_metrics']:
            for var, metrics in results['recovery_metrics'].items():
                # R² should be reasonable
                self.assertGreater(metrics['r2'], -0.5)

        # Classification should work
        clf_results = results['classification_results']['with_proxies']
        if 'auroc' in clf_results and not np.isnan(clf_results['auroc']):
            self.assertGreater(clf_results['auroc'], 0.5)


if __name__ == '__main__':
    unittest.main()
