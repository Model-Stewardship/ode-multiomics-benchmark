"""Tests for evaluation metrics module."""

import numpy as np
import unittest
import tempfile
from pathlib import Path
import json

from src.evaluation import (
    compute_recovery_metrics,
    compute_classification_metrics,
    compare_pipelines,
    run_robustness_experiment,
    save_results,
    convert_numpy_to_native,
    aggregate_replicate_results,
    HAS_SKLEARN,
)


class TestRecoveryMetrics(unittest.TestCase):
    """Test recovery metric computation."""

    def test_perfect_prediction(self):
        """Test metrics when prediction matches ground truth."""
        true_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred_vals = true_vals.copy()

        metrics = compute_recovery_metrics(pred_vals, true_vals, 'test')

        self.assertAlmostEqual(metrics['r2'], 1.0, places=5)
        self.assertAlmostEqual(metrics['rmse'], 0.0, places=10)
        self.assertAlmostEqual(metrics['spearman_r'], 1.0, places=5)

    def test_poor_prediction(self):
        """Test metrics with poor predictions."""
        true_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred_vals = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Inverted

        metrics = compute_recovery_metrics(pred_vals, true_vals, 'test')

        # R² should be negative (worse than mean)
        self.assertLess(metrics['r2'], 0)
        # Spearman r should be -1 (perfect anti-correlation)
        self.assertAlmostEqual(metrics['spearman_r'], -1.0, places=5)

    def test_nan_handling(self):
        """Test that NaN values are handled gracefully."""
        true_vals = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        pred_vals = np.array([1.1, 2.1, 2.9, 4.0, 5.1])

        metrics = compute_recovery_metrics(pred_vals, true_vals, 'test')

        # Should still compute metrics on valid values
        self.assertIsNotNone(metrics['r2'])
        self.assertIsNotNone(metrics['rmse'])
        self.assertEqual(metrics['n_samples'], 4)

    def test_metric_dict_structure(self):
        """Verify metric output structure."""
        true_vals = np.array([1.0, 2.0, 3.0])
        pred_vals = np.array([1.1, 2.1, 3.1])

        metrics = compute_recovery_metrics(pred_vals, true_vals, 'my_var')

        required_keys = {'r2', 'rmse', 'spearman_r', 'spearman_pval', 'n_samples', 'variable'}
        self.assertEqual(set(metrics.keys()), required_keys)
        self.assertEqual(metrics['variable'], 'my_var')

    def test_2d_input(self):
        """Test recovery metrics with 2D arrays."""
        true_vals = np.array([[1, 2], [3, 4]])
        pred_vals = np.array([[1.1, 2.1], [2.9, 4.1]])

        metrics = compute_recovery_metrics(pred_vals, true_vals)

        self.assertGreater(metrics['r2'], 0.8)
        self.assertLess(metrics['rmse'], 0.2)


class TestClassificationMetrics(unittest.TestCase):
    """Test classification metric computation."""

    @unittest.skipIf(not HAS_SKLEARN, "sklearn not available")
    def test_binary_classification_perfect(self):
        """Test perfect binary classification."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        metrics = compute_classification_metrics(y_pred, y_true)

        self.assertAlmostEqual(metrics['f1'], 1.0, places=5)
        self.assertAlmostEqual(metrics['auroc'], 1.0, places=5)

    @unittest.skipIf(not HAS_SKLEARN, "sklearn not available")
    def test_binary_classification_random(self):
        """Test binary classification with random predictions."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 0, 1, 0])

        metrics = compute_classification_metrics(y_pred, y_true)

        # Should have reasonable values
        self.assertGreaterEqual(metrics['f1'], 0)
        self.assertLessEqual(metrics['f1'], 1)
        self.assertGreaterEqual(metrics['auroc'], 0)
        self.assertLessEqual(metrics['auroc'], 1)

    @unittest.skipIf(not HAS_SKLEARN, "sklearn not available")
    def test_multiclass_classification(self):
        """Test multi-class classification (3 classes)."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 0, 2])

        class_names = ['resolution', 'chronic', 'death']
        metrics = compute_classification_metrics(y_pred, y_true, class_names)

        self.assertEqual(metrics['n_classes'], 3)
        self.assertIn('f1_macro', metrics)
        self.assertEqual(len(metrics['confusion_matrix']), 3)

    @unittest.skipIf(not HAS_SKLEARN, "sklearn not available")
    def test_confusion_matrix_shape(self):
        """Verify confusion matrix shape matches number of classes."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = compute_classification_metrics(y_pred, y_true)

        cm = np.array(metrics['confusion_matrix'])
        self.assertEqual(cm.shape, (3, 3))

    def test_class_names_preserved(self):
        """Verify class names are preserved in output."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        class_names = ['neg', 'pos']
        metrics = compute_classification_metrics(y_pred, y_true, class_names)

        self.assertEqual(metrics['class_names'], class_names)

    def test_classification_metrics_basic(self):
        """Test basic classification metrics work (with or without sklearn)."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        metrics = compute_classification_metrics(y_pred, y_true)

        # Always available: n_samples, n_classes, class_names, accuracy
        self.assertEqual(metrics['n_samples'], 6)
        self.assertEqual(metrics['n_classes'], 2)
        self.assertAlmostEqual(metrics['accuracy'], 1.0, places=5)


class TestPipelineComparison(unittest.TestCase):
    """Test pipeline comparison functionality."""

    def test_compare_pipelines_structure(self):
        """Verify comparison output structure."""
        motif_results = {
            'recovery_metrics': {
                'proxy_p': {'r2': 0.85, 'rmse': 0.1},
                'proxy_d': {'r2': 0.80, 'rmse': 0.15},
            },
            'classification_metrics': {'auroc_macro': 0.92, 'f1_macro': 0.89},
        }

        ude_results = {
            'recovery_metrics': {
                'P': {'r2': 0.80, 'rmse': 0.15},
                'D': {'r2': 0.75, 'rmse': 0.20},
            },
            'classification_metrics': {'auroc_macro': 0.88, 'f1_macro': 0.85},
        }

        comparison = compare_pipelines(motif_results, ude_results)

        self.assertIn('pipelines', comparison)
        self.assertIn('comparison', comparison)
        self.assertIn('motif', comparison['pipelines'])
        self.assertIn('ude', comparison['pipelines'])
        self.assertIn('timestamp', comparison)

    def test_compare_pipelines_winner(self):
        """Verify comparison correctly identifies better method."""
        motif_results = {
            'recovery_metrics': {
                'proxy_p': {'r2': 0.90},  # Better
            },
            'classification_metrics': {'auroc_macro': 0.85},
        }

        ude_results = {
            'recovery_metrics': {
                'P': {'r2': 0.75},
            },
            'classification_metrics': {'auroc_macro': 0.80},
        }

        comparison = compare_pipelines(motif_results, ude_results)

        self.assertTrue(comparison['comparison']['recovery']['P']['motif_better'])


class TestRobustnessExperiment(unittest.TestCase):
    """Test robustness experiment framework."""

    def test_robustness_experiment_runs(self):
        """Verify robustness experiment executes without error."""
        def dummy_fn(param_val, seed):
            np.random.seed(seed)
            return {
                'metric1': np.random.uniform(0.5, 0.95),
                'metric2': np.random.uniform(0.1, 0.3),
            }

        results = run_robustness_experiment(
            dummy_fn,
            'test_param',
            [0.1, 0.2],
            n_replicates=2,
            verbose=False,
        )

        self.assertEqual(results['param_name'], 'test_param')
        self.assertEqual(len(results['results_by_param']), 2)

    def test_robustness_experiment_aggregation(self):
        """Verify aggregation of replicates."""
        def dummy_fn(param_val, seed):
            np.random.seed(seed)
            return {'metric': param_val + np.random.uniform(-0.01, 0.01)}

        results = run_robustness_experiment(
            dummy_fn,
            'param',
            [1.0],
            n_replicates=3,
            verbose=False,
        )

        summary = results['results_by_param']['1.0']['summary']
        self.assertIn('metric_mean', summary)
        self.assertIn('metric_std', summary)
        self.assertEqual(summary['metric_n'], 3)

    def test_robustness_experiment_error_handling(self):
        """Verify robustness experiment handles errors gracefully."""
        def failing_fn(param_val, seed):
            raise ValueError("Test error")

        results = run_robustness_experiment(
            failing_fn,
            'param',
            [0.1],
            n_replicates=1,
            verbose=False,
        )

        # Should still return results structure
        self.assertIn('results_by_param', results)
        # Summary should indicate all replicates failed
        summary = results['results_by_param']['0.1']['summary']
        self.assertIn('error', summary)


class TestAggregation(unittest.TestCase):
    """Test replicate aggregation."""

    def test_aggregate_valid_replicates(self):
        """Verify aggregation of valid results."""
        replicates = [
            {'metric': 0.8},
            {'metric': 0.9},
            {'metric': 0.85},
        ]

        summary = aggregate_replicate_results(replicates)

        self.assertAlmostEqual(summary['metric_mean'], 0.85, places=5)
        self.assertGreater(summary['metric_std'], 0)
        self.assertEqual(summary['metric_n'], 3)

    def test_aggregate_with_errors(self):
        """Verify aggregation skips error results."""
        replicates = [
            {'metric': 0.8},
            {'error': 'some error'},
            {'metric': 0.9},
        ]

        summary = aggregate_replicate_results(replicates)

        self.assertAlmostEqual(summary['metric_mean'], 0.85, places=5)
        self.assertEqual(summary['metric_n'], 2)


class TestSerialization(unittest.TestCase):
    """Test results serialization."""

    def test_save_results_json(self):
        """Verify results can be saved to JSON."""
        results = {
            'metric1': 0.85,
            'array': np.array([1, 2, 3]),
            'nested': {
                'metric2': np.float64(0.95),
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'results.json'
            save_results(results, str(output_path))

            self.assertTrue(output_path.exists())

            # Verify can be loaded back
            with open(output_path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded['metric1'], 0.85)
            self.assertEqual(loaded['array'], [1, 2, 3])
            self.assertEqual(loaded['nested']['metric2'], 0.95)

    def test_convert_numpy_types(self):
        """Verify numpy type conversion."""
        obj = {
            'int': np.int64(42),
            'float': np.float64(3.14),
            'bool': np.bool_(True),
            'array': np.array([1, 2, 3]),
            'list': [np.int32(1), np.float32(2.5)],
        }

        converted = convert_numpy_to_native(obj)

        self.assertEqual(converted['int'], 42)
        self.assertAlmostEqual(converted['float'], 3.14)
        self.assertEqual(converted['bool'], True)
        self.assertEqual(converted['array'], [1, 2, 3])
        self.assertEqual(converted['list'], [1, 2.5])


if __name__ == '__main__':
    unittest.main()
