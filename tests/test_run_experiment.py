"""Tests for experiment runner."""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.run_experiment import (
    run_experiment,
    run_single_replicate,
    aggregate_results,
)
from src.experiment_utils import ExperimentConfig


class TestAggregateResults(unittest.TestCase):
    """Test result aggregation without full pipeline."""

    def test_aggregate_results_empty_list(self):
        """Verify aggregation handles empty replicate list."""
        metrics = aggregate_results([])
        self.assertEqual(metrics['n_replicates'], 0)
        self.assertIsInstance(metrics['motif_recovery'], dict)
        self.assertIsInstance(metrics['ude_recovery'], dict)

    def test_aggregate_results_structure(self):
        """Verify aggregation creates proper structure."""
        # Create mock replicate data
        mock_cohort = []
        mock_motif_res = {
            'recovery_metrics': {
                'P': {'r2': 0.8},
                'D': {'r2': 0.7},
                'h': {'r2': 0.75},
            },
            'classification_results': {
                'with_proxies': {'auroc': 0.85},
            }
        }
        mock_ude_res = {
            'recovery_metrics': {
                'P': {'r2': 0.85},
                'D': {'r2': 0.78},
                'h': {'r2': 0.80},
            },
            'classification_results': {
                'without_proxies': {'auroc': 0.82},
            }
        }
        replicates = [(mock_cohort, mock_motif_res, mock_ude_res)]

        metrics = aggregate_results(replicates)

        self.assertEqual(metrics['n_replicates'], 1)
        self.assertIn('P', metrics['motif_recovery'])
        self.assertIn('D', metrics['motif_recovery'])
        self.assertIn('h', metrics['motif_recovery'])
        self.assertIn('mean_r2', metrics['motif_recovery']['P'])
        self.assertIn('std_r2', metrics['motif_recovery']['P'])

    def test_aggregate_results_computes_statistics(self):
        """Verify aggregation computes mean and std correctly."""
        mock_cohort = []
        mock_motif_res = {
            'recovery_metrics': {
                'P': {'r2': 0.8},
                'D': {'r2': 0.7},
                'h': {'r2': 0.75},
            },
            'classification_results': {}
        }
        mock_ude_res = {
            'recovery_metrics': {
                'P': {'r2': 0.82},
                'D': {'r2': 0.72},
                'h': {'r2': 0.77},
            },
            'classification_results': {}
        }
        replicates = [
            (mock_cohort, mock_motif_res, mock_ude_res),
            (mock_cohort, mock_motif_res, mock_ude_res),
        ]

        metrics = aggregate_results(replicates)

        # Check MOTIF P recovery
        self.assertAlmostEqual(metrics['motif_recovery']['P']['mean_r2'], 0.8, places=5)
        self.assertAlmostEqual(metrics['motif_recovery']['P']['std_r2'], 0.0, places=5)


class TestExperimentRunner(unittest.TestCase):
    """Test experiment runner orchestration."""

    def setUp(self):
        """Create temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('src.run_experiment.run_single_replicate')
    def test_run_experiment_creates_output_dir(self, mock_replicate):
        """Verify run_experiment creates output directory."""
        # Mock replicate function
        mock_cohort = []
        mock_motif_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_ude_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_replicate.return_value = (mock_cohort, mock_motif_res, mock_ude_res)

        config = ExperimentConfig(
            experiment_name='test_exp',
            n_patients=10,
            n_replicates=1,
            output_dir=self.test_dir
        )
        output_dir = run_experiment(config, verbose=False)
        self.assertTrue(Path(output_dir).exists())

    @patch('src.run_experiment.run_single_replicate')
    def test_run_experiment_creates_metrics_file(self, mock_replicate):
        """Verify run_experiment creates metrics.json."""
        # Mock replicate function
        mock_cohort = []
        mock_motif_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_ude_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_replicate.return_value = (mock_cohort, mock_motif_res, mock_ude_res)

        config = ExperimentConfig(
            experiment_name='test_exp',
            n_patients=10,
            n_replicates=1,
            output_dir=self.test_dir
        )
        output_dir = run_experiment(config, verbose=False)

        metrics_file = Path(output_dir) / 'metrics.json'
        self.assertTrue(metrics_file.exists())

    @patch('src.run_experiment.run_single_replicate')
    def test_run_experiment_metrics_structure(self, mock_replicate):
        """Verify metrics.json has expected structure."""
        # Mock replicate function
        mock_cohort = []
        mock_motif_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_ude_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_replicate.return_value = (mock_cohort, mock_motif_res, mock_ude_res)

        config = ExperimentConfig(
            experiment_name='test_exp',
            n_patients=10,
            n_replicates=1,
            output_dir=self.test_dir
        )
        output_dir = run_experiment(config, verbose=False)

        metrics_file = Path(output_dir) / 'metrics.json'
        with open(metrics_file) as f:
            metrics = json.load(f)

        self.assertIn('config', metrics)
        self.assertIn('timestamp', metrics)
        self.assertIn('n_replicates', metrics)
        self.assertIn('motif_recovery', metrics)
        self.assertIn('ude_recovery', metrics)

    @patch('src.run_experiment.run_single_replicate')
    def test_run_experiment_returns_output_dir_path(self, mock_replicate):
        """Verify run_experiment returns output directory path."""
        # Mock replicate function
        mock_cohort = []
        mock_motif_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_ude_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_replicate.return_value = (mock_cohort, mock_motif_res, mock_ude_res)

        config = ExperimentConfig(
            experiment_name='test_exp',
            n_patients=10,
            n_replicates=1,
            output_dir=self.test_dir
        )
        output_dir = run_experiment(config, verbose=False)
        self.assertIsInstance(output_dir, str)
        self.assertTrue(Path(output_dir).exists())

    @patch('src.run_experiment.run_single_replicate')
    def test_run_experiment_saves_replicates(self, mock_replicate):
        """Verify run_experiment saves individual replicate files."""
        # Mock replicate function
        mock_cohort = []
        mock_motif_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_ude_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_replicate.return_value = (mock_cohort, mock_motif_res, mock_ude_res)

        config = ExperimentConfig(
            experiment_name='test_exp',
            n_patients=10,
            n_replicates=2,
            output_dir=self.test_dir
        )
        output_dir = run_experiment(config, verbose=False)

        # Check that replicate directories were created
        rep0_dir = Path(output_dir) / 'replicate_0'
        rep1_dir = Path(output_dir) / 'replicate_1'

        self.assertTrue(rep0_dir.exists())
        self.assertTrue(rep1_dir.exists())

        # Check that pickle files were created
        self.assertTrue((rep0_dir / 'cohort.pkl').exists())
        self.assertTrue((rep0_dir / 'motif_results.pkl').exists())
        self.assertTrue((rep0_dir / 'ude_results.pkl').exists())

    @patch('src.run_experiment.run_single_replicate')
    def test_run_experiment_config_serialization(self, mock_replicate):
        """Verify config is properly serialized in metrics.json."""
        # Mock replicate function
        mock_cohort = []
        mock_motif_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_ude_res = {
            'recovery_metrics': {},
            'classification_results': {},
        }
        mock_replicate.return_value = (mock_cohort, mock_motif_res, mock_ude_res)

        config = ExperimentConfig(
            experiment_name='test_exp',
            n_patients=50,
            n_replicates=1,
            output_dir=self.test_dir
        )
        output_dir = run_experiment(config, verbose=False)

        metrics_file = Path(output_dir) / 'metrics.json'
        with open(metrics_file) as f:
            metrics = json.load(f)

        self.assertEqual(metrics['config']['experiment_name'], 'test_exp')
        self.assertEqual(metrics['config']['n_patients'], 50)
        self.assertEqual(metrics['config']['n_replicates'], 1)


if __name__ == '__main__':
    unittest.main()
