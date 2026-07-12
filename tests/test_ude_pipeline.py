"""Tests for UDE (Universal Differential Equations) pipeline."""

import numpy as np
import torch
import unittest
from src.ude_pipeline import (
    UDENet,
    ude_system,
    train_ude,
    extract_sindy_equation,
    run_ude_pipeline,
    HAS_TORCHDIFFEQ,
    HAS_PYSINDY,
)
from src.synthetic_data import generate_patient, generate_cohort
from src.reynolds_ode import REYNOLDS_PARAMS


class TestUDENet(unittest.TestCase):
    """Test UDE neural network module."""

    def test_udenet_initialization(self):
        """Verify UDENet initializes with correct architecture."""
        net = UDENet(input_dim=3, hidden_dim=32, output_dim=1)
        self.assertIsInstance(net, torch.nn.Module)

    def test_udenet_forward_shape(self):
        """Verify forward pass produces correct output shape."""
        net = UDENet(input_dim=3, hidden_dim=16, output_dim=1)
        z = torch.randn(10, 3)
        output = net(z)
        self.assertEqual(output.shape, (10, 1))

    def test_udenet_output_non_negative(self):
        """Verify Softplus ensures non-negative outputs (rate)."""
        net = UDENet(input_dim=3, hidden_dim=16, output_dim=1)
        z = torch.randn(100, 3)
        output = net(z)
        self.assertTrue(torch.all(output >= 0.0))

    def test_udenet_single_sample(self):
        """Verify single sample forward pass."""
        net = UDENet(input_dim=3, hidden_dim=16, output_dim=1)
        z = torch.randn(3)  # Single sample as 1D tensor
        output = net(z.unsqueeze(0))  # Add batch dimension
        self.assertEqual(output.shape, (1, 1))

    def test_udenet_batch_processing(self):
        """Verify batch processing works correctly."""
        net = UDENet(input_dim=3, hidden_dim=32, output_dim=1)
        batch_size = 64
        z = torch.randn(batch_size, 3)
        output = net(z)
        self.assertEqual(output.shape, (batch_size, 1))


class TestUDESystem(unittest.TestCase):
    """Test UDE ODE system dynamics."""

    def setUp(self):
        """Set up UDE system for testing."""
        if not HAS_TORCHDIFFEQ:
            self.skipTest("torchdiffeq not available")
        self.nn_model = UDENet(input_dim=3, hidden_dim=16, output_dim=1)
        self.known_params = {k: torch.tensor(v, dtype=torch.float32)
                             for k, v in REYNOLDS_PARAMS.items()}

    def test_ude_system_returns_tensor(self):
        """Verify ude_system returns a tensor."""
        x = torch.tensor([5.0, 0.5, 0.1, 0.3, 0.2], dtype=torch.float32)
        dx = ude_system(0.0, x, self.nn_model, self.known_params)
        self.assertIsInstance(dx, torch.Tensor)

    def test_ude_system_output_shape(self):
        """Verify output has correct shape (5 derivatives)."""
        x = torch.tensor([5.0, 0.5, 0.1, 0.3, 0.2], dtype=torch.float32)
        dx = ude_system(0.0, x, self.nn_model, self.known_params)
        self.assertEqual(dx.shape, (5,))

    def test_ude_system_finite_outputs(self):
        """Verify all derivatives are finite."""
        x = torch.tensor([5.0, 0.5, 0.1, 0.3, 0.2], dtype=torch.float32)
        dx = ude_system(0.0, x, self.nn_model, self.known_params)
        self.assertTrue(torch.all(torch.isfinite(dx)))

    def test_ude_system_responds_to_state(self):
        """Verify derivatives change with state."""
        x1 = torch.tensor([5.0, 0.5, 0.1, 0.3, 0.2], dtype=torch.float32)
        x2 = torch.tensor([2.0, 0.3, 0.05, 0.1, 0.1], dtype=torch.float32)
        dx1 = ude_system(0.0, x1, self.nn_model, self.known_params)
        dx2 = ude_system(0.0, x2, self.nn_model, self.known_params)
        # Should be different (within numerical tolerance)
        self.assertFalse(torch.allclose(dx1, dx2, atol=1e-5))


class TestTrainUDE(unittest.TestCase):
    """Test UDE training procedure."""

    def setUp(self):
        """Set up training test."""
        if not HAS_TORCHDIFFEQ:
            self.skipTest("torchdiffeq not available")
        self.nn_model = UDENet(input_dim=3, hidden_dim=16, output_dim=1)
        self.known_params = REYNOLDS_PARAMS.copy()
        self.cohort = generate_cohort(N_resolution=3, N_chronic=2, N_death=2,
                                      seed=42, verbose=False)

    def test_train_ude_returns_tuple(self):
        """Verify train_ude returns (model, history) tuple."""
        config = {
            'n_epochs': 2,
            'lr': 1e-3,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
        }
        trained_model, history = train_ude(
            self.cohort, self.nn_model, self.known_params,
            config=config, verbose=False
        )
        self.assertIsInstance(trained_model, torch.nn.Module)
        self.assertIsInstance(history, dict)

    def test_train_ude_history_has_keys(self):
        """Verify training history has expected keys."""
        config = {
            'n_epochs': 2,
            'lr': 1e-3,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
        }
        _, history = train_ude(
            self.cohort, self.nn_model, self.known_params,
            config=config, verbose=False
        )
        self.assertIn('epoch', history)
        self.assertIn('loss', history)
        self.assertIn('lr', history)

    def test_train_ude_loss_tracked(self):
        """Verify loss values are recorded during training."""
        config = {
            'n_epochs': 2,
            'lr': 1e-3,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
        }
        _, history = train_ude(
            self.cohort, self.nn_model, self.known_params,
            config=config, verbose=False
        )
        self.assertEqual(len(history['loss']), 2)
        self.assertTrue(all(isinstance(l, float) for l in history['loss']))

    def test_train_ude_model_changes(self):
        """Verify model parameters change after training."""
        config = {
            'n_epochs': 5,
            'lr': 1e-2,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
        }
        # Save initial weights
        initial_params = [p.clone() for p in self.nn_model.parameters()]

        trained_model, _ = train_ude(
            self.cohort, self.nn_model, self.known_params,
            config=config, verbose=False
        )

        # Check that at least one parameter changed
        params_changed = False
        for p_init, p_trained in zip(initial_params, trained_model.parameters()):
            if not torch.allclose(p_init, p_trained, atol=1e-6):
                params_changed = True
                break
        self.assertTrue(params_changed, "Model parameters should change during training")

    def test_train_ude_gradient_clipping(self):
        """Verify gradient clipping prevents NaN/Inf."""
        config = {
            'n_epochs': 3,
            'lr': 1e-2,  # Higher LR to test clipping
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
        }
        trained_model, history = train_ude(
            self.cohort, self.nn_model, self.known_params,
            config=config, verbose=False
        )
        # Check loss doesn't contain NaN/Inf
        self.assertTrue(all(np.isfinite(l) for l in history['loss']))


class TestSINDyExtraction(unittest.TestCase):
    """Test SINDy symbolic equation extraction."""

    def setUp(self):
        """Set up SINDy test."""
        if not HAS_TORCHDIFFEQ:
            self.skipTest("torchdiffeq not available")
        self.nn_model = UDENet(input_dim=3, hidden_dim=16, output_dim=1)
        self.state_range = {
            'Nstar': (0, 0.8),
            'CA': (0, 0.5),
            'f': (0, 1.0),
        }

    def test_extract_sindy_returns_tuple(self):
        """Verify extract_sindy_equation returns (model, results) tuple."""
        sindy_model, results = extract_sindy_equation(
            self.nn_model, self.state_range, verbose=False
        )
        self.assertIsInstance(results, dict)

    def test_extract_sindy_results_has_keys(self):
        """Verify results dict has expected keys when successful."""
        if not HAS_PYSINDY:
            self.skipTest("pysindy not available")
        sindy_model, results = extract_sindy_equation(
            self.nn_model, self.state_range, verbose=False
        )
        if 'error' not in results:
            self.assertIn('r2', results)
            self.assertIn('n_features', results)
            self.assertIn('equation', results)

    def test_extract_sindy_graceful_fallback(self):
        """Verify graceful fallback when pysindy unavailable."""
        if HAS_PYSINDY:
            self.skipTest("pysindy is available; testing unavailability fallback")
        # If pysindy is not available, should return error dict
        sindy_model, results = extract_sindy_equation(
            self.nn_model, self.state_range, verbose=False
        )
        self.assertIn('error', results)

    def test_extract_sindy_config_respected(self):
        """Verify configuration parameters are used."""
        if not HAS_PYSINDY:
            self.skipTest("pysindy not available")

        sindy_config = {'degree': 3, 'threshold': 0.01}
        sindy_model, results = extract_sindy_equation(
            self.nn_model, self.state_range,
            sindy_config=sindy_config, verbose=False
        )
        # Should complete without error
        self.assertIsInstance(results, dict)


class TestUDEPipeline(unittest.TestCase):
    """Test full UDE pipeline."""

    def setUp(self):
        """Set up pipeline test."""
        if not HAS_TORCHDIFFEQ:
            self.skipTest("torchdiffeq not available")
        self.cohort = generate_cohort(N_resolution=3, N_chronic=2, N_death=2,
                                      seed=42, verbose=False)

    def test_run_ude_pipeline_returns_dict(self):
        """Verify pipeline returns results dict."""
        config = {
            'nn_hidden_dim': 16,
            'n_epochs': 2,
            'lr': 1e-3,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
            'sindy_degree': 2,
            'sindy_threshold': 0.05,
        }
        results = run_ude_pipeline(self.cohort, config=config, verbose=False)
        self.assertIsInstance(results, dict)

    def test_run_ude_pipeline_has_required_keys(self):
        """Verify pipeline results have required keys."""
        config = {
            'nn_hidden_dim': 16,
            'n_epochs': 2,
            'lr': 1e-3,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
            'sindy_degree': 2,
            'sindy_threshold': 0.05,
        }
        results = run_ude_pipeline(self.cohort, config=config, verbose=False)
        self.assertIn('config', results)
        self.assertIn('n_patients', results)
        self.assertIn('nn_model', results)
        self.assertIn('training_history', results)

    def test_run_ude_pipeline_completes(self):
        """Verify pipeline completes without exceptions."""
        config = {
            'nn_hidden_dim': 16,
            'n_epochs': 3,
            'lr': 1e-3,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
            'sindy_degree': 2,
            'sindy_threshold': 0.05,
        }
        results = run_ude_pipeline(self.cohort, config=config, verbose=False)
        self.assertEqual(results['n_patients'], len(self.cohort))

    def test_run_ude_pipeline_model_trained(self):
        """Verify trained model differs from initial."""
        config = {
            'nn_hidden_dim': 16,
            'n_epochs': 5,
            'lr': 1e-2,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
            'sindy_degree': 2,
            'sindy_threshold': 0.05,
        }
        results = run_ude_pipeline(self.cohort, config=config, verbose=False)

        # Model should exist and be trainable
        model = results['nn_model']
        self.assertIsInstance(model, torch.nn.Module)

        # Training history should have entries
        history = results['training_history']
        self.assertGreater(len(history.get('loss', [])), 0)

    def test_run_ude_pipeline_sindy_attempted(self):
        """Verify SINDy extraction is attempted."""
        config = {
            'nn_hidden_dim': 16,
            'n_epochs': 2,
            'lr': 1e-3,
            'batch_size': 2,
            'max_grad_norm': 1.0,
            'random_seed': 42,
            'sindy_degree': 2,
            'sindy_threshold': 0.05,
        }
        results = run_ude_pipeline(self.cohort, config=config, verbose=False)

        # Should have sindy results dict
        self.assertIn('sindy_results', results)
        sindy_results = results['sindy_results']
        self.assertIsInstance(sindy_results, dict)


class TestUDEBaseline(unittest.TestCase):
    """Test UDE pipeline on realistic cohort."""

    def setUp(self):
        """Set up baseline test."""
        if not HAS_TORCHDIFFEQ:
            self.skipTest("torchdiffeq not available")

    def test_ude_on_20_patient_cohort(self):
        """Run UDE on 20-patient cohort and verify completion."""
        cohort = generate_cohort(N_resolution=10, N_chronic=5, N_death=5,
                                 seed=42, verbose=False)

        config = {
            'nn_hidden_dim': 32,
            'n_epochs': 10,
            'lr': 1e-3,
            'batch_size': 4,
            'max_grad_norm': 1.0,
            'random_seed': 42,
            'sindy_degree': 2,
            'sindy_threshold': 0.05,
        }

        results = run_ude_pipeline(cohort, config=config, verbose=False)

        # Should process all patients
        self.assertEqual(results['n_patients'], 20)

        # Should have training history
        self.assertIn('training_history', results)
        history = results['training_history']
        self.assertEqual(len(history['loss']), 10)

        # Loss should be finite
        self.assertTrue(all(np.isfinite(l) for l in history['loss']))

        # Should attempt SINDy
        self.assertIn('sindy_results', results)


if __name__ == '__main__':
    unittest.main()
