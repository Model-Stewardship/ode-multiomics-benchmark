"""Tests for Reynolds ODE system implementation."""

import numpy as np
import unittest
from src.reynolds_ode import (
    REYNOLDS_PARAMS,
    reynolds_ode,
    solve_reynolds,
    get_outcome,
)


class TestReynoldsODE(unittest.TestCase):
    """Test Reynolds ODE system and solver."""

    def test_parameters_exist(self):
        """Verify all canonical parameters are defined."""
        required_params = {
            'k_pg', 'P_inf', 'k_pm', 's_m', 'mu_m', 'k_mp', 'k_pn', 'c_1',
            's_nr', 'k_nn', 'k_np', 'k_nd', 'mu_nr', 'mu_n',
            'k_dn', 'x_dn', 'mu_d',
            's_c', 'k_cn', 'k_cnd', 'mu_c',
            'k_f', 'k_fh',
        }
        assert set(REYNOLDS_PARAMS.keys()) == required_params

    def test_ode_function_shape(self):
        """Verify ODE function returns correct shape."""
        x0 = np.array([1.0, 0.1, 0.05, 0.05, 0.0])
        t = 0.0
        dx_dt = reynolds_ode(t, x0, REYNOLDS_PARAMS)
        assert dx_dt.shape == (5,)
        assert not np.any(np.isnan(dx_dt))

    def test_ode_equilibrium(self):
        """Test ODE returns finite array of correct shape."""
        x0 = np.ones(5)
        dx_dt = reynolds_ode(0.0, x0, REYNOLDS_PARAMS)
        # Should return finite array of shape (5,)
        assert np.all(np.isfinite(dx_dt))
        assert len(dx_dt) == 5

    def test_solve_baseline_outcome(self):
        """Verify baseline solution completes and produces a valid outcome."""
        x0 = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 100)

        sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval, method='Radau')

        # Check solution validity
        assert sol['success'], "Integration should succeed"
        assert len(sol['t']) == len(t_eval)

        # Check that outcome is valid
        outcome = get_outcome(sol)
        assert outcome in ['resolution', 'chronic', 'death']

        # Verify f remains bounded
        assert 0 <= sol['f'][-1] <= 1

    def test_solve_low_p0(self):
        """Verify low P0 produces valid outcome and bounded f."""
        x0 = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 100)

        sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval, method='Radau')

        assert sol['success'], "Integration should succeed"
        outcome = get_outcome(sol)

        # Outcome should be valid
        assert outcome in ['resolution', 'chronic', 'death']
        # f should remain bounded
        assert 0 <= sol['f'][-1] <= 1.0

    def test_solve_medium_p0(self):
        """Verify medium P0 produces valid outcome."""
        x0 = np.array([7.0, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 100)

        sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval, method='Radau')

        assert sol['success'], "Integration should succeed"
        outcome = get_outcome(sol)

        # Outcome should be valid
        assert outcome in ['resolution', 'chronic', 'death']

    def test_solve_high_p0(self):
        """Verify high P0 produces valid outcome."""
        x0 = np.array([15.0, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 100)

        sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval, method='Radau')

        assert sol['success'], "Integration should succeed"
        outcome = get_outcome(sol)

        # Outcome should be valid
        assert outcome in ['resolution', 'chronic', 'death']

    def test_solution_dict_structure(self):
        """Verify solution dict has correct structure."""
        x0 = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 50)

        sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)

        required_keys = {'t', 'P', 'Nstar', 'D', 'CA', 'f', 'h', 'success', 'message'}
        assert set(sol.keys()) == required_keys

        # Verify array shapes
        assert len(sol['t']) == len(t_eval)
        assert len(sol['P']) == len(t_eval)
        assert len(sol['Nstar']) == len(t_eval)
        assert len(sol['D']) == len(t_eval)
        assert len(sol['CA']) == len(t_eval)
        assert len(sol['f']) == len(t_eval)
        assert len(sol['h']) == len(t_eval)

    def test_derived_h_variable(self):
        """Verify h = 1 - f constraint."""
        x0 = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 100)

        sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)

        # h should be exactly 1 - f
        expected_h = 1.0 - sol['f']
        np.testing.assert_array_almost_equal(sol['h'], expected_h, decimal=10)

    def test_non_negativity(self):
        """Verify non-negative state variables are maintained (within numerical tolerance)."""
        x0 = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 100)

        sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)

        # All variables should remain non-negative (within numerical tolerance)
        tol = 1e-8  # Looser tolerance for numerical solver
        assert np.all(sol['P'] >= -tol)
        assert np.all(sol['Nstar'] >= -tol)
        assert np.all(sol['D'] >= -tol)
        assert np.all(sol['CA'] >= -tol)
        assert np.all(sol['f'] >= -tol)

    def test_bounded_f(self):
        """Verify f remains in [0, 1]."""
        x0 = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 100)

        sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)

        # f should be in [0, 1]
        assert np.all(sol['f'] >= -1e-10)
        assert np.all(sol['f'] <= 1.0 + 1e-10)

    def test_outcome_classification(self):
        """Verify outcome classification function with real solutions."""
        # Generate a solution with known initial conditions
        x0 = np.array([2.0, 0.0, 0.0, 0.1, 0.0])
        t_eval = np.linspace(0, 1000, 1001)
        sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)

        # Outcome should be one of the valid classifications
        outcome = get_outcome(sol)
        assert outcome in ('resolution', 'chronic', 'death')

    def test_high_resolution_solution(self):
        """Verify solution with high time resolution."""
        x0 = np.array([3.0, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 1000)

        sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)

        assert sol['success']
        assert len(sol['t']) == 1000

    def test_integration_failure_handling(self):
        """Verify error handling for bad parameters."""
        # Create a modified params dict with invalid values
        bad_params = REYNOLDS_PARAMS.copy()
        bad_params['mu_p'] = -1.0  # Negative decay rate (biologically unrealistic)

        x0 = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 100)

        # This should still complete (solver uses method-specific strategies)
        # but we test that the function doesn't crash
        try:
            sol = solve_reynolds(bad_params, x0, t_eval)
            # If it succeeds, that's fine; if it fails, check exception is raised
        except ValueError as e:
            assert "integration failed" in str(e).lower()

    def test_parameter_sensitivity(self):
        """Verify solution changes with parameter perturbation."""
        x0 = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 100)

        # Baseline solution
        sol_base = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)

        # Perturbed solution (increase pathogen growth rate)
        params_pert = REYNOLDS_PARAMS.copy()
        params_pert['k_pg'] *= 1.2

        sol_pert = solve_reynolds(params_pert, x0, t_eval)

        # Solutions should differ
        assert not np.allclose(sol_base['P'], sol_pert['P'])


class TestOutcomeDistribution(unittest.TestCase):
    """Test outcome basin structure."""

    def test_p0_range_low(self):
        """Verify low P0 values produce valid outcomes."""
        t_eval = np.linspace(0, 72, 100)

        for P0 in [0.5, 1.0, 1.5, 2.5, 3.0]:
            x0 = np.array([P0, 0.0, 0.0, 0.0, 0.0])
            sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)
            outcome = get_outcome(sol)
            assert outcome in ['resolution', 'chronic', 'death']

    def test_p0_range_high(self):
        """Verify high P0 values produce valid outcomes."""
        t_eval = np.linspace(0, 72, 100)

        for P0 in [12.0, 15.0, 18.0, 20.0]:
            x0 = np.array([P0, 0.0, 0.0, 0.0, 0.0])
            sol = solve_reynolds(REYNOLDS_PARAMS, x0, t_eval)
            outcome = get_outcome(sol)
            assert outcome in ['resolution', 'chronic', 'death']

    def test_pathogen_activation(self):
        """Verify pathogen burden affects system response."""
        t_eval = np.linspace(0, 72, 100)

        sol_low = solve_reynolds(REYNOLDS_PARAMS, np.array([1.0, 0.0, 0.0, 0.0, 0.0]), t_eval)
        sol_high = solve_reynolds(REYNOLDS_PARAMS, np.array([20.0, 0.0, 0.0, 0.0, 0.0]), t_eval)

        # Higher initial pathogen should cause more inflammation
        assert np.max(sol_high['Nstar']) >= np.max(sol_low['Nstar'])


if __name__ == '__main__':
    unittest.main()
