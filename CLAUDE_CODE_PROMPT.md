# Claude Code Kickoff Prompt
## ODE–Multiomics Benchmark: MOTIF vs. UDE

---

## What You Are Building

You are implementing a Python research project that benchmarks two methods for
integrating mechanistic ODE models with multiomics data, using synthetic data
from a canonical inflammation model as ground truth.

**The two methods being compared:**

1. **MOTIF** (Funk, Bangs, Paterson 2025-26): Run a known ODE forward, treat
   synthetic state variable trajectories as proxy columns in a multiomics matrix,
   recover latent biology through correlation analysis.

2. **UDE + SINDy** (Rackauckas 2021 + Chang): Feed observed multiomics into a
   Universal Differential Equation with a neural network term, learn unknown rate
   functions from data, then recover interpretable equations via SINDy symbolic
   regression.

**The benchmark model:** Reynolds et al. (2006) acute inflammation — 6 ODEs,
published parameters, three bistable outcome trajectories.

**Why this exists:** These methods represent inverse approaches to the same problem
(ODE → data in MOTIF; data → ODE in UDE). Both claim to recover biological
insight from partial observations. This benchmark compares them quantitatively
under controlled synthetic conditions where ground truth is known.

---

## Full Specifications

Read these files before writing any code:

- `TECHNICAL_SPEC.md` — complete ODE equations, parameters, pipeline specs,
  evaluation metrics, implementation notes
- `EXPERIMENT_DESIGN.md` — all 6 experiments, figure specifications,
  statistical analysis plan
- `README.md` — project overview, directory structure, quickstart

Papers are in `papers/` — read them for biological and methodological context,
especially Reynolds 2006 for the ODE system.

---

## Implementation Order

Build in this exact order. Each step should be fully working and tested
before moving to the next.

### Step 1: Environment and Project Structure

Create the full directory structure from README.md. Then create:

```
environment.yml         ← conda environment
requirements.txt        ← pip requirements
pyproject.toml          ← package config (src layout)
src/__init__.py
```

Key dependencies:
```
numpy scipy matplotlib seaborn pandas scikit-learn pyyaml pytest
torch torchdiffeq pysindy jupyter
```

### Step 2: Reynolds ODE (`src/reynolds_ode.py`)

This is the foundation. Everything else depends on it.

Implement:
- `REYNOLDS_PARAMS` dict with all canonical parameters from TECHNICAL_SPEC §1.3
- `reynolds_ode(t, x, params)` — the 5-ODE system (h is derived as 1-f)
- `solve_reynolds(params, x0, t_eval, method='Radau')` — returns solution dict
  with keys: t, P, Nstar, D, CA, f, h
- `get_outcome(f_final)` — returns 'resolution', 'chronic', or 'death'
  based on f(t=72h)
- `plot_phase_portrait(ax)` — phase portrait for Figure 1A
- `plot_trajectories(solutions, ax)` — Figure 1B

Test: Verify that P₀=2, 7, 15 produce resolution, chronic, death respectively.
All three solutions must have `sol.success == True`.

**Critical:** Use `method='Radau'` in solve_ivp for stiffness. Check success flag.
Raise an informative ValueError if integration fails.

### Step 3: Synthetic Patient Generator (`src/synthetic_data.py`)

Implement:
- `generate_patient(P0, params_base, seed)` — single patient with parameter noise
- `generate_cohort(N=500, seed=42)` — full cohort, stratified by outcome
- `save_cohort(cohort, path)` — pickle + CSV summary
- `load_cohort(path)` — reload

Patient data structure (dict per patient):
```python
{
    'id': int,
    'P0': float,
    'outcome': str,
    'params': dict,
    # Observed (noisy, 6 timepoints)
    'obs_Nstar': np.ndarray,  # shape (6,)
    'obs_CA':    np.ndarray,
    'obs_f':     np.ndarray,
    'obs_t':     np.ndarray,  # [0, 6, 12, 24, 48, 72]
    # Ground truth (for evaluation only)
    'true_P':    np.ndarray,  # shape (6,) at obs timepoints
    'true_D':    np.ndarray,
    'true_h':    np.ndarray,
    # Full high-resolution trajectory
    'true_traj': np.ndarray,  # shape (6, 1000) for all state vars
    'true_t':    np.ndarray,  # shape (1000,)
}
```

Test: Generate N=500 cohort. Verify outcome distribution ~40/30/30.
Plot 3 example trajectories (one per outcome).

### Step 4: MOTIF Pipeline (`src/motif_pipeline.py`)

Implement in this sub-order:

**4a. Parameter calibration:**
```python
calibrate_patient(patient, params_init=None, n_restarts=3)
```
Fit ODE params to observed N*, CA, f using scipy.optimize.minimize (L-BFGS-B).
Optimise: k_pg, mu_p, s_nr, s_c, plus P0 (as free parameter).
Hold other params fixed. Return fitted_params, fitted_P0, fit_quality (R²).

**4b. Proxy generation:**
```python
generate_motif_proxies(patient, fitted_params, fitted_P0)
```
Run ODE with fitted params. Return dict of proxy trajectories and summary stats.

**4c. Feature extraction:**
```python
extract_features(patient, proxies=None)
```
Returns feature vector for one patient. If proxies=None, use observed only.
Features: AUC, peak, value at 72h, for each observed/proxy variable.

**4d. Correlation analysis:**
```python
motif_correlation_analysis(patients_with_proxies)
```
Compute Spearman correlation matrix between observed features and proxy features
across patient cohort. Return correlation matrix + p-values.

**4e. Outcome classification:**
```python
motif_classify_outcomes(train_patients, test_patients, use_proxies=True)
```
Train logistic regression. Return AUROC, F1, confusion matrix.

**4f. Full pipeline runner:**
```python
run_motif_pipeline(cohort, config)
```
Runs all steps; returns results dict.

Test: Run on N=50 patients. Verify R²(proxy_P, true_P) > 0.60 for baseline config.

### Step 5: Evaluation (`src/evaluation.py`)

Implement:
- `compute_recovery_metrics(pred, true, variable_name)` — R², RMSE, Spearman r
- `compute_classification_metrics(y_pred, y_true)` — AUROC, F1, confusion matrix
- `compare_pipelines(motif_results, ude_results)` — summary comparison table
- `run_robustness_experiment(experiment_fn, param_name, param_values, n_replicates)`

### Step 6: Plotting (`src/plotting.py`)

Implement all figures from EXPERIMENT_DESIGN.md §Figure Plan.
Each figure function should:
- Accept results dict as input
- Return a matplotlib Figure object
- Save as PDF and PNG to `figures/`
- Also save LinkedIn-format 1080×1080 PNG to `figures/linkedin/`

Use style: `plt.style.use('seaborn-v0_8-whitegrid')` with custom color palette:
```python
PALETTE = {
    'resolution': '#2ECC71',
    'chronic':    '#F39C12',
    'death':      '#E74C3C',
    'motif':      '#3498DB',
    'ude':        '#9B59B6',
    'truth':      '#2C3E50',
}
```

### Step 7: UDE Pipeline (`src/ude_pipeline.py`)

This is the most complex component. Implement incrementally:

**7a. Neural network:**
```python
class UDENet(nn.Module):
    # See TECHNICAL_SPEC §4.2
    # Input: [N*, CA, f] (3 dims)
    # Output: scalar clearance rate modifier (Softplus activation)
```

**7b. UDE ODE system:**
```python
ude_system(t, x, nn_model, known_params)
# See TECHNICAL_SPEC §4.3
# Known terms for N*, D, CA, f; learned term for P clearance
```

**7c. Training loop:**
```python
train_ude(patients, nn_model, known_params, config)
# See TECHNICAL_SPEC §4.4
# Use torchdiffeq.odeint with method='rk4'
# Loss on observed variables only (N*, CA, f)
# Gradient clipping: max_norm=1.0
```

**7d. SINDy extraction:**
```python
extract_sindy_equation(nn_model, state_range, sindy_config)
# See TECHNICAL_SPEC §4.5
# Evaluate NN on grid, fit SINDy, return model + equation string
```

**7e. Full pipeline runner:**
```python
run_ude_pipeline(cohort, config)
```

**Note on P₀ estimation in UDE:** Since P is unobserved, treat P₀ as a
learnable parameter per patient. Initialise at 5.0. This adds N_patients
scalar parameters to the optimisation — use a separate optimiser step or
include in the main loss with regularisation.

### Step 8: Experiment Runners

```python
# src/run_experiment.py
python -m src.run_experiment --config experiments/config_baseline.yaml
```

Each config YAML specifies:
```yaml
experiment_name: baseline
n_patients: 500
noise_sigma: 0.10
n_timepoints: 6
ode_n_vars: 6          # for misspecification experiment
n_replicates: 5
random_seed: 42
ude:
  n_epochs: 500
  lr: 0.001
  hidden_dim: 32
  batch_size: 32
sindy:
  degree: 2
  threshold: 0.05
output_dir: results/
```

### Step 9: Notebooks

Create 5 notebooks in `notebooks/`. Each should be self-contained and
runnable from top to bottom. Import from `src/` rather than re-implementing.

---

## Testing Requirements

Every `src/` module must have a corresponding `tests/test_*.py`.
Minimum test coverage:

- `test_reynolds_ode.py`: Test all three outcome trajectories, parameter
  sensitivity, that sol.success is True for all test cases
- `test_synthetic_data.py`: Test cohort generation, outcome distribution,
  noise level
- `test_motif_pipeline.py`: Test on 20-patient mini-cohort, verify R² > 0.5
- `test_ude_pipeline.py`: Test NN forward pass, ODE integration with NN term,
  short training run (10 epochs), SINDy on known function

Run with: `pytest tests/ -v`

---

## Git Commit Strategy

After each step:
```bash
git add -A
git commit -m "Step N: [component name] — [one-line summary]"
```

---

## What Success Looks Like

The project is complete when:

1. `pytest tests/ -v` passes all tests
2. `python -m src.run_experiment --config experiments/config_baseline.yaml`
   completes without errors and produces results in `results/`
3. All 6 figures in Figure Plan are generated in `figures/`
4. All 5 notebooks run top-to-bottom without errors
5. The comparison table in `metrics.json` shows:
   - MOTIF: R²(proxy_P, true_P) > 0.70 at baseline
   - UDE: θ RMSE < 0.15 at baseline
   - Both: AUROC > 0.80 at baseline

---

## Context: Why This Matters

This benchmark is not just a coding exercise. The scientific argument is:

- Hoffmann (2002) showed that ODE state variables predict multiomics outcomes
  (gene expression) from signal dynamics — proto-MOTIF
- Funk/Bangs/Paterson (2025-26) formalised this as MOTIF: run ODE forward,
  use synthetic variables as multiomics proxy columns
- Chang's UDE proposal inverts the pipeline: multiomics → encoder → ODE learning
- This benchmark is the first direct comparison of both approaches under
  controlled conditions with known ground truth

The key expected finding: the methods have complementary failure modes.
MOTIF fails when ODE is misspecified; UDE fails when data is sparse.
Both outperform pure data-driven approaches on downstream classification.
This is the publishable scientific contribution.
