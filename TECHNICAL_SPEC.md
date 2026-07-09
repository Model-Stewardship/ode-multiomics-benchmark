# Technical Specification: ODE–Multiomics Benchmark

## 1. The Reynolds 2006 ODE System

### 1.1 State Variables

| Symbol | Variable | Biological Analog | Units |
|---|---|---|---|
| P | Pathogen burden | Bacterial load | dimensionless (normalized) |
| N* | Activated neutrophils / early pro-inflammatory mediator | IL-6, TNF-α | dimensionless |
| D | Late pro-inflammatory / damage-associated mediator | DAMPs, HMGB1 | dimensionless |
| CA | Anti-inflammatory mediator | IL-10, TGF-β | dimensionless |
| f | Tissue damage fraction | Organ failure score, lactate | [0, 1] |
| h | Tissue health (h = 1 − f) | Complement to damage | [0, 1] |

Note: h is a derived variable (h = 1 − f) and need not be solved as an independent ODE.
Reynolds includes it for biological clarity. Implement as 5 ODEs + 1 algebraic relation.

### 1.2 ODE System

```
dP/dt  = k_pg * P * (1 - P/P_inf) - k_pm * (N* / (s_dm + N*)) * P - μ_p * P

dN*/dt = s_nr / (1 + (CA/ε_nr)^2) * (P/(s_nm + P) + D/(s_nd + D)) * (1 - N*/N*_inf)
         - μ_nr * N*

dD/dt  = k_dn * N* + k_df * f - μ_d * D

dCA/dt = s_c * (N*^2) / (s_cn^2 + N*^2) - μ_c * CA

df/dt  = k_f * N* * (1 - f) - k_fh * h * f

h      = 1 - f
```

### 1.3 Canonical Parameters (Reynolds 2006 Table 1)

These are the ground-truth parameters used to generate synthetic data.
All values taken directly from Reynolds et al. (2006) J Theor Biol 242:220–36.

```python
REYNOLDS_PARAMS = {
    # Pathogen dynamics
    'k_pg':    0.5,      # pathogen growth rate
    'P_inf':   20.0,     # pathogen carrying capacity
    'k_pm':    1.8,      # max pathogen clearance rate by N*
    's_dm':    0.05,     # half-saturation of N* clearance of pathogen
    'mu_p':    0.2,      # pathogen natural decay

    # Early pro-inflammatory mediator (N*)
    's_nr':    0.08,     # max production rate of N*
    'epsilon_nr': 0.32,  # anti-inflammatory suppression threshold (CA)
    's_nm':    0.1,      # half-saturation pathogen activation of N*
    's_nd':    0.1,      # half-saturation damage activation of N*
    'N_inf':   0.6,      # N* carrying capacity (max activation)
    'mu_nr':   0.12,     # N* decay rate

    # Late pro-inflammatory / DAMP mediator (D)
    'k_dn':    0.02,     # N*-driven DAMP production
    'k_df':    0.02,     # tissue-damage-driven DAMP production
    'mu_d':    0.05,     # D decay rate

    # Anti-inflammatory mediator (CA)
    's_c':     0.04,     # max CA production rate
    's_cn':    0.08,     # half-saturation of N* driving CA
    'mu_c':    0.1,      # CA decay rate

    # Tissue damage
    'k_f':     0.01,     # N*-driven tissue damage rate
    'k_fh':    0.12,     # tissue repair rate (health-dependent)
}
```

### 1.4 Bistability and Outcome Basins

The Reynolds model has three stable steady states depending on initial pathogen load P₀:

| Outcome | P₀ range (approx) | Final state | Biological meaning |
|---|---|---|---|
| Resolution | P₀ < 4 | P→0, f→0, h→1 | Infection cleared, full recovery |
| Chronic inflammation | 4 ≤ P₀ ≤ 10 | P~low, f~moderate | Persistent low-grade inflammation |
| Death | P₀ > 10 | f→1, h→0 | Overwhelming sepsis, organ failure |

These thresholds are approximate and depend on other initial conditions.
Use numerical simulation to confirm basin membership for each virtual patient.

---

## 2. Synthetic Patient Generation

### 2.1 Design Principles

Generate N=500 virtual patients total, stratified across the three outcome basins.
Each patient is defined by:
- Initial conditions vector `x0 = [P₀, N*₀, D₀, CA₀, f₀]`
- A small per-patient parameter perturbation (biological variability)
- Simulated at timepoints `t = [0, 6, 12, 24, 48, 72]` hours

### 2.2 Initial Condition Sampling

```python
# Stratify patients across outcome basins
N_resolution  = 200  # P0 sampled from Uniform(0.5, 3.5)
N_chronic     = 150  # P0 sampled from Uniform(4.0, 9.0)
N_death       = 150  # P0 sampled from Uniform(11.0, 20.0)

# Non-pathogen initial conditions (all patients)
N*_0 = 0.0      # no pre-existing inflammation
D_0  = 0.0
CA_0 = 0.0
f_0  = 0.0      # no prior tissue damage
```

### 2.3 Parameter Variability

Apply ±15% lognormal noise to each parameter independently per patient:

```python
patient_params = {k: v * np.exp(np.random.normal(0, 0.15))
                  for k, v in REYNOLDS_PARAMS.items()}
```

This creates realistic inter-patient heterogeneity while preserving the
qualitative basin structure.

### 2.4 Measurement Noise

Add proportional Gaussian noise to all observed quantities after simulation:

```python
# Observation noise: 10% coefficient of variation
observed = true_value * (1 + np.random.normal(0, 0.10))
observed = np.clip(observed, 0, None)  # non-negative constraint
```

### 2.5 Observation Model (The "Multiomics" Setup)

**Observed variables** (available to both pipelines):
- `N*` — early pro-inflammatory mediator (analog: IL-6 / TNF plasma proteomics)
- `CA` — anti-inflammatory mediator (analog: IL-10 plasma proteomics)
- `f`  — tissue damage fraction (analog: organ failure score / lactate)

**Latent variables** (unobserved; ground truth used for evaluation only):
- `P`  — pathogen burden (often unmeasured in clinical settings)
- `D`  — late DAMP mediator (not directly measurable)
- `h`  — tissue health (derived; 1 − f)

This gives each patient a 3 × 6 observation matrix (3 variables × 6 timepoints).
Flatten to an 18-dimensional feature vector for cross-sectional analyses,
or keep as a time series for trajectory-based methods.

### 2.6 Output Data Format

```python
# Per-patient data structure
patient = {
    'id': int,
    'outcome': str,              # 'resolution', 'chronic', 'death'
    'P0': float,                 # initial pathogen load
    'params': dict,              # per-patient parameters
    # Observed (noisy)
    'obs_Nstar': np.ndarray,     # shape (6,) — timepoints
    'obs_CA':    np.ndarray,
    'obs_f':     np.ndarray,
    # Latent (ground truth — for evaluation only)
    'true_P':    np.ndarray,
    'true_D':    np.ndarray,
    'true_h':    np.ndarray,
    # Full trajectories (ground truth — 1000 timepoints)
    'true_traj': np.ndarray,     # shape (6, 1000) — all 6 state variables
    'true_t':    np.ndarray,     # shape (1000,)
}
```

---

## 3. MOTIF Pipeline Specification

### 3.1 Conceptual Basis

MOTIF (Multi-Omic Twin-Inferred Function) — from Funk, Bangs, Paterson (2025-26):
1. Run the ODE forward for each patient using estimated parameters
2. Generate synthetic trajectories for ALL state variables (including latent ones)
3. Treat synthetic state variable trajectories as proxy columns alongside real data
4. Compute correlation structure between real observed variables and synthetic proxies
5. Use proxy–data correlations to recover biological signal and classify outcomes

### 3.2 ODE Parameter Estimation (Calibration Step)

Before generating synthetic proxies, calibrate ODE parameters to each patient's
observed data using scipy.optimize.minimize with L-BFGS-B:

```python
def calibrate_patient(patient_obs, params_init=REYNOLDS_PARAMS):
    """
    Fit ODE parameters to observed N*, CA, f trajectories.
    Optimise over a subset of parameters (k_pg, mu_p, s_nr, s_c)
    while holding others fixed to canonical values.
    """
    def residual(log_params):
        params = decode_params(log_params)
        sim = solve_reynolds(params, patient_obs['P0_guess'], t_obs)
        obs_vars = ['Nstar', 'CA', 'f']
        return sum_of_squared_residuals(sim, patient_obs, obs_vars)

    result = minimize(residual, encode_params(params_init), method='L-BFGS-B')
    return decode_params(result.x)
```

Note: P₀ is also estimated during calibration since it is unobserved.
Use the canonical value as initialisation and allow it to vary.

### 3.3 Synthetic Proxy Generation

After calibration, run the ODE forward with fitted parameters and extract
synthetic trajectories for the latent variables:

```python
def generate_motif_proxies(patient, fitted_params):
    """
    Returns synthetic proxy columns for P, D, h.
    """
    sol = solve_reynolds(fitted_params, fitted_P0, t_fine)
    return {
        'proxy_P': sol['P'],          # pathogen burden proxy
        'proxy_D': sol['D'],          # DAMP proxy
        'proxy_h': sol['h'],          # tissue health proxy
        # Derived summary statistics
        'proxy_P_auc':  auc(sol['P']),
        'proxy_P_peak': max(sol['P']),
        'proxy_D_auc':  auc(sol['D']),
        'proxy_h_min':  min(sol['h']),
    }
```

### 3.4 Correlation Analysis (Proxy–Multiomics Integration)

Compute Spearman correlations between proxy summary statistics and observed
multiomics features across the patient cohort:

```python
# Feature matrix: N_patients × N_features
# Real features: obs_Nstar_auc, obs_CA_auc, obs_f_max, obs_Nstar_peak, ...
# Proxy features: proxy_P_auc, proxy_D_auc, proxy_h_min, ...

correlation_matrix = spearmanr(real_features, proxy_features)
```

### 3.5 Outcome Classification Using Proxies

Train a logistic regression on proxy features to predict outcome:

```python
X_train = np.hstack([real_features, proxy_features])
y_train = outcome_labels  # 0=resolution, 1=chronic, 2=death

clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
clf.fit(X_train, y_train)
```

Evaluate:
- Without proxies (real features only)
- With proxies added (real + synthetic)
- On held-out 20% test set

### 3.6 Misspecification Experiment

Test MOTIF under model misspecification by removing the D variable from the ODE
(use 5-variable instead of 6-variable model for proxy generation):

```python
MISSPECIFIED_PARAMS = {k: v for k, v in REYNOLDS_PARAMS.items()
                        if k not in ['k_dn', 'k_df', 'mu_d']}
# Run same pipeline but with 5-ODE model
# Measure degradation in proxy–truth correlation
```

---

## 4. UDE + SINDy Pipeline Specification

### 4.1 Conceptual Basis

Universal Differential Equations (Rackauckas et al. 2021) augment a known ODE
with a neural network term that learns unknown biological functions from data.

Architecture for this benchmark:

```
Known ODE terms:          dN*/dt = [known N* dynamics] + θ(N*, CA, f) * correction
Unknown term:             θ(z, t) — MLP that modifies pathogen clearance rate
                          using only observed variables z = [N*, CA, f]

Post-training:            SINDy symbolic regression on θ(z) outputs
                          to recover h(z) = interpretable rate equation
```

### 4.2 UDE Architecture

The neural network term θ represents the unknown pathogen dynamics
conditioned on observed inflammatory state:

```python
import torch
import torch.nn as nn

class UDENet(nn.Module):
    """
    Neural network term for the UDE.
    Input:  observed state z = [N*, CA, f] at time t
    Output: correction to pathogen clearance rate
    """
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # ensure non-negative output (clearance rate)
        )

    def forward(self, z):
        return self.net(z)
```

### 4.3 UDE ODE System

The UDE replaces the pathogen clearance term with the learned network:

```python
def ude_system(t, x, nn_model, known_params):
    """
    UDE: known ODE terms for N*, D, CA, f
         plus neural network term for pathogen clearance
    x = [P, N*, D, CA, f]  (P is latent but included in ODE)
    obs = [N*, CA, f]       (what we feed to the network)
    """
    P, Nstar, D, CA, f = x
    p = known_params

    # Observed state fed to neural network
    z = torch.tensor([Nstar, CA, f], dtype=torch.float32)
    nn_clearance = nn_model(z).item()

    # Pathogen dynamics (partially learned)
    dP = (p['k_pg'] * P * (1 - P / p['P_inf'])
          - nn_clearance * P         # ← learned term
          - p['mu_p'] * P)

    # All other terms remain as known biology
    dNstar = (p['s_nr'] / (1 + (CA / p['epsilon_nr'])**2)
              * (P / (p['s_nm'] + P) + D / (p['s_nd'] + D))
              * (1 - Nstar / p['N_inf'])
              - p['mu_nr'] * Nstar)

    dD = p['k_dn'] * Nstar + p['k_df'] * f - p['mu_d'] * D

    dCA = p['s_c'] * Nstar**2 / (p['s_cn']**2 + Nstar**2) - p['mu_c'] * CA

    df = p['k_f'] * Nstar * (1 - f) - p['k_fh'] * (1 - f) * f

    return [dP, dNstar, dD, dCA, df]
```

### 4.4 UDE Training

Use `torchdiffeq.odeint` for differentiable ODE solving. Train on observed
trajectories of N*, CA, f only (P and D are not used in the loss):

```python
from torchdiffeq import odeint

def train_ude(patients, nn_model, known_params, n_epochs=500, lr=1e-3):
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for patient in patients:
            x0 = torch.tensor(patient['x0_guess'])  # includes P0 estimate
            t_obs = torch.tensor([0, 6, 12, 24, 48, 72], dtype=torch.float32)

            # Solve UDE forward
            pred_traj = odeint(
                lambda t, x: ude_system(t, x, nn_model, known_params),
                x0, t_obs, method='rk4'
            )

            # Loss on observed variables only
            obs_pred = pred_traj[:, [1, 3, 4]]  # N*, CA, f indices
            obs_true = patient['obs_tensor']     # shape (6, 3)

            loss = F.mse_loss(obs_pred, obs_true)
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: loss = {total_loss.item():.4f}")
```

### 4.5 SINDy Symbolic Extraction

After UDE training, evaluate the learned θ(z) on a dense grid of states
and apply SINDy to recover a symbolic rate equation:

```python
import pysindy as ps

def extract_sindy_equation(nn_model, state_grid):
    """
    state_grid: array of shape (N_grid, 3) covering
                the physiologically relevant range of [N*, CA, f]
    """
    # Evaluate neural network on grid
    with torch.no_grad():
        z_tensor = torch.tensor(state_grid, dtype=torch.float32)
        nn_outputs = nn_model(z_tensor).numpy()

    # SINDy: find sparse polynomial basis representation
    feature_names = ['Nstar', 'CA', 'f']
    lib = ps.PolynomialLibrary(degree=3, include_interaction=True)

    sindy_model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0.01),  # sparsity threshold
        feature_library=lib,
        feature_names=feature_names
    )
    sindy_model.fit(state_grid, x_dot=nn_outputs)
    sindy_model.print()

    return sindy_model
```

**Expected output:** A sparse polynomial in [N*, CA, f] that approximates
the true clearance function: `θ(z) ≈ k_pm * N* / (s_dm + N*)` — a
Michaelis-Menten form.

---

## 5. Evaluation Metrics

### 5.1 Primary Recovery Metrics

For each pipeline, compute:

**Proxy recovery accuracy (MOTIF):**
```
R²(proxy_P, true_P)    — correlation between MOTIF P proxy and ground truth P
R²(proxy_D, true_D)    — correlation between MOTIF D proxy and ground truth D
R²(proxy_h, true_h)    — correlation between MOTIF h proxy and ground truth h
```

**Rate function recovery accuracy (UDE + SINDy):**
```
RMSE(θ_sindy(z), θ_true(z))    — root mean squared error of recovered
                                    clearance function vs. true function
R²(θ_sindy, θ_true)            — coefficient of determination
coefficient_sparsity            — fraction of zero coefficients in SINDy output
                                    (higher = simpler recovered equation)
```

**Outcome classification (both pipelines):**
```
AUROC (macro-averaged, 3-class)
F1 (macro-averaged)
Confusion matrix
```

### 5.2 Robustness Experiments

Run each pipeline under degrading conditions and report primary metrics:

| Experiment | Varied parameter | Values tested |
|---|---|---|
| Patient count | N_patients | 500, 200, 100, 50, 20 |
| Noise level | sigma_obs | 0.05, 0.10, 0.20, 0.35 |
| Sampling frequency | n_timepoints | 6, 4, 3, 2 |
| ODE misspecification (MOTIF) | n_vars_in_model | 6, 5, 4 |
| NN capacity (UDE) | hidden_dim | 64, 32, 16, 8 |

### 5.3 Complementary Failure Mode Analysis

The key scientific finding to report:

```
MOTIF strengths:   correct ODE structure, sparse data, low computational cost
MOTIF weaknesses:  degrades with ODE misspecification

UDE strengths:     robust to ODE misspecification, learns unknown terms
UDE weaknesses:    needs more data, computationally expensive, SINDy may fail
                   to converge to correct symbolic form in high noise
```

---

## 6. Figure Specifications

All figures should be 300 DPI, saved as both PDF (publication) and PNG (web).

| Figure | Content | Key panels |
|---|---|---|
| Fig 1 | Reynolds model and synthetic data overview | Phase portrait, 3 outcome trajectories, observation model schematic |
| Fig 2 | Virtual patient cohort | Distribution of P₀, outcome breakdown, example patient trajectories |
| Fig 3 | MOTIF pipeline results | Proxy vs. ground truth scatter (P, D, h), correlation heatmap, classification improvement |
| Fig 4 | UDE + SINDy results | Training loss curves, SINDy recovered equation vs. true function, RMSE across grid |
| Fig 5 | Head-to-head comparison | Recovery R² by method × variable, AUROC comparison, failure mode curves |
| Fig 6 | Robustness sweep | 2×3 grid: N_patients × noise × misspecification for both methods |

---

## 7. Implementation Notes for Claude Code

### Dependencies to install

```bash
pip install numpy scipy matplotlib seaborn pandas scikit-learn
pip install torch torchdiffeq  # for UDE
pip install pysindy             # for SINDy
pip install pyyaml jupyter      # utilities
pip install pytest              # testing
```

### Known implementation pitfalls

1. **Reynolds ODE stiffness:** Use `method='Radau'` or `method='LSODA'` in
   `solve_ivp` for the chronic inflammation basin — RK45 can fail there.
   Always check that `sol.success == True`.

2. **P₀ estimation in MOTIF:** Since P is unobserved, initialise P₀ at 5.0
   (mid-range) and allow it to vary freely during calibration. Bound between 0.1
   and 20.0.

3. **UDE training instability:** The ODE solver inside the training loop can
   diverge for early epochs. Use gradient clipping (`torch.nn.utils.clip_grad_norm_`
   with max_norm=1.0) and a learning rate scheduler.

4. **SINDy feature selection:** Start with `degree=2` and `threshold=0.05`.
   Only increase degree if the degree-2 model has R² < 0.90. Too many features
   relative to grid points causes SINDy to overfit.

5. **Reproducibility:** Set seeds at the top of every script:
   ```python
   np.random.seed(42)
   torch.manual_seed(42)
   ```

6. **Outcome determination:** After simulation, classify patient outcome by the
   value of `f` at t=72h:
   - `f(72) < 0.1` → resolution
   - `0.1 ≤ f(72) < 0.5` → chronic
   - `f(72) ≥ 0.5` → death
   
   Verify this against phase portrait expectations.
