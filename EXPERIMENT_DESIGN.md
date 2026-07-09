# Experiment Design: MOTIF vs. UDE Comparative Benchmark

## Overview

This document specifies the full experimental protocol for comparing MOTIF-style
proxy expansion and UDE + SINDy structure learning on synthetic data generated
from the Reynolds 2006 acute inflammation model.

**Research questions:**

1. How accurately does each method recover latent biological state (P, D, h)
   from partial observations (N*, CA, f)?
2. Under what data conditions (patient count, noise, sampling density) does
   each method degrade, and how?
3. What happens to MOTIF when the ODE is misspecified?
4. Can SINDy recover the true functional form of the pathogen clearance rate?
5. Does adding synthetic proxies (MOTIF) or learning structure (UDE) improve
   downstream outcome classification?

---

## Experiment 1: Baseline Comparison

**Purpose:** Establish the primary performance profile of both methods under
"ideal" conditions (N=500 patients, 10% noise, 6 timepoints, correct ODE).

**Protocol:**
1. Generate N=500 synthetic patients (see TECHNICAL_SPEC.md §2)
2. Apply 80/20 train/test split (stratified by outcome)
3. Run MOTIF pipeline on training set; evaluate on test set
4. Run UDE + SINDy pipeline on training set; evaluate on test set
5. Report all primary metrics (§5.1 of TECHNICAL_SPEC)

**Expected runtime:**
- MOTIF: ~5 minutes on laptop CPU
- UDE: ~30–60 minutes on laptop CPU (GPU recommended for faster training)

**Config file:** `experiments/config_baseline.yaml`

---

## Experiment 2: Data Sparsity Sweep

**Purpose:** Identify the minimum patient count at which each method remains
reliable. Critical for understanding applicability to real clinical cohorts
(where N is often 50–200).

**Protocol:**
- Vary N_patients ∈ {500, 200, 100, 50, 20, 10}
- Hold all other parameters at baseline values
- Run 5 replicate experiments per N (different random seeds)
- Report mean ± SD of R²(proxy_P, true_P) and AUROC

**Expected finding:**
- MOTIF degrades gracefully (it uses the ODE, not the data, to generate proxies;
  only the calibration step is data-dependent)
- UDE degrades sharply below N~100 (neural network needs sufficient patients)

**Config file:** `experiments/config_sparse_data.yaml`

---

## Experiment 3: Measurement Noise Sweep

**Purpose:** Test robustness to realistic biological and technical noise.
Real proteomics has 20–40% CV; clinical measures may be worse.

**Protocol:**
- Vary sigma_obs ∈ {0.05, 0.10, 0.20, 0.35, 0.50}
- Hold all other parameters at baseline values
- Run 5 replicates per noise level
- Report mean ± SD of all primary metrics

**Expected finding:**
- Both methods degrade with noise, but at different rates
- MOTIF is affected primarily through parameter calibration (noisy data → poor
  parameter estimates → poor proxies)
- UDE is affected through training (noisy targets → poorly trained network)

**Config file:** `experiments/config_noise_sweep.yaml`

---

## Experiment 4: ODE Misspecification (MOTIF Vulnerability Test)

**Purpose:** Test what happens to MOTIF when the biologist's ODE is wrong.
This is the primary vulnerability of the MOTIF approach — it assumes ODE
structure is correct.

**Protocol:**
- Generate ground-truth data from the FULL 6-variable Reynolds model
- Run MOTIF with progressively misspecified ODE models:
  - **Model A (correct):** 6-variable model — P, N*, D, CA, f
  - **Model B (missing D):** 5-variable model — P, N*, CA, f
    (D dynamics collapsed into N* term)
  - **Model C (minimal):** 4-variable model — P, N*, CA, f
    (D removed entirely; simplified N* equation)
  - **Model D (severely wrong):** Linear 3-variable model — P, N*, CA
    (ignores tissue damage entirely)
- Report R²(proxy_P, true_P) and AUROC for each model specification

**Expected finding:**
- R² degrades as misspecification worsens
- Classification still possible at intermediate misspecification (N*, CA proxies
  remain informative even if D proxy fails)
- Complete failure of P proxy when pathogen dynamics equation is wrong

**Config file:** `experiments/config_misspecified_ode.yaml`

---

## Experiment 5: SINDy Recovery Accuracy (UDE Symbolic Extraction Test)

**Purpose:** Evaluate whether SINDy recovers the true symbolic form of the
pathogen clearance rate function.

**True function:**
```
θ_true(N*) = k_pm * N* / (s_dm + N*)   [Michaelis-Menten]
           = 1.8 * N* / (0.05 + N*)
```

**Protocol:**
1. Train UDE to convergence on N=500 patients
2. Evaluate θ_learned(z) on a 3D grid of [N*, CA, f] values
3. Apply SINDy with candidate libraries:
   - Polynomial degree 2: {1, N*, CA, f, N*², CA², f², N*·CA, N*·f, CA·f}
   - Polynomial degree 3: adds cubic and triple-interaction terms
   - Rational basis (if pysindy supports): {N*/(c+N*)} Michaelis-Menten form
4. Report:
   - Which coefficients SINDy selects (sparsity pattern)
   - RMSE between SINDy function and true θ_true(N*)
   - Whether the functional form is qualitatively correct (monotone in N*,
     insensitive to CA and f — since true θ only depends on N*)

**Expected finding:**
- SINDy should select primarily N* terms (as in truth)
- May not recover exact Michaelis-Menten form with polynomial basis,
  but should recover a monotone-increasing function of N*
- R² > 0.90 with degree-2 polynomial if UDE is well-trained

---

## Experiment 6: Sampling Frequency Test

**Purpose:** Test robustness to reduced temporal resolution.
Clinical data is often sampled at 2–4 timepoints, not 6.

**Protocol:**
- Vary n_timepoints ∈ {6, 4, 3, 2}
- Hold all other parameters at baseline (N=500, sigma=0.10)
- Timepoint subsets:
  - 6: [0, 6, 12, 24, 48, 72]
  - 4: [0, 12, 24, 72]
  - 3: [0, 24, 72]
  - 2: [0, 72]
- Run 5 replicates per condition

**Expected finding:**
- MOTIF is relatively tolerant of sparse sampling (ODE interpolates)
- UDE is sensitive to sparse sampling (less signal for neural network training)

---

## Summary Table: Hypothesized Results

| Condition | MOTIF R²(proxy_P) | UDE θ RMSE | MOTIF AUROC | UDE AUROC |
|---|---|---|---|---|
| Baseline (N=500, σ=0.10) | > 0.85 | < 0.10 | > 0.90 | > 0.85 |
| Sparse (N=50) | > 0.80 | < 0.20 | > 0.85 | > 0.75 |
| High noise (σ=0.35) | > 0.70 | < 0.20 | > 0.80 | > 0.75 |
| Missing D (misspecified) | ~0.50 | N/A | > 0.75 | > 0.85 |
| 2 timepoints | > 0.75 | < 0.30 | > 0.80 | > 0.70 |

These are hypotheses — the benchmark exists to test whether they hold.
Unexpected results are scientifically interesting and should be reported as such.

---

## Figure Plan

### Figure 1: Model Overview (3 panels)

**Panel A:** Phase portrait of the Reynolds model showing the three basins of
attraction. X-axis = P₀, Y-axis = f(72h). Three colored regions (resolution,
chronic, death). Separatrix lines. Caption: "The Reynolds 2006 system exhibits
bistability with three clinically distinct outcome trajectories."

**Panel B:** Three representative patient trajectories (one per outcome).
X-axis = time (hours). Y-axes = N*, CA, f (one curve each, three line styles).
Solid = observed (noisy), dashed = ground truth.

**Panel C:** Observation model schematic. Diagram showing 6 ODE state variables;
3 highlighted as "observed" (N*, CA, f) and 3 marked as "latent" (P, D, h).
Two arrows leaving the diagram: one to "MOTIF pipeline" (ODE → proxies), one
to "UDE pipeline" (data → ODE learning).

### Figure 2: Virtual Patient Cohort

**Panel A:** Histogram of P₀ values, colored by outcome (resolution/chronic/death).
**Panel B:** 3×6 heatmap grid — mean observed trajectories per outcome class
per variable (N*, CA, f).
**Panel C:** PCA of 18-dimensional feature vectors (6 timepoints × 3 variables),
colored by outcome. Should show reasonable separation even in raw data.

### Figure 3: MOTIF Pipeline Results

**Panel A–C:** Scatter plots — proxy_P vs. true_P, proxy_D vs. true_D,
proxy_h vs. true_h. One point per patient. R² annotated. Color = outcome.

**Panel D:** Correlation heatmap — observed features (N*_auc, CA_auc, f_max, ...)
vs. proxy features (proxy_P_auc, proxy_D_auc, proxy_h_min). Reveal which
observations correlate with which latent processes.

**Panel E:** Classification performance — bar chart of AUROC for three classifiers:
(1) observed features only, (2) proxy features only, (3) observed + proxies.
Should show highest AUROC with combined features.

### Figure 4: UDE + SINDy Results

**Panel A:** Training loss curve over 500 epochs. Should show convergence.

**Panel B:** Learned θ(N*, CA=0.2, f=0.1) vs. true θ(N*) as a function of N*,
for fixed CA and f. Blue = true Michaelis-Menten, orange = UDE learned,
green = SINDy recovered. Ribbon = uncertainty if available.

**Panel C:** SINDy coefficient bar plot. Show which terms were selected (non-zero)
vs. zeroed out. Annotate the recovered equation.

**Panel D:** RMSE of θ_sindy vs. θ_true as a function of N_patients and noise.
2D heatmap.

### Figure 5: Head-to-Head Comparison

**Panel A:** Bar chart — R² for recovery of P, D, h.
- MOTIF bars (blue) vs. UDE (orange, if UDE also estimates latent states)
- Show only latent variables that each method targets

**Panel B:** AUROC comparison — bar chart with error bars across 5 replicates.
Baseline conditions.

**Panel C–D:** Robustness sweep curves.
- C: R² or AUROC vs. N_patients (two lines: MOTIF, UDE)
- D: R² or AUROC vs. noise level (two lines: MOTIF, UDE)

### Figure 6: ODE Misspecification (MOTIF Only)

**Panel A:** R²(proxy_P, true_P) vs. model specification level
(correct 6-var → 5-var → 4-var → 3-var linear).
Bar chart with individual patient points overlaid.

**Panel B:** AUROC vs. model specification level.

**Panel C:** Example proxy trajectories — correct model vs. misspecified model
vs. ground truth for a single representative patient.

---

## Statistical Analysis Plan

- All experiments with multiple replicates: report mean ± SD
- Primary statistical comparison (MOTIF vs. UDE AUROC): paired t-test across
  replicates, two-tailed, α=0.05
- Recovery R²: Pearson correlation coefficient with 95% CI via bootstrap (N=1000)
- Robustness curves: fit linear trend to log-transformed N_patients axis;
  report slope as "degradation rate"
- All p-values: Bonferroni-corrected for multiple comparisons within experiment

---

## Output Artifacts

Each experiment run should produce:

```
results/
  experiment_{name}_{timestamp}/
    config.yaml           ← copy of configuration used
    synthetic_patients.pkl ← generated patient data
    motif_results.pkl     ← MOTIF pipeline outputs
    ude_results.pkl       ← UDE pipeline outputs
    metrics.json          ← all computed metrics
    
figures/
  experiment_{name}_{timestamp}/
    fig1_model_overview.pdf
    fig2_cohort.pdf
    fig3_motif.pdf
    fig4_ude.pdf
    fig5_comparison.pdf
    fig6_misspecification.pdf  ← only for experiment 4
```

---

## LinkedIn / Communication Output

Each figure should also be rendered as a square 1080×1080 PNG suitable for
LinkedIn posting, with:
- White background
- Larger fonts (minimum 14pt equivalent at final size)
- Simplified titles and minimal axis labels
- Stewart Chang / Model Stewardship LLC attribution in bottom-right corner

These are stored in `figures/linkedin/` separately from the publication figures.
