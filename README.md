# ODE–Multiomics Benchmark: MOTIF vs. UDE Pilot

## Project Overview

This repository implements a synthetic-data benchmark comparing two complementary
approaches to integrating mechanistic ODE models with multiomics data:

1. **MOTIF-style proxy expansion** (Funk/Bangs/Paterson approach): Run a known ODE
   forward, use synthetic state-variable trajectories as proxy columns in a multiomics
   matrix, and recover biological signal through correlation analysis.

2. **UDE + SINDy structure learning** (Chang/Rackauckas approach): Use a partial
   observation model to feed observed "multiomics" into a Universal Differential
   Equation, learn unknown rate functions with a neural network, and recover
   interpretable equations via Sparse Identification of Nonlinear Dynamics (SINDy).

The benchmark model is Reynolds et al. (2006) — a 6-ODE system of acute inflammation
with known bistability and three clinically distinct outcome trajectories (resolution,
chronic inflammation, death). Ground-truth parameters are known, making precise
recovery evaluation possible.

---

## Scientific Context

### The Reynolds (2006) Model

Reynolds A, Rubin J, Clermont G, Day J, Vodovotz Y, Ermentrout GB.
"A reduced mathematical model of the acute inflammatory response: I. Derivation of
model and analysis of anti-inflammation."
*J Theor Biol* 242(1):220–36. (2006)

Six ODEs representing:
- `P`  — pathogen burden
- `N*` — early pro-inflammatory mediator (activated neutrophils / early cytokines; analog: IL-6, TNF-α)
- `D`  — late/damage-associated pro-inflammatory mediator (DAMPs, tissue signals)
- `CA` — anti-inflammatory mediator (analog: IL-10, TGF-β)
- `f`  — tissue damage fraction
- `h`  — tissue health/integrity (1 − f)

**Observation model for this benchmark:** Only `N*`, `CA`, and `f` are "observed"
(the multiomics). `P`, `D`, and `h` are latent — the biology to be recovered.

### Why This Model?

| Property | Relevance |
|---|---|
| Published canonical parameters | Ground truth for quantitative recovery evaluation |
| Bistability (3 attractors) | Non-trivial recovery problem; tests both methods under realistic complexity |
| Small state space (6 vars) | Both pipelines are tractable on a laptop |
| Infection-adjacent biology | Directly relevant to Stewart Chang's TB/inflammation background |
| Plasma proteomics analog | N* ≈ IL-6; CA ≈ IL-10; f ≈ organ failure score — realistic clinical mapping |

### Methodological Lineage

```
Hoffmann 2002 (Science)
  → Known ODE → experimental validation → predicts gene expression
  → "Forward model: biology → ODE → data"

Funk / Bangs / Paterson 2025-26 (MOTIF)
  → Known ODE → synthetic state variable trajectories → multiomics proxy columns
  → "Forward model + expansion: ODE generates interpretable proxies for multiomics"

Chang (this proposal / UDE approach)
  → Multiomics → encoder → UDE (known + learned terms) → SINDy → interpretable ODE
  → "Inverse problem: data trains the ODE structure via hybrid learning"
```

The benchmark directly compares methods 2 and 3 on shared synthetic data with
known ground truth from method 1.

---

## Repository Structure

```
ode-multiomics-benchmark/
│
├── README.md                          ← This file
├── CLAUDE_CODE_PROMPT.md              ← Full kickoff prompt for Claude Code
├── TECHNICAL_SPEC.md                  ← Full mathematical and implementation spec
├── EXPERIMENT_DESIGN.md               ← Experimental design, evaluation metrics, figures
│
├── papers/                            ← Reference PDFs (add manually — see below)
│   ├── Reynolds_2006_JTheorBiol.pdf
│   ├── Hoffmann_2002_Science.pdf
│   ├── Brunton_2016_Science_SINDy.pdf
│   ├── Rackauckas_2021_UDE.pdf
│   ├── Funk_2026_JPAD.pdf
│   └── README_papers.md               ← DOIs and download instructions
│
├── src/
│   ├── __init__.py
│   ├── reynolds_ode.py                ← Reynolds 2006 ODE system + parameters
│   ├── synthetic_data.py              ← Virtual patient generator
│   ├── motif_pipeline.py              ← MOTIF proxy expansion pipeline
│   ├── ude_pipeline.py                ← UDE + SINDy pipeline
│   ├── evaluation.py                  ← Recovery metrics and statistical comparison
│   └── plotting.py                    ← All figure generation
│
├── notebooks/
│   ├── 01_reynolds_ode_exploration.ipynb
│   ├── 02_synthetic_patient_generation.ipynb
│   ├── 03_motif_pipeline.ipynb
│   ├── 04_ude_sindy_pipeline.ipynb
│   └── 05_comparison_and_figures.ipynb
│
├── experiments/
│   ├── config_baseline.yaml           ← Default experiment configuration
│   ├── config_sparse_data.yaml        ← Low-N patient experiment
│   ├── config_misspecified_ode.yaml   ← ODE misspecification experiment
│   └── config_noise_sweep.yaml        ← Noise level sweep
│
├── results/
│   └── .gitkeep
│
├── figures/
│   └── .gitkeep
│
├── tests/
│   ├── test_reynolds_ode.py
│   ├── test_synthetic_data.py
│   ├── test_motif_pipeline.py
│   └── test_ude_pipeline.py
│
├── environment.yml                    ← Conda environment specification
├── requirements.txt                   ← pip requirements
└── pyproject.toml                     ← Package configuration
```

---

## Papers to Add to `papers/`

Download these manually and place in `papers/`. DOIs and sources are in
`papers/README_papers.md`. These are the primary references Claude Code
needs to understand model equations and methodological context.

| File | Reference | Where to get |
|---|---|---|
| `Reynolds_2006_JTheorBiol.pdf` | Reynolds et al. 2006 | doi:10.1016/j.jtbi.2006.01.015 |
| `Hoffmann_2002_Science.pdf` | Hoffmann et al. 2002 | doi:10.1126/science.1071914 |
| `Brunton_2016_Science_SINDy.pdf` | Brunton, Proctor, Kutz 2016 | doi:10.1126/science.afd0755 |
| `Rackauckas_2021_UDE.pdf` | Rackauckas et al. 2021 | arXiv:2001.04385 |
| `Funk_2026_JPAD.pdf` | Funk et al. 2026 | doi:10.1016/j.tjpad.2025.100402 |

---

## Quickstart

```bash
# 1. Clone / create project directory
mkdir ode-multiomics-benchmark && cd ode-multiomics-benchmark

# 2. Create environment
conda env create -f environment.yml
conda activate ode-multiomics

# 3. Install package in dev mode
pip install -e .

# 4. Run baseline experiment
python -m src.run_experiment --config experiments/config_baseline.yaml

# 5. Launch notebooks
jupyter lab notebooks/
```

---

## Implementation Priority Order for Claude Code

1. `src/reynolds_ode.py` — ODE system (highest priority; everything else depends on it)
2. `src/synthetic_data.py` — virtual patient generator
3. `src/motif_pipeline.py` — MOTIF pipeline (faster to implement; validate first)
4. `src/evaluation.py` — recovery metrics
5. `src/ude_pipeline.py` — UDE + SINDy (most complex; implement last)
6. `src/plotting.py` — figures
7. All notebooks

---

## Key Design Decisions

- **Language:** Python 3.11+
- **ODE solver:** `scipy.integrate.solve_ivp` with RK45 for ground truth;
  `torchdiffeq` for the UDE (allows autograd through ODE solver)
- **SINDy:** `pysindy` library (Kaptanoglu et al. 2022)
- **UDE neural network:** PyTorch (MLP, 2 hidden layers, tanh activation)
- **Plotting:** matplotlib + seaborn; all figures publication-quality at 300 DPI
- **Configuration:** YAML via PyYAML; all experiment parameters externalised
- **Random seeds:** All stochastic components seeded; reproducibility required

---

## Contact / Project Origin

Developed by Stewart Chang (Model Stewardship LLC) as a methodological pilot
comparing MOTIF-style ODE proxy expansion (Funk/Bangs/Paterson 2025-26) with
UDE-based structure learning (Rackauckas 2021 + Chang TB concept note 2026)
on a canonical acute inflammation model.

Intended outputs:
- Preprint / methods paper comparing both pipelines
- LinkedIn post series demonstrating QSP + generative AI competency
- Open-source Python package for the broader QSP / systems biology community
