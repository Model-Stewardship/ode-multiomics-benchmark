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
├── papers/                            ← Reference PDFs
│   ├── Bangs 2025 - Developing a digital twin clinical decision support tool.pdf
│   ├── Brunton 2016 - Discovering governing equations from data by sparse identification of nonlinear dynamical systems.pdf
│   ├── Funk 2026 - Mining the gaps Deciphering Alzheimer's biology through AI-driven reconciliation.pdf
│   ├── Hoffmann 2002 - The IkB-NF-kB signaling module Temporal control and selective gene activation.pdf
│   ├── Paterson 2025 - From digital twins to multiomic inference A systems-level framework.pdf
│   ├── Rackauckas 2021 - Universal differential equations for scientific machine learning.pdf
│   ├── Reynolds 2006 - A reduced mathematical model of the acute inflammatory response.pdf
│   └── README_papers.md               ← Citation details and usage notes
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

## Papers Reference

All papers are present in `papers/`. For full citation details, DOIs, and usage notes,
see `papers/README_papers.md`. These are the primary references needed to understand
model equations and methodological context.

| Paper | Type | Year |
|---|---|---|
| Reynolds 2006 | Journal article | 2006 |
| Hoffmann 2002 | Journal article | 2002 |
| Brunton 2016 | Journal article | 2016 |
| Rackauckas 2021 | ArXiv preprint | 2021 |
| Funk 2026 | Journal article | 2026 |
| Bangs 2025 | Conference poster (AAIC) | 2025 |
| Paterson 2025 | Conference poster (AAIC) | 2025 |

---

## Quickstart

**If starting fresh (clone from GitHub):**

```bash
# 1. Clone repository
git clone https://github.com/Model-Stewardship/ode-multiomics-benchmark.git
cd ode-multiomics-benchmark

# 2. Create virtual environment and install dependencies
uv sync --extra dev

# 3. Run baseline experiment
uv run python -m src.run_experiment --config experiments/config_baseline.yaml

# 4. Launch notebooks
uv run jupyter lab notebooks/
```

**If you already have the directory locally:**

```bash
# Just create the virtual environment and install dependencies
uv sync --extra dev

# Then run experiments or notebooks as above
uv run python -m src.run_experiment --config experiments/config_baseline.yaml
uv run jupyter lab notebooks/
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
