# Papers Directory

Place PDF files here. They are referenced by Claude Code for biological and
methodological context during implementation.

## Required Papers

### 1. Reynolds 2006 — The Benchmark Model
**File:** `Reynolds_2006_JTheorBiol.pdf`
**Citation:** Reynolds A, Rubin J, Clermont G, Day J, Vodovotz Y, Ermentrout GB.
"A reduced mathematical model of the acute inflammatory response: I. Derivation
of model and analysis of anti-inflammation."
*Journal of Theoretical Biology* 242(1):220–36. (2006)
**DOI:** 10.1016/j.jtbi.2006.01.015
**Get it:** https://doi.org/10.1016/j.jtbi.2006.01.015
**Why:** This is the benchmark model. Contains the full ODE system, all canonical
parameters (Table 1), phase portraits, and the three-basin bistability analysis.
Claude Code must read the parameter table to implement `REYNOLDS_PARAMS` correctly.

---

### 2. Hoffmann 2002 — The Canonical Systems Biology Precedent
**File:** `Hoffmann_2002_Science.pdf`
**Citation:** Hoffmann A, Levchenko A, Scott ML, Baltimore D.
"The IκB–NF-κB signaling module: temporal control and selective gene activation."
*Science* 298(5596):1241–5. (2002)
**DOI:** 10.1126/science.1071914
**Supplement:** https://science.sciencemag.org/content/suppl/2003/09/05/298.5596.1241.DC1
**Why:** The intellectual ancestor of MOTIF. Shows that ODE state variables
(IκBα, IκBβ, IκBε dynamics) predict downstream transcriptomic outcomes
(RANTES vs. IP-10 gene expression). The ODE supplement contains the full
26-ODE model used in the original paper — the Lipniacki 2004 simplification
is more tractable for implementation.

---

### 3. Brunton 2016 — SINDy Method
**File:** `Brunton_2016_Science_SINDy.pdf`
**Citation:** Brunton SL, Proctor JL, Kutz JN.
"Discovering governing equations from data by sparse identification of
nonlinear dynamical systems."
*Science* 352(6284):477–9. (2016)
**DOI:** 10.1126/science.afd0755
**Why:** The SINDy method used in the UDE pipeline post-processing step.
Explains sparse regression on a library of candidate functions to recover
symbolic differential equations from data. Essential for understanding the
SINDy extraction step in `src/ude_pipeline.py`.

---

### 4. Rackauckas 2021 — Universal Differential Equations
**File:** `Rackauckas_2021_UDE.pdf`
**Citation:** Rackauckas C, Ma Y, Martensen J, Warner C, Zubov K, Supekar R,
Skinner D, Ramadhan A, Edelman A.
"Universal differential equations for scientific machine learning."
*arXiv:2001.04385* (2021)
**Get it:** https://arxiv.org/abs/2001.04385 (free)
**Why:** The UDE framework that underpins the second pipeline. Explains how to
augment mechanistic ODE systems with neural network terms and train them
end-to-end using automatic differentiation through ODE solvers.

---

### 5. Funk 2026 — MOTIF Conceptual Framework
**File:** `Funk_2026_JPAD.pdf`
**Citation:** Funk CC, Paterson T, Bangs A, Cannon DM, Savage G, Ringger E, Hood L.
"Mining the gaps: Deciphering Alzheimer's biology through AI-driven reconciliation."
*Journal of Prevention of Alzheimer's Disease* 13(2026):100402.
**DOI:** 10.1016/j.tjpad.2025.100402
**Why:** Describes the MOTIF (Multi-Omic Twin-Inferred Function) framework
that this benchmark implements and tests. The "AI scientist" architecture
(orchestrator, enforcer, architect) and the synthetic proxy concept are
described here.

---

## Optional but Useful Papers

### Lipniacki 2004 — Simplified NF-κB Model
**File:** `Lipniacki_2004_JTheorBiol.pdf` (optional)
**Citation:** Lipniacki T et al. "Mathematical model of NF-κB regulatory module."
*J Theor Biol* 228:195–215. (2004)
**DOI:** 10.1016/j.jtbi.2004.01.001
**Why:** The 15-ODE simplified NF-κB model — cleaner than Hoffmann for
extraction purposes; all equations in main text, not supplement.

### Vodovotz 2024 — Inflammation Digital Twin (Clinical Application)
**Citation:** Cannon JW et al. "Digital twin mathematical models suggest
individualized hemorrhagic shock resuscitation strategies."
*Communications Medicine* 4:68. (2024)
**DOI:** 10.1038/s43856-024-00535-6
**Why:** Demonstrates ODE inflammation model calibrated to patient proteomics
from a real clinical cohort — the closest existing work to what this benchmark
implements. Shows MOTIF-style approach applied to trauma/sepsis.

### Kaptanoglu 2022 — PySINDy Software
**Citation:** Kaptanoglu AA et al. "Promoting global stability in data-driven
models of quadratic nonlinear dynamics." *Physical Review Fluids* 6:094401 (2021)
**DOI:** 10.1098/rspa.2021.0904
**Why:** Documents the `pysindy` Python library used in the UDE pipeline.
See also: https://pysindy.readthedocs.io/

---

## Notes for Claude Code

When you read these papers, focus on:

**Reynolds 2006:**
- Table 1: all parameter values (copy directly into REYNOLDS_PARAMS)
- Figure 2: phase portrait showing three basins
- Equations 1–6: the ODE system (verify against TECHNICAL_SPEC)

**Brunton 2016:**
- Algorithm 1: the SINDy procedure
- Figure 1: the candidate function library concept

**Rackauckas 2021:**
- Figure 1: the UDE architecture diagram
- Section 2: how the neural network term augments a known ODE

If a paper is missing from this directory, proceed with the equations and
parameters specified in TECHNICAL_SPEC.md — they are fully reproduced there
and do not require the original PDFs to implement.
