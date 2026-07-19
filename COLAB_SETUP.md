# Running Baseline on Google Colab

This guide helps you run the full baseline experiment (500 patients, 5 replicates, ~3–5 hours on CPU) in ~30–90 minutes on Colab's free T4 GPU.

## Quick Start

### Path 1: GitHub Clone (Recommended)

1. **Push your repo to GitHub** (if not already there):
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/ode-multiomics-benchmark.git
   git push -u origin master
   ```

2. **Open Colab notebook**:
   - Go to https://colab.research.google.com
   - Click `File > Open notebook > GitHub`
   - Paste: `https://github.com/YOUR_USERNAME/ode-multiomics-benchmark`
   - Select `notebooks/run_baseline_colab.ipynb`

3. **Update GitHub URL in notebook**:
   - In Cell 2 (Option A), replace `YOUR_USERNAME` with your actual GitHub username

4. **Enable GPU**:
   - Click `Runtime > Change runtime type > T4 GPU`
   - Click `Save`

5. **Run All**:
   - Click `Runtime > Run all` or step through cells manually
   - Watch progress for ~30–90 minutes

6. **Download results**:
   - After Cell 6 completes, open Google Drive
   - Go to `ode-multiomics-results/baseline_<timestamp>/`
   - Download the folder to your machine's `results/` directory

---

### Path 2: Upload ZIP (Fallback if GitHub unavailable)

1. **Create the ZIP archive locally**:
   ```powershell
   # From the repo root
   .\scripts\package_for_colab.ps1
   ```
   This creates `ode-multiomics-benchmark.zip` (~30 MB)

2. **Upload to Google Drive**:
   - Go to https://drive.google.com
   - Upload `ode-multiomics-benchmark.zip` to the root of `My Drive`

3. **Open Colab notebook**:
   - Go to https://colab.research.google.com
   - Click `File > New notebook`
   - Copy the contents of `notebooks/run_baseline_colab.ipynb` into it
   - OR upload the `.ipynb` directly to Drive and open it from there

4. **Use Option B in Cell 2**:
   - Comment out Option A (Cell 2, top)
   - Uncomment Option B (Cell 2, bottom)

5. **Continue as Path 1** (steps 4–6)

---

## What Happens

**Cell 1**: Mounts your Google Drive and creates `ode-multiomics-results/` folder

**Cell 2**: Gets the code (either clones from GitHub or unzips from Drive)

**Cell 3**: Installs missing Python packages (`torchdiffeq`, `pysindy`, etc.)
- Colab already has: torch, numpy, scipy, scikit-learn, pandas, matplotlib

**Cell 4**: Checks that GPU is available (should show "T4 GPU" or similar)

**Cell 5**: Runs the baseline experiment
- Generates 500 synthetic patients × 5 replicates
- MOTIF: calibrates each patient's ODE parameters
- UDE: trains a neural ODE for 500 epochs
- SINDy: recovers symbolic equations from UDE predictions
- Creates `results/baseline_<timestamp>/` with metrics and pickle files

**Cell 6**: Copies results to Google Drive

**Cell 7** (optional): Quick summary of R² values and UDE loss

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No GPU detected" | Click `Runtime > Change runtime type > Select T4 GPU (or A100 if available)` |
| GitHub clone times out | Use Path 2 (ZIP upload) instead |
| `ModuleNotFoundError: torchdiffeq` | Cell 3 didn't run — re-run it |
| Cell 5 takes >2 hours | GPU session may have slowed down; it's normal but rare — wait or restart |
| Drive mount says "No such file or directory" | Re-run Cell 1 and authorize access when prompted |
| Results don't appear after Cell 6 | Check that Cell 5 completed without errors; look for `results/baseline_*/` in repo root |

---

## After the Run

1. **Download results locally**:
   - Open Google Drive
   - Right-click `ode-multiomics-results/baseline_<timestamp>/` → `Download`
   - Extract to your machine's `results/` folder

2. **Analyze locally**:
   ```bash
   # Evaluate fast run quality (works for baseline too)
   python evaluate_fast_run.py  # adjust script to point to baseline_<timestamp>
   ```

3. **Next steps**:
   - If quality is good: baseline run is done, ready for publication
   - If quality issues: consider re-running with different config or more replicates

---

## Cost

**Free Colab**: ~1 run per day on T4 GPU (within fair-use limits)
- This baseline run = ~1–2 hours, so fits within one session

**Colab Pro** ($10/month): Priority GPU access, longer runtime limits (up to 24h)

---

## Upgrade: GCP Vertex AI (Later)

Once this works and you want to run recurring baselines, migrate to GCP:
```bash
# Create a Dockerfile that runs the same script
# Deploy via: gcloud ai custom-jobs create --config job.yaml
# Results sync to Cloud Storage instead of Drive
```
See plan file for details.
