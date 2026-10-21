# Patch-level versus multiple instance learning for carcinoma ex pleomorphic adenoma

Code and per-patient results for the paper *"Patch-Level Versus Multiple Instance Learning for
Carcinoma Ex Pleomorphic Adenoma on Whole-Slide Images"* (submitted to *Biomedicines*).

The study classifies 19 H&E whole-slide images (14 pleomorphic adenoma, 5 carcinoma ex pleomorphic
adenoma) with leave-one-patient-out cross-validation (LOPOCV). It compares a patch-level classifier
with three multiple instance learning (MIL) aggregators (mean pooling, ABMIL, CLAM-SB) on frozen
features from three extractors (ResNet-50, CTransPath, UNI), with and without Macenko stain
normalisation and with three class-imbalance strategies: 66 configurations in total.

The slides are not public (patient privacy). The repository contains the full pipeline and the
LOPOCV predictions of every configuration, from which all metrics, tables and most figures of the
paper can be regenerated without the slides.

## Layout

```
config.yaml                  all settings (paths, preprocessing, model, training, evaluation)
run_pipeline.py              runs every stage in order
src/preprocess.py            tissue detection and patch coordinates (256 px, level 1 = 0.487 µm/px)
src/extract_features.py      frozen-extractor features per slide
src/models/                  feature extractors, mean pooling / ABMIL / CLAM-SB, patch MLP
src/train.py                 MIL training with LOPOCV
src/patch_baseline.py        patch-level baseline with LOPOCV
src/evaluate.py              metrics, bootstrap CIs, LOO Youden threshold -> summary_metrics.csv
scripts/stainnorm_extract.py Macenko-normalised features
scripts/epoch_budget.py      epoch-sensitivity sweep and training-loss curves (fixed 50-epoch budget)
scripts/make_*.py            paper figures and tables (English; --lang tr for the thesis versions)
outputs/results/             per-patient predictions and training curves of all 66 configurations
tests/                       pytest suite
```

## Installation

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt      # Windows: venv\Scripts\pip
```

See the top of `requirements.txt` for installing the CUDA build of PyTorch. OpenSlide's binaries
come with `openslide-bin`.

## Reproducing the paper from the shipped results (no slides needed)

```bash
python -m src.evaluate                         # outputs/results/summary_metrics.csv
python -m scripts.make_main_tables             # Tables 2-5 (Table 1 needs the patch coordinates)
python -m scripts.make_appendix_tables         # Tables A1-A3
for m in make_forest_5_1 make_marginal_5_2 make_interaction_5_3 make_patch_vs_mil_5_3 \
         make_auroc_ece_5_4 make_auroc_ap_scatter; do python -m scripts.$m; done
```

Figures go to `outputs/figures/en/`, tables to `outputs/tables/en/`. The regenerated
`summary_metrics.csv` is identical to the shipped one.

Patients are pseudonymised as P01-P19. The pseudonyms follow the alphabetical order of the
original IDs, so the paired bootstrap, which aligns two result files by patient, reproduces the
published intervals exactly.

## Running the full pipeline (needs the slides)

1. **Slides.** MIRAX (3DHISTECH) slides go in `Bening/<id>/` and `Malign/<id>/`, each folder with its
   `Slidedat.ini`. OpenSlide opens MIRAX through a `<id>.mrxs` file next to the folder; an empty one
   is created automatically.
2. **Weights** in `pretrained/`:
   - `ctranspath.pth`: the released CTransPath checkpoint ([TransPath](https://github.com/Xiyue-Wang/TransPath)).
     It was saved with timm 0.5.4; the loader remaps it to current timm.
   - `uni.bin`: `pytorch_model.bin` from [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)
     (gated: request access on Hugging Face). Without the local file the loader tries the Hub and
     refuses to fall back to random weights.
   - ResNet-50 uses the torchvision `IMAGENET1K_V2` weights (downloaded automatically).
3. **Run** `python run_pipeline.py`. Stages: `preprocess`, `extract`, `stainnorm`, `train`,
   `evaluate`; `--start train` reuses existing features. Failed steps are listed at the end and
   make the exit code non-zero.

Every stage skips work whose output already exists (patch coordinates, feature files, finished
`*_folds.csv`), and an interrupted configuration resumes from its `*_partial.pt`. **After changing
`config.yaml`, delete the affected outputs**, otherwise the old results are kept.

## Reproducibility

- Each fold is seeded with `seed + fold index`; cuDNN runs in deterministic mode. On the same GPU
  and library versions, re-running training reproduced the stored `*_folds.csv` and `*_curves.csv`
  byte for byte (checked for MIL configurations including CLAM-SB, and for the patch baseline on
  shorter runs). Other GPUs or library versions may give slightly different numbers.
- Macenko normalisation (torchstain) is not bit-deterministic: normalised pixels can differ by one
  intensity level between runs, so re-extracted Macenko features differ in the last digits. The
  stain reference itself is deterministic: the patch with the highest mean HSV saturation among 40
  random tissue patches (seed 0) of the first benign slide. Patches whose normalisation fails
  (about 0.6%) are used unnormalised; the extraction prints the fraction.

## Implementation notes

- **CLAM-SB** is a re-implementation of the single-branch model of Lu et al. (2021) without the
  multi-class subtyping branch. As in the original, the total loss is
  `0.7 × bag CE + 0.3 × instance CE` (the original code also offers a smooth top-1 SVM instance
  loss; its default, cross-entropy, is used).
  `k = 8` top- and bottom-attended patches are used; for bags with fewer than 16 patches the two
  sets overlap.
- **Imbalance strategies.** `oversample` repeats minority bags until the classes are balanced (the
  patch baseline repeats whole slides before pooling their patches). `smote` runs SMOTE on
  mean-pooled bag vectors; each synthetic sample becomes a one-patch bag and is trained without the
  instance loss. Oversampling after SMOTE would change nothing, so that combination is not run.
  The patch baseline has no SMOTE arm.
- **Training.** Adam (lr 1e-4, weight decay 1e-5) for a fixed 50 epochs, final-epoch model. LOPOCV
  leaves no validation patient for early stopping. The budget was set from training-loss
  convergence (`scripts/epoch_budget.py trainloss`); the held-out sweep (`... sweep`) is a post hoc
  check only. The patch MLP has the same layers as the mean-pooling classifier, so the two differ
  only in where the averaging happens.
- **Evaluation.** Slide probability of the patch baseline = mean patch P(malignant). Confidence
  intervals: 1000 class-stratified bootstrap replicates (percentile method). The Youden threshold
  of each patient is chosen on the other 18 patients; its CIs hold these decisions fixed and are
  conditional on the threshold rule. Average precision is reported next to AUROC because with 5
  malignant cases it reflects minority-class performance more directly. ECE uses 5 bins.

## Tests

```bash
pytest                        # fast suite on synthetic data, no slides or weights, ~10 s on CPU
pytest -m "slow or data"      # pretrained extractors and a real-slide check (skipped if missing)
```

## Citation

If you use this code, please cite the paper (citation will be added after publication).

## License

MIT, see `LICENSE`. The pretrained models keep their own licences (UNI: CC-BY-NC-ND 4.0, gated).
