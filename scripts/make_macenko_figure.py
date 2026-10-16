import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.fig_lang import lang_args, sample_patches, save
from scripts.stainnorm_extract import fit_normalizer, normalize_or_none
from src.preprocess import get_patient_list, load_config

TEXT = {
    "tr": {"raw": "Ham", "title": "Macenko renk normalizasyonu: öncesi (üst) ve sonrası (alt)"},
    "en": {"raw": "Raw", "title": "Macenko stain normalisation: before (top) and after (bottom)"},
}


def main():
    parser = argparse.ArgumentParser(description="Raw vs Macenko-normalised patches from a PA and a CXPA slide")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pa", required=True, help="patient ID of the PA slide")
    parser.add_argument("--cxpa", required=True, help="patient ID of the CXPA slide")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5)
    args = lang_args("fig_macenko.png", parser)
    text = TEXT[args.lang]

    cfg = load_config(args.config)
    dirs = {pid: d for pid, _, d in get_patient_list(cfg)}
    patches_dir = os.path.join(cfg["data"]["output_dir"], "patches")
    normalizer = fit_normalizer(cfg)
    rng = np.random.default_rng(args.seed)
    columns = [(label, patch)
               for label, pid in (("PA", args.pa), ("CXPA", args.cxpa))
               for patch in sample_patches(dirs[pid], os.path.join(patches_dir, f"{pid}_coords.h5"), args.n, rng)]

    fig, axes = plt.subplots(2, len(columns), figsize=(len(columns) * 2.0, 4.6))
    for c, (label, raw) in enumerate(columns):
        axes[0, c].imshow(raw)
        axes[1, c].imshow(normalize_or_none(normalizer, raw) or raw)
        axes[0, c].set_title(label, fontsize=9)
        for ax in axes[:, c]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, row_label in zip(axes[:, 0], (text["raw"], "Macenko")):
        ax.set_ylabel(row_label, fontsize=12, fontweight="bold", rotation=0, labelpad=24, va="center")
    fig.suptitle(text["title"], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, args.out)


if __name__ == "__main__":
    main()
