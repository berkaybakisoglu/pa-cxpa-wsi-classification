import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.fig_lang import lang_args, sample_patches, save
from src.preprocess import get_patient_list, load_config

TITLES = {"tr": "Örnek yamalar (256×256, Level 1, ham renk)",
          "en": "Example patches (256×256, level 1, raw colour)"}


def main():
    parser = argparse.ArgumentParser(description="Example extracted patches from a PA and a CXPA slide")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pa", required=True, help="patient ID of the PA slide")
    parser.add_argument("--cxpa", required=True, help="patient ID of the CXPA slide")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = lang_args("fig_patch_examples.png", parser)

    cfg = load_config(args.config)
    dirs = {pid: d for pid, _, d in get_patient_list(cfg)}
    patches_dir = os.path.join(cfg["data"]["output_dir"], "patches")
    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(2, args.n, figsize=(args.n * 2.0, 4.4))
    for row, (label, pid) in zip(axes, (("PA", args.pa), ("CXPA", args.cxpa))):
        patches = sample_patches(dirs[pid], os.path.join(patches_dir, f"{pid}_coords.h5"), args.n, rng)
        for ax, patch in zip(row, patches):
            ax.imshow(patch)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(label, fontsize=12, fontweight="bold", rotation=0, labelpad=28, va="center")
    fig.suptitle(TITLES[args.lang], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, args.out)


if __name__ == "__main__":
    main()
