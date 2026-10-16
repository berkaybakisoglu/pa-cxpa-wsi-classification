import matplotlib.pyplot as plt
import numpy as np

from scripts.fig_lang import EXTRACTOR_LABELS, lang_args, load_summary, prevalence, save

TEXT = {
    "tr": {"ap_title": "(b) AP (ortalama kesinlik)", "patch": "Yama düzeyi", "mil": "En iyi MIL"},
    "en": {"ap_title": "(b) AP (average precision)", "patch": "Patch-level", "mil": "Best MIL"},
}
EXTRACTORS = ["uni", "ctranspath", "resnet50"]
PATCH_COLOUR, MIL_COLOUR = "#2ca02c", "#1f77b4"


def main():
    args = lang_args("fig_patch_vs_mil.png")
    text = TEXT[args.lang]
    df = load_summary(macenko_only=True)
    df = df[df["strat"] == "none"]
    is_patch = df["aggregator"] == "patch_level"
    width = 0.36
    x = np.arange(len(EXTRACTORS))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, title, chance in [(axes[0], "auroc", "(a) AUROC", 0.5),
                                      (axes[1], "ap", text["ap_title"], prevalence(df))]:
        patch = [df[is_patch & (df["extractor"] == e)][metric].max() for e in EXTRACTORS]
        mil = [df[~is_patch & (df["extractor"] == e)][metric].max() for e in EXTRACTORS]
        ax.bar(x - width / 2, patch, width, label=text["patch"], color=PATCH_COLOUR)
        ax.bar(x + width / 2, mil, width, label=text["mil"], color=MIL_COLOUR)
        for i in range(len(EXTRACTORS)):
            ax.text(x[i] - width / 2, patch[i] + 0.012, f"{patch[i]:.2f}", ha="center", fontsize=8)
            ax.text(x[i] + width / 2, mil[i] + 0.012, f"{mil[i]:.2f}", ha="center", fontsize=8)
        ax.axhline(chance, color="red", ls="--", lw=0.9, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([EXTRACTOR_LABELS[e] for e in EXTRACTORS])
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
