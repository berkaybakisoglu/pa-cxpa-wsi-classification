import matplotlib.pyplot as plt
import numpy as np

from scripts.fig_lang import EXTRACTOR_LABELS, lang_args, load_summary, save

TEXT = {
    "tr": {"agg_ticks": ["Ort. havuz.", "Yama", "CLAM-SB", "ABMIL"], "colour_ticks": ["Ham", "Macenko"],
           "a": "(a) Öznitelik çıkarıcı (Macenko, 11'er)",
           "b": "(b) Toplama yöntemi (ortak stratejiler, 6'şar)",
           "c": "(c) Dengesizlik stratejisi (yalnız MIL, 9'ar)",
           "d": "(d) Renk normalizasyonu (tüm 66)"},
    "en": {"agg_ticks": ["Mean pool.", "Patch", "CLAM-SB", "ABMIL"], "colour_ticks": ["Raw", "Macenko"],
           "a": "(a) Feature extractor (Macenko, 11 each)",
           "b": "(b) Aggregation (shared strategies, 6 each)",
           "c": "(c) Imbalance strategy (MIL only, 9 each)",
           "d": "(d) Stain normalisation (all 66)"},
}


def strip(ax, df, column, levels, tick_labels, title, rng):
    for i, level in enumerate(levels):
        values = df[df[column] == level]["auroc"].values
        x = np.full(len(values), i) + rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(x, values, s=34, alpha=0.6, color="#7f7f7f", edgecolors="white", linewidths=0.4, zorder=2)
        mean = values.mean()
        ax.plot([i - 0.25, i + 0.25], [mean, mean], color="#d62728", lw=2.2, zorder=3)
        ax.text(i, 1.03, f"{mean:.2f}", ha="center", fontsize=8.5, color="#d62728")
    ax.axhline(0.5, color="red", ls="--", lw=0.9, alpha=0.5)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_ylim(0.3, 1.08)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def main():
    args = lang_args("fig_marginal.png")
    text = TEXT[args.lang]
    df = load_summary()
    mac = df[df["macenko"]]
    rng = np.random.default_rng(0)
    extractors = ["resnet50", "ctranspath", "uni"]

    # Each level is averaged over a comparable set: the patch baseline has no SMOTE arm,
    # so (b) uses only the strategies it shares and (c) uses only the MIL aggregators.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    strip(axes[0, 0], mac, "extractor", extractors, [EXTRACTOR_LABELS[e] for e in extractors], text["a"], rng)
    strip(axes[0, 1], mac[mac["strat"].isin(["none", "oversample"])], "aggregator",
          ["mean_pool", "patch_level", "clam_sb", "abmil"], text["agg_ticks"], text["b"], rng)
    strip(axes[1, 0], mac[mac["aggregator"] != "patch_level"], "strat",
          ["none", "oversample", "smote"], ["none", "oversample", "SMOTE"], text["c"], rng)
    strip(axes[1, 1], df, "macenko", [False, True], text["colour_ticks"], text["d"], rng)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC")
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
