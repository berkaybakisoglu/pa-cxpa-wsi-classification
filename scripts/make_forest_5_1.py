import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.fig_lang import EXTRACTOR_LABELS, agg_styles, lang_args, legend_handles, load_summary, save

TEXT = {
    "tr": {"chance": "Rastgele (0,5)", "best_lo": " en iyi AUROC alt sınırı",
           "xlabel": "AUROC (%95 bootstrap güven aralığı)"},
    "en": {"chance": "Chance (0.5)", "best_lo": " lower CI bound of best AUROC",
           "xlabel": "AUROC (95% bootstrap confidence interval)"},
}


def main():
    args = lang_args("fig_forest_auroc.png")
    text = TEXT[args.lang]
    styles = agg_styles(args.lang)
    mac = load_summary(macenko_only=True).sort_values("auroc").reset_index(drop=True)
    labels = [f"{styles[r.aggregator][0]} / {EXTRACTOR_LABELS[r.extractor]} / {r.strat}"
              for r in mac.itertuples()]

    fig, ax = plt.subplots(figsize=(9, 10))
    for i, r in enumerate(mac.itertuples()):
        colour = styles[r.aggregator][1]
        ax.errorbar(r.auroc, i, xerr=[[r.auroc - r.auroc_ci_lower], [r.auroc_ci_upper - r.auroc]],
                    fmt="o", color=colour, ecolor=colour, elinewidth=1.4, capsize=2.5,
                    markersize=6, alpha=0.9)
    ax.axvline(0.5, color="red", ls="--", lw=1, alpha=0.6, label=text["chance"])
    best_lower = mac.loc[mac["auroc"].idxmax(), "auroc_ci_lower"]
    ax.axvline(best_lower, color="#444", ls=":", lw=1, alpha=0.7)
    ax.text(best_lower, len(mac) - 0.3, text["best_lo"], fontsize=8, color="#444", va="top")
    ax.set_yticks(np.arange(len(mac)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(text["xlabel"])
    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(-1, len(mac))
    ax.grid(axis="x", alpha=0.25)
    handles = legend_handles(styles.values()) + [Line2D([0], [0], color="red", ls="--", label=text["chance"])]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
