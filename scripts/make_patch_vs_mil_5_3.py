import matplotlib.pyplot as plt

from scripts.fig_lang import EXTRACTOR_LABELS, agg_styles, lang_args, legend_handles, load_folds, save
from src.evaluate import paired_bootstrap_ci

EXTRACTORS = ["ctranspath", "uni", "resnet50"]
MIL_CONFIGS = {"mean_pool": "mean_pool_{}_stainnorm_none", "abmil": "abmil_{}_stainnorm_none",
               "clam_sb": "{}_stainnorm_none"}
TEXT = {
    "tr": {"xlabel": "AUROC farkı (patch − MIL), %95 paired bootstrap GA", "legend": "Karşılaştırılan MIL"},
    "en": {"xlabel": "AUROC difference (patch − MIL), 95% paired bootstrap CI", "legend": "Compared MIL"},
}


def main():
    args = lang_args("fig_paired_delta.png")
    text = TEXT[args.lang]
    styles = agg_styles(args.lang)
    rows = []
    for extractor in EXTRACTORS:
        patch = load_folds(f"patch_{extractor}_stainnorm_none")
        for agg, template in MIL_CONFIGS.items():
            label, colour = styles[agg]
            r = paired_bootstrap_ci(patch, load_folds(template.format(extractor)), metric_names=["auroc"])["auroc"]
            rows.append((f"{EXTRACTOR_LABELS[extractor]}: patch − {label}", r["delta"],
                         r["ci_lower"], r["ci_upper"], colour))
    rows.reverse()

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for i, (_, delta, lo, hi, colour) in enumerate(rows):
        ax.errorbar(delta, i, xerr=[[delta - lo], [hi - delta]], fmt="o", color=colour,
                    ecolor=colour, elinewidth=1.6, capsize=3, markersize=7)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel(text["xlabel"])
    ax.set_xlim(-0.65, 0.65)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=legend_handles(styles[agg] for agg in MIL_CONFIGS), loc="upper right",
              fontsize=8.5, framealpha=0.9, title=text["legend"])
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
