import matplotlib.pyplot as plt

from scripts.fig_lang import agg_styles, lang_args, legend_handles, load_summary, save

TEXT = {
    "tr": {"xlabel": "AUROC (ayrım) →", "ylabel": "ECE (kalibrasyon hatası) — düşük daha iyi",
           "legend": "Toplama yöntemi"},
    "en": {"xlabel": "AUROC (discrimination) →", "ylabel": "ECE (calibration error) — lower is better",
           "legend": "Aggregation"},
}


def main():
    args = lang_args("fig_auroc_ece.png")
    text = TEXT[args.lang]
    styles = agg_styles(args.lang)
    mac = load_summary(macenko_only=True)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for agg, (label, colour) in styles.items():
        rows = mac[mac["aggregator"] == agg]
        ax.scatter(rows["auroc"], rows["ece"], c=colour, s=70, alpha=0.85,
                   edgecolors="white", linewidths=0.6, label=label)
    ax.set_xlabel(text["xlabel"])
    ax.set_ylabel(text["ylabel"])
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(handles=legend_handles(styles.values()), loc="lower left", fontsize=9,
              framealpha=0.9, title=text["legend"])
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
