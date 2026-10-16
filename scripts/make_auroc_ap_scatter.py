import matplotlib.pyplot as plt

from scripts.fig_lang import agg_styles, lang_args, load_summary, prevalence, save

TEXT = {
    "tr": {"ap_chance": "AP rastgele = prevalans ({:.2f})", "auroc_chance": "AUROC rastgele (0,5)",
           "ylabel": "Ortalama kesinlik (AP)", "legend": "Toplama yöntemi"},
    "en": {"ap_chance": "AP chance = prevalence ({:.2f})", "auroc_chance": "AUROC chance (0.5)",
           "ylabel": "Average precision (AP)", "legend": "Aggregation"},
}


def main():
    args = lang_args("fig_auroc_ap.png")
    text = TEXT[args.lang]
    mac = load_summary(macenko_only=True)
    chance_ap = prevalence(mac)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    for agg, (label, colour) in agg_styles(args.lang).items():
        rows = mac[mac["aggregator"] == agg]
        ax.scatter(rows["auroc"], rows["ap"], c=colour, s=70, alpha=0.85,
                   edgecolors="white", linewidths=0.6, label=label)
    ax.axvline(0.5, color="red", ls="--", lw=1, alpha=0.5)
    ax.axhline(chance_ap, color="red", ls="--", lw=1, alpha=0.5)
    ax.text(0.205, chance_ap + 0.01, text["ap_chance"].format(chance_ap), fontsize=8, color="red", alpha=0.8)
    ax.text(0.505, 0.02, text["auroc_chance"], fontsize=8, color="red", alpha=0.8, rotation=90, va="bottom")
    ax.set_xlabel("AUROC")
    ax.set_ylabel(text["ylabel"])
    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, title=text["legend"])
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
