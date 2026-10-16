import argparse
import os

from scripts.fig_lang import bold, config_label, latex_table, load_summary, number, with_ci, write

GENERATOR = "scripts/make_appendix_tables.py"
LATEX_DISC_HEADER = ["Aggregator/Extractor/Strategy", "AUROC [95\\% CI]", "AP [95\\% CI]", "Brier", "ECE"]
LATEX_OP_HEADER = ["Aggregator/Extractor/Strategy", "Sens (0.5)", "Spec (0.5)", "F1 (0.5)", "Thr. (Y)",
                   "Sens (Y) [95\\% CI]", "Spec (Y) [95\\% CI]", "F1 (Y)"]
LATEX_CAPTIONS = {
    "a1": "Macenko pipeline: discrimination and calibration (33 configurations), "
          "sorted by AUROC. Brackets are 95\\% bootstrap confidence intervals.",
    "a2": "Macenko pipeline: operating points (33 configurations), sorted by AUROC. "
          "(0.5) = default threshold; (Y) = leave-one-out Youden threshold. "
          "The Thr.\\ (Y) column is the median of the 19 patient-specific "
          "leave-one-out thresholds; each patient was classified with its own "
          "threshold (Section~\\ref{sec:evaluation}). Confidence intervals of the "
          "Youden metrics are computed with these decisions held fixed, so they are "
          "conditional on the threshold-selection rule.",
    "a3": "Raw pipeline (no stain normalisation): discrimination and calibration "
          "(33 configurations), sorted by AUROC; the second arm of the paired "
          "robustness comparison in Section~\\ref{sec:axes}.",
}
MARKDOWN_INTRO = [
    "# EK-A. KONFİGÜRASYON DÜZEYİNDE AYRINTILI SONUÇLAR",
    "",
    "Bu ek, Bölüm 5'te özetlenen 66 konfigürasyonun tamamının sonuçlarını sunmaktadır. Tüm",
    "değerler 19 katlı LOPOCV tahminlerinin havuzlanmasıyla hesaplanmış, güven aralıkları 1000",
    "yinelemeli sınıf-katmanlı bootstrap ile elde edilmiştir (Bölüm 4.9). Satırlar AUROC'a göre",
    "azalan sırada dizilmiştir. Güven aralıklarının genişliği ve örtüşmesi nedeniyle bu sıralama",
    "bir üstünlük sıralaması olarak okunmamalıdır (Bölüm 5.1).",
    "",
]
MARKDOWN_A2_NOTE = [
    "Youden eşiği sütunu, 19 hastaya ait dışarıda bir gözlem bırakma eşiklerinin ortancasıdır;",
    "her hasta kendi eşiğiyle sınıflandırılmıştır (Bölüm 4.9). Youden ölçütlerinin güven",
    "aralıkları, sabit tutulan bu kararlar üzerinden hesaplandığından eşik seçimi değişkenliğini",
    "içermez ve eşik seçme kuralına koşulludur.",
]
MARKDOWN_A3_NOTE = [
    "Renk normalizasyonu uygulanmamış konfigürasyonlar; Bölüm 5.2'deki eşleştirilmiş",
    "robustluk karşılaştırmasının ikinci kolunu oluşturur.",
]


def discrimination_rows(df, lang):
    return [[config_label(r, lang), with_ci(r, "auroc", 3, lang), with_ci(r, "ap", 3, lang),
             number(r.brier, 3, lang), number(r.ece, 3, lang)] for r in df.itertuples()]


def operating_rows(df, lang):
    return [[config_label(r, lang), number(r.sensitivity, 2, lang), number(r.specificity, 2, lang),
             number(r.f1, 2, lang), number(r.youden_threshold_median, 2, lang),
             with_ci(r, "sensitivity_youden", 2, lang), with_ci(r, "specificity_youden", 2, lang),
             number(r.f1_youden, 2, lang)] for r in df.itertuples()]


def markdown_table(header, alignment, rows):
    return "\n".join([f"| {' | '.join(header)} |", f"|{'|'.join(alignment)}|"]
                     + [f"| {' | '.join(r)} |" for r in rows])


def write_latex(mac, raw, out_dir):
    tables = {
        "a1": latex_table(GENERATOR, LATEX_CAPTIONS["a1"], "tab:a1", "lCCCC", bold(LATEX_DISC_HEADER),
                          discrimination_rows(mac, "en"), small=True),
        "a2": latex_table(GENERATOR, LATEX_CAPTIONS["a2"], "tab:a2", "lccccccc", bold(LATEX_OP_HEADER),
                          operating_rows(mac, "en"), small=True),
        "a3": latex_table(GENERATOR, LATEX_CAPTIONS["a3"], "tab:a3", "lCCCC", bold(LATEX_DISC_HEADER),
                          discrimination_rows(raw, "en"), small=True),
    }
    for key, tex in tables.items():
        write(os.path.join(out_dir, f"tab_appendix_{key}.tex"), tex)


def write_markdown(mac, raw, out_dir):
    disc_header = ["Toplayıcı / Çıkarıcı / Strateji", "AUROC [%95 GA]", "AP [%95 GA]", "Brier", "ECE"]
    op_header = ["Toplayıcı / Çıkarıcı / Strateji", "0,5: dyr", "0,5: özg", "0,5: F1", "Youden eşiği",
                 "Youden: dyr [%95 GA]", "Youden: özg [%95 GA]", "Youden: F1"]
    disc_alignment = ["---", "---", "---", "---:", "---:"]
    op_alignment = ["---", "---:", "---:", "---:", "---:", "---", "---", "---:"]
    parts = MARKDOWN_INTRO + [
        "## Tablo A.1. Macenko boru hattı: ayrım ve kalibrasyon (33 konfigürasyon)", "",
        markdown_table(disc_header, disc_alignment, discrimination_rows(mac, "tr")), "",
        "## Tablo A.2. Macenko boru hattı: çalışma noktaları (33 konfigürasyon)", "",
        *MARKDOWN_A2_NOTE, "",
        markdown_table(op_header, op_alignment, operating_rows(mac, "tr")), "",
        "## Tablo A.3. Ham boru hattı: ayrım ve kalibrasyon (33 konfigürasyon)", "",
        *MARKDOWN_A3_NOTE, "",
        markdown_table(disc_header, disc_alignment, discrimination_rows(raw, "tr")), "",
    ]
    write(os.path.join(out_dir, "ek-a-tablolar.md"), "\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Appendix A tables (all 66 configurations)")
    parser.add_argument("--lang", choices=("tr", "en"), default="en")
    parser.add_argument("--out", default=None, help="output directory; default outputs/tables/<lang>")
    args = parser.parse_args()
    out_dir = args.out or os.path.join("outputs", "tables", args.lang)

    df = load_summary()
    mac = df[df["macenko"]].sort_values("auroc", ascending=False)
    raw = df[~df["macenko"]].sort_values("auroc", ascending=False)
    (write_latex if args.lang == "en" else write_markdown)(mac, raw, out_dir)


if __name__ == "__main__":
    main()
