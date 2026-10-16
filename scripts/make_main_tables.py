import argparse
import glob
import os

import h5py

from scripts.fig_lang import (EXTRACTOR_LABELS, bold, config_label, latex_table, load_summary, number,
                              with_ci, write)
from src.preprocess import load_config

GENERATOR = "scripts/make_main_tables.py"
EXTRACTORS = ["uni", "ctranspath", "resnet50"]
OPERATING_POINT_CONFIGS = ["uni_stainnorm_none", "patch_uni_stainnorm_none", "abmil_resnet50_stainnorm_none"]


def signed(x):
    text = f"{abs(x):.2f}"
    return text if text == "0.00" else ("+" if x > 0 else "$-$") + text


def thousands(n):
    return f"{n:,}"


def cohort_table(cfg):
    counts = {0: [], 1: []}
    for path in glob.glob(os.path.join(cfg["data"]["output_dir"], "patches", "*_coords.h5")):
        with h5py.File(path, "r") as f:
            counts[int(f.attrs["label"])].append(int(f.attrs["num_patches"]))

    def row(name, n):
        return [name, str(len(n)), thousands(sum(n)), f"{thousands(min(n))}--{thousands(max(n))}",
                thousands(round(sum(n) / len(n)))]

    rows = [row("Pleomorphic adenoma (PA, benign)", counts[0]), row("Carcinoma ex PA (CXPA, malignant)", counts[1]),
            "\\midrule", row("Total", counts[0] + counts[1])]
    return latex_table(
        GENERATOR, "Summary of the dataset. Each patient contributes one whole-slide image, so the number of "
        "patients equals the number of slides.", "tab:cohort", "lCCCC",
        bold(["Class", "Patients (Slides)", "Total Patches", "Patches per Patient (Min--Max)", "Mean per Patient"]),
        rows)


def design_table(df):
    is_patch = df["aggregator"] == "patch_level"
    n_colours, n_extractors = df["macenko"].nunique(), df["extractor"].nunique()
    rows = [["MIL", str(n_colours), str(n_extractors), "3 (mean pooling, ABMIL, CLAM-SB)",
             "3 (none, oversample, SMOTE)", str((~is_patch).sum())],
            ["Patch-level", str(n_colours), str(n_extractors), "1", "2 (none, oversample)", str(is_patch.sum())],
            "\\midrule", f"Total & & & & & {len(df)}\\\\"]
    return latex_table(
        GENERATOR, "Experimental design and number of configurations. SMOTE is not applied to the patch-level "
        "model (Section~\\ref{sec:imbalance}), so this family has only two imbalance strategies.", "tab:design",
        "lCCCCC", bold(["Model Family", "Colour", "Feature Extractor", "Aggregator", "Imbalance Strategy",
                        "Configurations"]), rows)


def top_configs_table(mac):
    rows = [[config_label(r, "en"), with_ci(r, "auroc", 3, "en"), with_ci(r, "ap", 3, "en"),
             number(r.brier, 3, "en"), number(r.ece, 3, "en")] for r in mac.head(8).itertuples()]
    return latex_table(
        GENERATOR, "The eight configurations with the highest AUROC in the Macenko "
        "pipeline. Values in brackets are 95\\% bootstrap confidence intervals.", "tab:topconfigs", "lCCCC",
        bold(["Aggregator/Extractor/Strategy", "AUROC [95\\% CI]", "AP [95\\% CI]", "Brier", "ECE"]), rows)


def patch_vs_mil_table(mac):
    none = mac[mac["strat"] == "none"]
    rows = []
    for extractor in EXTRACTORS:
        configs = none[none["extractor"] == extractor]
        patch = configs[configs["aggregator"] == "patch_level"].iloc[0]
        mil = configs[configs["aggregator"] != "patch_level"].sort_values("auroc", ascending=False).iloc[0]
        rows.append([EXTRACTOR_LABELS[extractor], number(patch.auroc, 3, "en"), number(mil.auroc, 3, "en"),
                     signed(patch.auroc - mil.auroc), number(patch.ap, 3, "en"), number(mil.ap, 3, "en"),
                     signed(patch.ap - mil.ap)])
    return latex_table(
        GENERATOR, "Patch-level model versus the best MIL aggregator for each feature extractor (Macenko pipeline, "
        "configurations without imbalance correction). $\\Delta$ = patch $-$ best MIL; negative values favour MIL. "
        "The best MIL aggregator is CLAM-SB for UNI and CTransPath and ABMIL for ResNet-50.", "tab:patchvsmil",
        "lCCCCCC", ["\\textbf{Feature Extractor}", "\\textbf{Patch AUROC}", "\\textbf{Best MIL AUROC}",
                    "\\boldmath{$\\Delta$}\\textbf{AUROC}", "\\textbf{Patch AP}", "\\textbf{Best MIL AP}",
                    "\\boldmath{$\\Delta$}\\textbf{AP}"], rows)


def operating_points_table(mac):
    best = mac.iloc[0]
    chosen = [best] + [mac[mac["config"] == name].iloc[0] for name in OPERATING_POINT_CONFIGS]
    rows = []
    for r in chosen:
        label = config_label(r, "en") + (" (best)" if r.config == best.config else "")
        at_half = "/".join(number(getattr(r, m), 2, "en") for m in ("sensitivity", "specificity", "f1"))
        rows.append([label, at_half, with_ci(r, "sensitivity_youden", 2, "en"),
                     with_ci(r, "specificity_youden", 2, "en"), number(r.f1_youden, 2, "en")])
    return latex_table(
        GENERATOR, "Threshold-dependent performance of representative configurations at two operating points "
        "(Macenko pipeline, configurations without imbalance correction; the best configuration uses SMOTE). "
        "Values in brackets are 95\\% bootstrap confidence intervals. Sensitivity is the fraction of the five "
        "malignant cases detected.", "tab:operating", "lCCCC",
        bold(["Configuration", "Threshold 0.5: Sens/Spec/F1", "Youden: Sens [95\\% CI]",
              "Youden: Spec [95\\% CI]", "Youden: F1"]), rows)


def main():
    parser = argparse.ArgumentParser(description="Results tables of the paper (Tables 1-5)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=os.path.join("outputs", "tables", "en"))
    args = parser.parse_args()

    df = load_summary()
    mac = df[df["macenko"]].sort_values("auroc", ascending=False)
    tables = {"tab_design": design_table(df), "tab_top_configs": top_configs_table(mac),
              "tab_patch_vs_mil": patch_vs_mil_table(mac), "tab_operating_points": operating_points_table(mac)}
    cfg = load_config(args.config)
    if glob.glob(os.path.join(cfg["data"]["output_dir"], "patches", "*_coords.h5")):
        tables["tab_cohort"] = cohort_table(cfg)
    else:
        print("No patch coordinates found; skipping the cohort table.")
    for name, tex in tables.items():
        write(os.path.join(args.out, f"{name}.tex"), tex)


if __name__ == "__main__":
    main()
