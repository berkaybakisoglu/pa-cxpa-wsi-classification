import argparse

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scripts.fig_lang import lang_args, save
from src.preprocess import detect_tissue, extract_patch_coords, find_target_level, get_patient_list, load_config, open_wsi

TEXT = {
    "tr": {"a": "(a) Küçük görüntü (thumbnail)", "b": "(b) HSV doygunluk eşiği (>{sat})",
           "c": "(c) Morfolojik temizleme", "d": "(d) Yama ızgarası (≥%{pct} doku)",
           "accepted": "Kabul edilen yama"},
    "en": {"a": "(a) Thumbnail", "b": "(b) HSV saturation threshold (>{sat})",
           "c": "(c) Morphological cleanup", "d": "(d) Patch grid (≥{pct}% tissue)",
           "accepted": "Accepted patch"},
}


def thumbnail_boxes(coords, patch_size_l0, scale, mask_shape):
    (scale_x, scale_y), (mask_h, mask_w) = scale, mask_shape
    boxes = []
    for x, y in coords:
        x0 = max(0, min(int(x / scale_x), mask_w - 1))
        x1 = max(x0 + 1, min(int((x + patch_size_l0) / scale_x), mask_w))
        y0 = max(0, min(int(y / scale_y), mask_h - 1))
        y1 = max(y0 + 1, min(int((y + patch_size_l0) / scale_y), mask_h))
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Tissue detection and patch grid on one slide")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--patient_id", required=True)
    args = lang_args("fig_patch_pipeline.png", parser)
    text = TEXT[args.lang]

    cfg = load_config(args.config)
    prep = cfg["preprocessing"]
    patient_dir = next(d for pid, _, d in get_patient_list(cfg) if pid == args.patient_id)
    with open_wsi(patient_dir) as slide:
        target_level, _ = find_target_level(slide, prep["target_mpp"])
        clean_mask, thumbnail, scale = detect_tissue(slide, prep["thumbnail_downsample"],
                                                     prep["saturation_threshold"])
        coords = extract_patch_coords(slide, clean_mask, target_level, prep["patch_size"],
                                      prep["tissue_threshold"], scale_factors=scale)
        patch_size_l0 = int(prep["patch_size"] * slide.level_downsamples[target_level])
    raw_mask = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)[:, :, 1] > prep["saturation_threshold"]
    print(f"{len(coords)} accepted patches")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
    for ax, image in zip(axes, (thumbnail, raw_mask.astype("uint8"), clean_mask, thumbnail)):
        ax.imshow(image, cmap=None if image is thumbnail else "gray")
    for x, y, w, h in thumbnail_boxes(coords, patch_size_l0, scale, clean_mask.shape):
        axes[3].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#2ca02c", linewidth=0.35))
    titles = (text["a"], text["b"].format(sat=prep["saturation_threshold"]), text["c"],
              text["d"].format(pct=int(prep["tissue_threshold"] * 100)))
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[3].add_patch(Rectangle((0, 0), 0, 0, fill=False, edgecolor="#2ca02c", linewidth=1.5,
                                label=text["accepted"]))
    axes[3].legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
