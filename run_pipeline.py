import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.models.aggregators import AGGREGATORS
from src.preprocess import load_config

ROOT = Path(__file__).resolve().parent
STAGES = ("preprocess", "extract", "stainnorm", "train", "evaluate")
STAGE_MODULES = {
    "preprocess": "src.preprocess",
    "extract": "src.extract_features",
    "stainnorm": "scripts.stainnorm_extract",
    "evaluate": "src.evaluate",
}


def stage_steps(stage, config, extractors):
    if stage != "train":
        return [[STAGE_MODULES[stage], "--config", config]]
    steps = []
    for extractor in [*extractors, *(f"{e}_stainnorm" for e in extractors)]:
        steps += [["src.train", "--config", config, "--aggregator", aggregator, "--extractor", extractor]
                  for aggregator in AGGREGATORS]
        steps.append(["src.patch_baseline", "--config", config, "--extractor", extractor])
    return steps


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline: patches, features, 66 LOPOCV configs, metrics")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", choices=STAGES, default="preprocess",
                        help="first stage to run (e.g. 'train' when features already exist)")
    args = parser.parse_args()

    extractors = load_config(args.config)["feature_extraction"]["extractors"]
    failed = []
    t0 = time.time()
    for stage in STAGES[STAGES.index(args.start):]:
        if stage == "evaluate" and failed:
            print("\nSkipping evaluation because earlier steps failed.")
            break
        for step in stage_steps(stage, args.config, extractors):
            print(f"\n{'#' * 70}\n# {' '.join(step)}\n{'#' * 70}", flush=True)
            if subprocess.run([sys.executable, "-u", "-m", *step], cwd=ROOT).returncode != 0:
                failed.append(" ".join(step))

    print(f"\nPipeline finished in {(time.time() - t0) / 60:.1f} min")
    for step in failed:
        print(f"FAILED: {step}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
