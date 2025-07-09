#!/usr/bin/env python3
"""
submit_video_jobs.py

Submit a DETR/SegFormer video-processing job per .mp4 in a directory.

Example:
  python submit_video_jobs.py \
      --video_dir /path/to/mp4s \
      --model_path nvidia/segformer-b5-finetuned-ade-640-640 \
      --main_script /my/project/main_gen_vids_and_meshes.py \
      --container_image ops:5000/segformer_torch:2.1
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ----------------------------------------------------------------------
# 1.  Utility: turn any model path into a short, human-readable tag
# ----------------------------------------------------------------------
def parse_model_name(model_path: str) -> str:
    """
    Derive a compact tag from a checkpoint or HF repo path.
    • If the path contains “…/trained_models/<exp_name>/…”, use <exp_name>.
    • Otherwise take the last two path components such as
          nvidia/segformer-b5-finetuned-ade-640-640
      → "nvidia" + first token before '-'  →  "nvidia_segformer".
    • Normalise: insert underscore between "resnet" and digits, replace
      non-alphanumerics with underscores, collapse repeats.
    """
    parts = Path(model_path).parts

    if "trained_models" in parts:
        idx = parts.index("trained_models")
        raw = parts[idx + 1] if idx + 1 < len(parts) else parts[-1]
    else:
        # e.g. (.../)?nvidia/segformer-b5-finetuned-ade-640-640[/…]
        if len(parts) >= 2:
            vendor, model = parts[-2], parts[-1]
        else:  # flat path like "resnet50.pth"
            vendor, model = "", parts[-1]
        # first token before any hyphen in model name
        model = model.split("-")[0]
        raw = f"{vendor}_{model}" if vendor else model

    # resnet101  → resnet_101
    tag = re.sub(r"(resnet)(\d+)", r"\1_\2", raw)
    # replace non-alnum with '_' and collapse repeats
    tag = re.sub(r"[^A-Za-z0-9]+", "_", tag)
    tag = re.sub(r"_{2,}", "_", tag).strip("_")
    return tag


# ----------------------------------------------------------------------
# 2.  Main driver
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit LSF video-processing jobs for each .mp4."
    )
    parser.add_argument(
        "--video_dir",
        default=(
            "/home/projects/bagon/andreyg/Projects/"
            "Object_reps_neural/Programming/detr/VIDEO_PROCESSING_TOOLS/"
            "generate_detection_videos_and_meshes/exp_1_videos_org"
        ),
        help="Directory with .mp4 videos.",
    )
    parser.add_argument(
        "--model_path",
        default=(
            "/home/projects/bagon/andreyg/Projects/Variable_Resolution_DETR/"
            "Programming/detr_var/trained_models/full_resolution_resnet101/"
            "box_and_segm/checkpoint.pth"
        ),
        help="Path to a checkpoint or HF repo ID.",
    )
    parser.add_argument(
        "--main_script",
        default=(
            "/home/projects/bagon/andreyg/Projects/Object_reps_neural/"
            "Programming/detr/VIDEO_PROCESSING_TOOLS/"
            "generate_detection_videos_and_meshes/main_gen_vids_and_meshes.py"
        ),
        help="Python entry script executed inside the job.",
    )
    parser.add_argument(
        "--container_image",
        default="ops:5000/detr_facebook_torch_v2:2.1",
        help="Docker image passed to LSF (LSB_CONTAINER_IMAGE).",
    )
    args = parser.parse_args()

    # Change working dir (optional but kept from original script)
    try:
        os.chdir("/home/projects/bagon/andreyg")
    except OSError as e:
        sys.exit(f"❌  Cannot cd to working directory: {e}")

    video_dir = Path(args.video_dir).expanduser()
    mp4_files = sorted(video_dir.glob("*.mp4"))
    if not mp4_files:
        sys.exit(f"❌  No .mp4 files found in {video_dir}")

    model_tag = parse_model_name(args.model_path)
    print(f"Model tag: {model_tag}")

    for mp4 in mp4_files:
        # Rename spaces → '+' for safety
        safe_path = mp4
        if " " in mp4.stem:
            safe_path = mp4.with_name(mp4.name.replace(" ", "+"))
            try:
                mp4.rename(safe_path)
            except OSError as e:
                print(f"⚠️  Skip {mp4}: rename failed ({e})")
                continue

        job_name = f"object_reps_neural_{safe_path.stem}"
        out_log = (
            f"/home/projects/bagon/andreyg/Projects/Object_reps_neural/"
            f"Cluster_runtime/model_testing/useCase_out_{model_tag}-{safe_path.stem}_%J.log"
        )
        err_log = out_log.replace("useCase_out", "useCase_error")

        cmd = (
            "../shared/seq_arr.sh "
            '-c "bsub '
            f' -env LSB_CONTAINER_IMAGE={args.container_image} '
            " -app docker-gpu "
            " -gpu num=1:j_exclusive=yes "
            " -q waic-short "
            " -R rusage[mem=64000] "
            " -R affinity[thread*24] "
            " -R select[hname!=ibdgxa01] "
            " -R select[hname!=ibdgx010] "
            f' -o {out_log} '
            f' -e {err_log} '
            f' -J "{job_name}" '
            " -H python3 "
            f'{args.main_script} '
            f'--model_path {args.model_path} '
            f'--video_path {safe_path} '
            '--n_blobs 2" '
            "-e 1 -d ended"
        )

        print(f"🛈  Submitting {safe_path} → {job_name}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌  Job submission failed: {e}")

        print("⏳  Sleeping 10 s before next job…\n")
        time.sleep(10)


if __name__ == "__main__":
    main()
