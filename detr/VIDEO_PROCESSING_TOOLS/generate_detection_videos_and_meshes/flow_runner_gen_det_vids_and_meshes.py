#!/usr/bin/env python3

import os
import glob
import re
import subprocess
import time
import argparse

def parse_model_name(model_path):
    """
    Extract a folder name after 'trained_models/' from the model_path.
    Then insert an underscore between 'resnet' and the digits, so:
      variable_pretrained_resnet101 -> variable_pretrained_resnet_101
    Adapt/disable that part if you prefer the original name.
    """
    model_dir = os.path.dirname(model_path)
    parts = model_dir.split("/")
    try:
        tm_index = parts.index("trained_models")
        model_name = parts[tm_index + 1]  # e.g. 'variable_pretrained_resnet101'
    except (ValueError, IndexError):
        model_name = parts[-1]

    # Insert underscore between 'resnet' and digits (e.g. resnet101 -> resnet_101).
    model_name = re.sub(r'(resnet)(\d+)', r'\1_\2', model_name)
    return model_name

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Video generation script with optional arguments.")
    parser.add_argument(
        "--video_dir",
        default=(
            "/home/projects/bagon/andreyg/Projects/"
            "Object_reps_neural/Programming/detr/VIDEO_PROCESSING_TOOLS/"
            "generate_detection_videos_and_meshes/exp_1_videos_org"
        ),
        help="Path to the directory containing .mp4 files (default points to exp2TTC_files).",
        required=False
    )
    parser.add_argument(
        "--model_path",
        default=(
            "/home/projects/bagon/andreyg/Projects/Variable_Resolution_DETR/"
            "Programming/detr_var/trained_models/full_resolution_resnet101/"
            "box_and_segm/checkpoint.pth"
        ),
        help="Path to the model checkpoint file (default is resnet101 checkpoint).",
        required=False
    )
    args = parser.parse_args()

    # 1) Ensure we're in /home/projects/bagon/andreyg
    try:
        os.chdir("/home/projects/bagon/andreyg")
    except Exception as e:
        print(f"Error: Unable to cd into /home/projects/bagon/andreyg.\n{e}")
        return

    # 2) Configuration from command-line or defaults
    VIDEO_DIR = args.video_dir
    MODEL_PATH = args.model_path

    # 3) Parse model name
    model_name = parse_model_name(MODEL_PATH)
    print(f"Extracted model name: {model_name}")

    # 4) Find all .mp4 files
    mp4_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if not mp4_files:
        print(f"No .mp4 files found in {VIDEO_DIR}. Exiting.")
        return

    # 5) Loop over each .mp4 file
    for old_path in mp4_files:
        old_basename = os.path.basename(old_path)      # e.g. "BConcave+hConcave 4100.mp4"
        old_name_root, ext = os.path.splitext(old_basename)  # ("BConcave+hConcave 4100", ".mp4")

        # (a) Rename file if it contains space(s)
        if " " in old_name_root:
            new_name_root = old_name_root.replace(" ", "+")
            new_basename = new_name_root + ext  # e.g. "BConcave+hConcave+4100.mp4"
            new_path = os.path.join(os.path.dirname(old_path), new_basename)

            print(f"Renaming:\n  {old_path}\n  => {new_path}")
            try:
                os.rename(old_path, new_path)
            except OSError as e:
                print(f"Failed to rename {old_path}: {e}")
                continue
        else:
            # No spaces => no rename needed
            new_path = old_path

        # Use the new (renamed) path
        base_name = os.path.basename(new_path)        # e.g. "BConcave+hConcave+4100.mp4"
        prefix_name = os.path.splitext(base_name)[0]  # e.g. "BConcave+hConcave+4100"
        prefix_plus = prefix_name  # Already has no spaces

        print(f"Submitting job for: {new_path}")

        JOB_NAME = f"object_reps_neural_{prefix_plus}"

        # 6) Construct the command (no quotes needed around --video_path now)
        cmd = (
            "../shared/seq_arr.sh "
            "-c \"bsub "
            " -env LSB_CONTAINER_IMAGE=ops:5000/detr_facebook_torch_v2:2.1 "
            " -app docker-gpu "
            " -gpu num=1:j_exclusive=yes "
            " -q waic-short "
            " -R rusage[mem=64000] "
            " -R affinity[thread*24] "
            " -R select[hname!=ibdgxa01] "
            " -R select[hname!=ibdgx010] "
            f" -o /home/projects/bagon/andreyg/Projects/Object_reps_neural/Cluster_runtime/model_testing/useCase_out_{model_name}-{prefix_plus}_%J.log "
            f" -e /home/projects/bagon/andreyg/Projects/Object_reps_neural/Cluster_runtime/model_testing/useCase_error_{model_name}-{prefix_plus}_%J.log "
            f" -J \"{JOB_NAME}\" "
            " -H python3 "
            "/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/"
            "VIDEO_PROCESSING_TOOLS/generate_detection_videos_and_meshes/main_gen_vids_and_meshes.py "
            f"--model_path {MODEL_PATH} "
            f"--video_path {new_path} "
            "--n_blobs 2\" "
            "-e 1 -d ended"
        )

        # 7) Submit job
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as err:
            print(f"Failed to submit job for {new_path}. Error: {err}")

        # 8) Wait 10 seconds
        print("Waiting 10 seconds before next submission...\n")
        time.sleep(10)

if __name__ == "__main__":
    main()
