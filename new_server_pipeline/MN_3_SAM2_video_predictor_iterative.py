# Copyright (c) Meta Platforms, Inc. and affiliates.
"""SAM2-based video segmentation pipeline for creating worm masks.

This script converts session videos of planarian worms into binary segmentation
masks using Meta's SAM2 video predictor. It handles long periods of occlusion by
splicing each session video into un-occluded and occluded sections, propagating
labels through each independently, then stitching the results back together to
produce a full timeseries of masked worm data.

The pipeline:
  1. Discovers which session/worm videos still need masks.
  2. Loads the SAM2 video predictor model.
  3. For each video split, collects point prompts via an interactive UI, runs
     propagation, saves binary masks (.npz), and optionally generates QC GIFs.
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import csv
import io
import math
import re
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level default for the annotation object id used throughout SAM2 calls
# ---------------------------------------------------------------------------
ann_obj_id = 1

# ═══════════════════════════════════════════════════════════════════════════
# Configuration constants
# ═══════════════════════════════════════════════════════════════════════════

sessions_to_process = [
    '2025_10_15_10_20_58_trial_1_TC',
]

holylabs_base = "data/Raw_data/"
masks_base = "data/Masks/"
gifs_base = "data/GIFs/"
#boxes_base = "/n/holylabs/gershman_lab/Users/zkelso/DLC_Projects/WormTracking/labeled_data/"
video_folders_base = "data/temporary_jpgs/unlabeled-data"
video_split_csv = "hand_scored_datasheets/video_splits.csv"
fake_split_csv = "hand_scored_datasheets/fake_video_splits.csv"


# ═══════════════════════════════════════════════════════════════════════════
# Device setup
# ═══════════════════════════════════════════════════════════════════════════

def setup_device():
    """Select the best available compute device and configure autocast."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    return device


# ═══════════════════════════════════════════════════════════════════════════
# Discovery: find videos that still need masks
# ═══════════════════════════════════════════════════════════════════════════

def find_videos_to_process(sessions, video_folders_base, masks_base):
    """Scan *sessions* for worm video folders that lack a FINAL mask file.

    Returns
    -------
    done_session_worms : list[str]
        Paths to mask files that already exist.
    to_process : list[str]
        Full paths to video folders that still need processing.
    """
    done_session_worms = []
    to_process = []

    for session in sessions:
        print(f"\n{'=' * 80}")
        print(f"Checking session: {session}")
        print(f"{'=' * 80}")

        session_folder = Path(video_folders_base) / session

        if not session_folder.exists():
            print(f"WARNING: Session folder not found at {session_folder}")
            continue

        video_folders = list(session_folder.glob("*_fullvideo"))

        if video_folders:
            print(f"Found {len(video_folders)} video folder(s):")
            for folder in video_folders:
                print(f"  - {folder.name}")
        else:
            print(f"No video folders (*_fullvideo) found in {session}")
            continue

        for video_folder in video_folders:
            print(f"\nChecking: {video_folder.name}")

            jpg_count = sum(
                1
                for f in video_folder.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg"]
            )
            has_jpgs = jpg_count > 100

            if has_jpgs:
                print(f"  [OK] JPG files: {jpg_count} frames")
            else:
                print(f"  [X] JPG files: Only {jpg_count} frames (need >100)")
                continue

            npz_filename1 = f"{video_folder.name}_frame_split_FINAL_points_binary_masks.npz"
            npz_filename2 = f"{video_folder.name}_FINAL_boxes_binary_masks.npz"
            npz_path1 = Path(masks_base) / npz_filename1
            npz_path2 = Path(masks_base) / npz_filename2

            if npz_path1.exists():
                print(f"  [OK] Mask exists: {npz_filename1}")
                done_session_worms.append(str(npz_path1))
            elif npz_path2.exists():
                print(f"  [OK] Mask exists: {npz_filename2}")
                done_session_worms.append(str(npz_path2))
            else:
                print(f"  [PROCESS] Mask not found: {npz_filename1}")
                to_process.append(str(video_folder))

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nProcessed {len(sessions)} session(s)")
    print(f"Total videos found: {len(done_session_worms) + len(to_process)}")
    print(f"  Already have masks: {len(done_session_worms)}")
    print(f"  Need processing: {len(to_process)}")

    if to_process:
        print("\nVideos to process:")
        for video_path in to_process:
            print(f"  - [ ] {Path(video_path).name}")

    return done_session_worms, to_process


# ═══════════════════════════════════════════════════════════════════════════
# Video setup
# ═══════════════════════════════════════════════════════════════════════════

def setup_current_video(to_process, masks_base, video_folders_base):
    """Pick the next unprocessed video and load its frame list.

    Returns
    -------
    CURRENT_VIDEO : str
        Folder name of the video being processed.
    video_dir : str
        Full path to the directory of JPEG frames.
    frame_names : list[str]
        Sorted list of JPEG filenames in the video directory.
    """
    CURRENT_VIDEO = None
    for i in to_process:
        npz_path = masks_base + str(i.split("/")[-1:][0]) + "_binary_masks.npz"
        if os.path.exists(npz_path):
            to_process.remove(npz_path)
        else:
            CURRENT_VIDEO = str(i.split("/")[-1:][0])
            print("PROCESSING: " + CURRENT_VIDEO)
            break

    if CURRENT_VIDEO is None:
        raise RuntimeError("No videos left to process.")

    current_video_folder = CURRENT_VIDEO.split("_regions")[0]
    video_dir = f"{video_folders_base}/{current_video_folder}/{CURRENT_VIDEO}"

    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    frame_idx = 0
    plt.figure(figsize=(9, 6))
    plt.title(f"{CURRENT_VIDEO}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

    return CURRENT_VIDEO, video_dir, frame_names


# ═══════════════════════════════════════════════════════════════════════════
# Helper / visualisation functions  (kept exactly as in the notebook)
# ═══════════════════════════════════════════════════════════════════════════

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def pick_point(event, x, y, flags, param):
    curr_img, curr_coords, w_name = param
    event_counter = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        curr_coords.append([x, y])
        if event_counter > 2:
            cv2.circle(curr_img, (x, y), 7, (0, 0, 255), -1)
        elif event_counter < 3:
            cv2.circle(curr_img, (x, y), 7, (0, 255, 0), -1)
        else:
            print("Click events exceeded 4. Redo")
            return

        event_counter += 1
        cv2.putText(curr_img, str(len(curr_coords)), (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(w_name, curr_img)


def get_points_and_labels(frame, video_dir, frame_names):

    img_path = os.path.join(video_dir, frame_names[frame])
    print(f"Loading: {frame_names[frame]}")

    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load {img_path}")
        return

    display_img = image.copy()
    clicked_coords = []

    window_name = f"Frame {frame_names[frame]} - Add two points, subtract two points --> Enter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1000)

    cv2.setMouseCallback(window_name, pick_point, [display_img, clicked_coords, window_name])

    cv2.imshow(window_name, display_img)

    cv2.waitKey(0)

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    points = np.array(clicked_coords, dtype=np.float32)

    labels = np.array([1, 1, 0, 0], dtype=np.int32)

    return points, labels


def update_frame_data(file_path, frame_idx, coordinates):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Creating new database...")
        data = {}
        np.save(file_path, data)

    data = np.load(file_path, allow_pickle=True).item()

    if frame_idx not in data:
        data[frame_idx] = coordinates
        np.save(file_path, data)
        print(f"Frame {frame_idx} successfully saved to {file_path}.")
    else:
        print(f"Frame {frame_idx} is already in the file. No changes made.")


def get_boxes(frame, video_dir, frame_names, start, all_labeled_frames_path):
    img_path = os.path.join(video_dir, frame_names[frame])
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not load {img_path}")
        return None, None, None

    window_name = f"Frame {frame} | 1. GREEN: OBJECT -> ENTER | 2. RED: BACKGROUND -> ENTER"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1000)

    roi_pos = cv2.selectROI(window_name, image, showCrosshair=True)
    pos_box = np.array([roi_pos[0], roi_pos[1], roi_pos[0] + roi_pos[2], roi_pos[1] + roi_pos[3]], dtype=np.float32)

    display_img = image.copy()
    cv2.rectangle(display_img, (roi_pos[0], roi_pos[1]),
                  (roi_pos[0] + roi_pos[2], roi_pos[1] + roi_pos[3]), (0, 255, 0), 3)

    roi_neg = cv2.selectROI(window_name, display_img, showCrosshair=True)

    neg_points = []
    if roi_neg[2] > 0 and roi_neg[3] > 0:
        x_coords = np.linspace(roi_neg[0], roi_neg[0] + roi_neg[2], 3)
        y_coords = np.linspace(roi_neg[1], roi_neg[1] + roi_neg[3], 3)
        for x in x_coords:
            for y in y_coords:
                neg_points.append([x, y])

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    if len(neg_points) > 0:
        final_points = np.array(neg_points, dtype=np.float32)
        final_labels = np.zeros(len(neg_points), dtype=np.int32)
    else:
        final_points = np.empty((0, 2), dtype=np.float32)
        final_labels = np.empty((0,), dtype=np.int32)

    frame_data = {
        "points": final_points,
        "labels": final_labels,
        "box": pos_box,
    }

    update_frame_data(all_labeled_frames_path, frame, frame_data)

    return final_points, final_labels, pos_box


def dynamically_add_points(frame_list, prop_len, video_dir, frame_names, predictor, inference_state, start_frame, end_frame, mode="add"):

    for frame in frame_list:
        img_path = os.path.join(video_dir, frame_names[frame])
        print(f"Loading: {frame_names[frame]} (Frame {frame})")

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load {img_path}")
            return

        display_img = image.copy()
        clicked_coords = []

        window_name = f"Frame {frame} - Add two points, subtract two points --> Enter"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 1000)

        cv2.setMouseCallback(window_name, pick_point, [display_img, clicked_coords, window_name, mode])

        cv2.imshow(window_name, display_img)

        cv2.waitKey(0)

        cv2.destroyWindow(window_name)
        cv2.waitKey(1)

        points = np.array(clicked_coords, dtype=np.float32)

        labels = np.array([1, 1, 0, 0], dtype=np.int32)

        if clicked_coords == []:
            return
        else:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

            plt.figure(figsize=(9, 6))
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame])))
            show_points(points, labels, plt.gca())
            show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame,
        max_frame_num_to_track=end_frame - start_frame + 1,
        reverse=False,
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments


def display_new_frames(vis_frame_stride, start_frame, end_frame, video_dir, frame_names, video_segments):
    for out_frame_idx in range(start_frame, end_frame, vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


def make_a_gif(frame_names, video_dir, video_segments, output_path, start_frame):

    gif_frames = []
    vis_frame_stride = 1

    for out_frame_idx in range(1, len(video_segments), vis_frame_stride):
        out_frame_idx = start_frame + out_frame_idx
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        img_path = os.path.join(video_dir, frame_names[out_frame_idx])
        raw_image = Image.open(img_path)

        ax1.imshow(raw_image)
        ax1.set_title(f"Original Frame {out_frame_idx}")
        ax1.axis('off')

        ax2.imshow(raw_image)
        for out_mask in video_segments[out_frame_idx]:
            show_mask(out_mask, ax2, obj_id=1)
        ax2.set_title(f"Segmentation {out_frame_idx}")
        ax2.axis('off')

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        img = imageio.v2.imread(buf)
        gif_frames.append(img)

        plt.close(fig)

    imageio.mimsave(output_path + '_GIF.gif', gif_frames, fps=13, loop=0)

    print(f"Side-by-side GIF saved successfully at: {output_path}")


def _parse_bracket_lists(s):
    """
    From a string like "[510,1111,1270],[580,1170,1450]"
    return [[510,1111,1270], [580,1170,1450]].
    """
    groups = re.findall(r"\[(.*?)\]", s)
    lists = []
    for g in groups:
        g = g.strip()
        if not g:
            lists.append([])
            continue
        parts = [p.strip() for p in g.split(",") if p.strip() != ""]
        lists.append([int(p) for p in parts])
    return lists


def get_values_for_date(csv_path, target_date):
    """
    Read a CSV of format:
        name, [start1, start2, ...]           # old format (1 list)
    or:
        name, [start1, start2, ...], [end1, end2, ...]   # new format (2 lists)

    `target_date` is something like "..._fullvideo_points_GIF";
    we match the part before '_fullvideo'.

    Returns:
        start_frames (list of ints), stop_frames (list of ints)
    """
    base_name = target_date.split("_fullvideo")[0]

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            name = row[0].strip()
            if name != base_name:
                continue

            rest = ",".join(row[1:]).strip()
            lists = _parse_bracket_lists(rest)

            if not lists:
                split_starts, split_ends = [], []
            elif len(lists) == 1:
                split_starts = lists[0]
                split_ends = [f - 1 for f in split_starts]
            else:
                split_starts, split_ends = lists[0], lists[1]

            if (len(split_starts) == 0) & (len(split_ends) == 0):
                split_starts = [0]
                split_ends = [1799]

            return split_starts, split_ends

    raise ValueError(f"Video name {base_name!r} not found in CSV.")


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_predictor(device):
    """Build and return the SAM2 video predictor."""
    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "new_server_pipeline/sam_checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "new_server_pipeline/sam_checkpoints/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print("Loaded video predictor.")
    return predictor


# ═══════════════════════════════════════════════════════════════════════════
# Processing functions
# ═══════════════════════════════════════════════════════════════════════════

def process_single_split(
    predictor,
    video_dir,
    frame_names,
    frame_split_starts,
    frame_split_ends,
    split_idx,
    masks_base,
    gifs_base,
    CURRENT_VIDEO,
    make_gif=True,
):
    """Process one split of the video: prompt, propagate, save mask and GIF."""
    first_frame_idx = frame_split_starts[split_idx]
    last_frame_idx = frame_split_ends[split_idx]
    # all_points_path = (
    #     boxes_base
    #     + CURRENT_VIDEO
    #     + "frame_start_"
    #     + str(frame_split_starts)
    #     + "_SAM_points.npy"
    # )
    all_video_segments = {}

    inference_state = predictor.init_state(video_path=video_dir)

    points, labels = get_points_and_labels(first_frame_idx, video_dir, frame_names)

    frame_bundle = {
        "points": points,
        "labels": labels,
        "box": None,
    }

    if len(points) != len(labels):
        print(f"Adjusting labels: you clicked {len(points)} points.")
        labels = np.ones(len(points), dtype=np.int32)
        if len(points) > 2:
            labels[2:] = 0

    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=first_frame_idx,
        obj_id=1,
        points=points,
        labels=labels,
    )

    print("Propagating through the full video... this may take a moment.")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        all_video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()

    all_video_segments = {
        k: all_video_segments[k]
        for k in range(
            frame_split_starts[split_idx], frame_split_ends[split_idx]
        )
    }
    print(f"Propagation complete. {len(all_video_segments)} frames processed.")

    if make_gif:
        output_path = (
            gifs_base
            + CURRENT_VIDEO
            + "_frame_split_"
            + str(frame_split_starts[split_idx])
            + "_points"
        )
        make_a_gif(
            frame_names,
            video_dir,
            all_video_segments,
            output_path,
            frame_split_starts[split_idx],
        )

    sorted_frames = sorted(all_video_segments.keys())
    mask_stack = np.stack([all_video_segments[f] for f in sorted_frames])

    out_path = (
        masks_base
        + CURRENT_VIDEO
        + "_frame_split_"
        + str(frame_split_starts[split_idx])
        + "_"
        + str(frame_split_ends[split_idx])
        + "_points_binary_masks.npz"
    )
    np.savez_compressed(out_path, masks=mask_stack)


def process_all_splits(
    predictor,
    video_dir,
    frame_names,
    video_split_csv,
    CURRENT_VIDEO,
    masks_base,
    gifs_base,
    #boxes_base,
    make_gif=True,
):
    """Iterate over every split defined in the CSV and process each one."""
    frame_split_starts, frame_split_ends = get_values_for_date(
        video_split_csv, CURRENT_VIDEO
    )

    for i, item in enumerate(frame_split_starts):

        first_frame_idx = frame_split_starts[i]
        last_frame_idx = frame_split_ends[i]
        # all_points_path = (
        #     boxes_base
        #     + CURRENT_VIDEO
        #     + "frame_start_"
        #     + str(frame_split_starts)
        #     + "_SAM_points.npy"
        # )
        all_video_segments = {}

        inference_state = predictor.init_state(video_path=video_dir)

        points, labels = get_points_and_labels(first_frame_idx, video_dir, frame_names)

        frame_bundle = {
            "points": points,
            "labels": labels,
            "box": None,
        }

        if len(points) != len(labels):
            print(f"Adjusting labels: you clicked {len(points)} points.")
            labels = np.ones(len(points), dtype=np.int32)
            if len(points) > 2:
                labels[2:] = 0

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=first_frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
        )

        print("Propagating through the full video... this may take a moment.")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            all_video_segments[out_frame_idx] = (
                (out_mask_logits[0] > 0.0).cpu().numpy()
            )

        all_video_segments = {
            k: all_video_segments[k]
            for k in range(frame_split_starts[i], frame_split_ends[i])
        }
        print(f"Propagation complete. {len(all_video_segments)} frames processed.")

        if make_gif:
            output_path = (
                gifs_base
                + CURRENT_VIDEO
                + "_frame_split_"
                + str(frame_split_starts[i])
                + "_points"
            )
            make_a_gif(
                frame_names,
                video_dir,
                all_video_segments,
                output_path,
                frame_split_starts[i],
            )

        sorted_frames = sorted(all_video_segments.keys())
        mask_stack = np.stack([all_video_segments[f] for f in sorted_frames])

        out_path = (
            masks_base
            + CURRENT_VIDEO
            + "_frame_split_"
            + str(frame_split_starts[i])
            + "_"
            + str(frame_split_ends[i])
            + "_points_binary_masks.npz"
        )
        np.savez_compressed(out_path, masks=mask_stack)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = setup_device()

    _done, to_process = find_videos_to_process(
        sessions_to_process, video_folders_base, masks_base
    )

    CURRENT_VIDEO, video_dir, frame_names = setup_current_video(
        to_process, masks_base, video_folders_base
    )

    predictor = load_predictor(device)

    process_all_splits(
        predictor,
        video_dir,
        frame_names,
        video_split_csv,
        CURRENT_VIDEO,
        masks_base,
        gifs_base,
        #boxes_base,
        make_gif=True,
    )


if __name__ == "__main__":
    main()
