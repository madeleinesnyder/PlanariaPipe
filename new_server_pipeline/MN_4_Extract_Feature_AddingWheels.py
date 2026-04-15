"""Mask analysis and feature extraction pipeline (Long form).

Takes masks created from spatially segmented, temporally intact continuous worm
videos and analyzes masks across entire recordings. Generates frame-by-frame
measurements of area, perimeter, circularity, orientation, PCA coordinates, and
other morphological features. Includes stitching of split mask files, outlier
filtering, interpolation, and batch processing across sessions.

Usage::

    python LF_4_Extract_Features_AddingWheels.py
"""

# ============================================================================
# IMPORTS
# ============================================================================

import csv
import hashlib
import json
import os
import pickle
import re
import shutil
import time
import traceback
from collections import defaultdict
from itertools import chain, combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, median_filter
from scipy.signal import medfilt
from sklearn import preprocessing
from sklearn.decomposition import PCA

# ============================================================================
# VIDEO SELECTION / CONFIGURATION
# ============================================================================

VIDEO_PREFIX_FILTER = [
    '2025_10_15_10_20_58_trial_1_TC'
]

VIDEO_GROUP_FILTER = None
EXCLUSION_FILTER = []
LABEL_TYPE = "points"
reset_reference = False


# ============================================================================
# STEP 2: Feature extraction functions
# ============================================================================

def calculate_max_distance(binary_mask, visualize=False):
    """
    Calculate the maximum distance between any two points in a binary mask.

    Parameters:
    binary_mask: numpy.ndarray
        2D binary array where True/1 represents the object
    visualize: bool
        If True, displays the mask with the maximum distance line

    Returns:
    float: Maximum distance between any two points
    tuple: Coordinates of the two most distant points ((x1,y1), (x2,y2))
    """
    points = np.column_stack(np.where(binary_mask))

    if len(points) < 2:
        return 0, ((0, 0), (0, 0))

    max_distance = 0
    max_points = None

    for (y1, x1), (y2, x2) in combinations(points, 2):
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance > max_distance:
            max_distance = distance
            max_points = ((x1, y1), (x2, y2))

    if visualize:
        visualize_max_distance(binary_mask, max_points, max_distance)

    return max_distance, max_points


def visualize_max_distance(binary_mask, max_points, max_distance):
    """Visualize the binary mask with the maximum distance line."""
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_mask, cmap="gray", interpolation="nearest")

    (x1, y1), (x2, y2) = max_points
    plt.plot(
        [x1, x2], [y1, y2], "r-", linewidth=2,
        label=f"Max Distance: {max_distance:.2f}",
    )
    plt.plot([x1, x2], [y1, y2], "go", markersize=10, label="Endpoints")

    plt.title("Binary Mask with Maximum Distance")
    plt.legend()
    plt.grid(True)
    plt.colorbar(label="Binary Mask Value")
    plt.show()


def calculate_mask_area(binary_image):
    """
    Calculate the area of black pixels in a binary image.

    Returns:
    float: Area in pixels
    float: Percentage of total image area
    """
    if binary_image.max() > 1:
        binary_image = binary_image / 255.0
    if np.mean(binary_image) > 0.5:
        binary_image = 1 - binary_image

    total_pixels = binary_image.shape[0] * binary_image.shape[1]
    black_pixels = np.sum(binary_image)
    percentage = (black_pixels / total_pixels) * 100

    return black_pixels, percentage


def calculate_perimeter(binary_image):
    """
    Calculate the perimeter of black shapes in a binary image.

    Returns:
    float: Perimeter length in pixels
    list: List of contour points
    """
    if binary_image.dtype != np.uint8:
        binary_image = (binary_image * 255).astype(np.uint8)
    if np.mean(binary_image) > 127:
        binary_image = 255 - binary_image

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return 0, []

    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, closed=True)
    contour_points = largest_contour.squeeze().tolist()

    return perimeter, contour_points, contours


def calculate_orientation(binary_image):
    """
    Calculate the angle of the best fit line through the shape using PCA.

    Returns:
    float: Angle in degrees from the horizontal (x-axis), or NaN if empty mask
    numpy array: Main direction vector, or [NaN, NaN] if empty mask
    """
    y_coords, x_coords = np.nonzero(binary_image)
    coords = np.column_stack((x_coords, y_coords))

    if len(coords) < 2:
        return np.nan, np.array([np.nan, np.nan])

    pca = PCA(n_components=2)
    pca.fit(coords)

    main_direction = pca.components_[0]
    angle_rad = np.arctan2(main_direction[1], main_direction[0])
    angle_deg = np.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 180

    return angle_deg, main_direction


def calculate_centroid(binary_image):
    """
    Calculate the centroid of the black shape in a binary image.

    Returns:
    tuple: (x, y) coordinates of centroid
    """
    binary_image = (binary_image * 255).astype(np.uint8)
    moments = cv2.moments(binary_image)

    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = 0, 0

    return (cx, cy)


def make_wheel(perim, com):
    nPoints = 100
    pos = np.linspace(0, len(perim[0]), nPoints + 1).astype(int)
    pos = pos[:-1]
    worm_wheel = []
    for i in pos:
        worm_wheel.append(np.linalg.norm(com - perim[0][i]))
    loc = np.where(worm_wheel == np.max(worm_wheel))
    if len(loc[0]) > 1:
        loc = loc[0]
    worm_wheel = np.roll(worm_wheel, -loc[0])
    return worm_wheel


def analyze_shape(binary_image):
    """
    Perform comprehensive shape analysis including area, perimeter, centroids,
    orientation and their ratios.

    Returns:
    dict: Dictionary containing all shape measurements and derived metrics.
          Returns NaN values if mask is empty or invalid.
    """
    if binary_image.max() > 1:
        binary_image = (binary_image > 127).astype(np.uint8)
    else:
        binary_image = binary_image.astype(np.uint8)

    if np.sum(binary_image) == 0:
        return {
            "area": np.nan,
            "area_percentage": np.nan,
            "perimeter": np.nan,
            "theta": np.full(100, np.nan),
            "area_perimeter_ratio": np.nan,
            "circularity": np.nan,
            "concavity": np.nan,
            "hull_area": np.nan,
            "num_contour_points": 0,
            "contour_points": [],
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "orientation_angle": np.nan,
            "main_direction_x": np.nan,
            "main_direction_y": np.nan,
        }

    area, area_percentage = calculate_mask_area(binary_image)
    perimeter, contour_points, contour = calculate_perimeter(binary_image)
    centroid = calculate_centroid(binary_image)
    theta = make_wheel(contour, centroid)
    angle, direction = calculate_orientation(binary_image)

    area_perimeter_ratio = area / perimeter if perimeter > 0 else np.nan
    circularity = (
        (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else np.nan
    )

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return {
            "area": np.nan,
            "area_percentage": np.nan,
            "perimeter": np.nan,
            "theta": np.full(100, np.nan),
            "area_perimeter_ratio": np.nan,
            "circularity": np.nan,
            "concavity": np.nan,
            "hull_area": np.nan,
            "num_contour_points": 0,
            "contour_points": [],
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "orientation_angle": np.nan,
            "main_direction_x": np.nan,
            "main_direction_y": np.nan,
        }

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h != 0 else np.nan
    extent = float(area) / (w * h) if w * h != 0 else np.nan
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else np.nan
    concavity = (hull_area - area) / hull_area if hull_area != 0 else np.nan

    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments_log = [
        -np.sign(h) * np.log10(abs(h)) if h != 0 else 0 for h in hu_moments
    ]

    metrics = {
        "area": area,
        "area_percentage": area_percentage,
        "perimeter": perimeter,
        "theta": theta,
        "area_perimeter_ratio": area_perimeter_ratio,
        "circularity": circularity,
        "concavity": concavity,
        "hull_area": hull_area,
        "num_contour_points": len(contour_points),
        "contour_points": contour_points,
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
        "orientation_angle": angle,
        "main_direction_x": direction[0],
        "main_direction_y": direction[1],
    }

    return metrics


def plot_single_shape_analysis(metrics, figsize=(10, 10)):
    """Visualize the contour points and analysis results from analyze_shape()."""
    plt.figure(figsize=figsize)

    contour_points = np.array(metrics["contour_points"])
    if len(contour_points.shape) == 1:
        contour_points = contour_points.reshape(-1, 2)

    plt.plot(contour_points[:, 0], contour_points[:, 1], "b.", label="Contour Points")
    plt.plot(
        metrics["centroid_x"], metrics["centroid_y"], "r*", markersize=15,
        label="Centroid",
    )

    if "orientation_angle" in metrics and "main_direction_x" in metrics:
        line_length = max(contour_points.max() - contour_points.min(), 50)
        dx = metrics["main_direction_x"] * line_length / 2
        dy = metrics["main_direction_y"] * line_length / 2

        plt.plot(
            [metrics["centroid_x"] - dx, metrics["centroid_x"] + dx],
            [metrics["centroid_y"] - dy, metrics["centroid_y"] + dy],
            "g--",
            label=f"Orientation: {metrics['orientation_angle']:.1f}°",
        )

    info_text = (
        f"Area: {metrics['area']:.1f} pixels\n"
        f"Perimeter: {metrics['perimeter']:.1f} pixels\n"
        f"Circularity: {metrics['circularity']:.3f}\n"
        f"Points: {metrics['num_contour_points']}"
    )
    plt.text(
        0.02, 0.98, info_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.axis("equal")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.title("Shape Analysis Visualization")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt.gcf()


def plot_smoothed_thing(thing, window, title_thing, ylab, filter_method, start, end):
    """Plot smoothed time series data."""
    plt.figure(figsize=(10, 6))

    thing_array = np.array(thing)
    valid_mask = [x is not None for x in thing_array]

    if filter_method == "movmed":
        to_plot_thing = np.array([None] * len(thing_array))
        if any(valid_mask):
            valid_thing = np.array([x for x in thing_array[valid_mask]], dtype=float)
            to_plot_thing[valid_mask] = medfilt(valid_thing, window)

    plt.plot(to_plot_thing[start:end])
    plt.xlabel("Frame")
    plt.ylabel(ylab)
    plt.title(title_thing)
    plt.legend()
    plt.show()


def plot_scaled_thing(thing, thing_name, scale):
    """Plot normalized or raw feature traces."""
    plt.figure(figsize=(12, 6))

    if scale == 1:
        normalized_thing_stack = []
        for row in thing:
            min_val = min(row)
            max_val = max(row)
            normalized = [(x - min_val) / (max_val - min_val) for x in row]
            normalized_thing_stack.append(normalized)
            plt.plot(normalized, color="lightgrey", alpha=0.5)

        mean = np.nanmean(normalized_thing_stack, axis=0)
        plt.plot(mean, color="black", linewidth=2, label="Mean")
        plt.title("NORMALIZED " + thing_name)
    else:
        for row in thing:
            plt.plot(row, color="lightgrey", alpha=0.5)

        mean = np.nanmean(thing, axis=0)
        plt.plot(mean, color="black", linewidth=2, label="Mean")
        plt.title(thing_name)

    plt.xlabel("Time Point")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


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

    ``target_date`` is something like "..._fullvideo_points_GIF";
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


def check_date_in_splits(csv_path, target_date):
    base_name = target_date.split("_fullvideo")[0]
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip() == base_name:
                return True
    return False


def parse_frame_range_from_name(filename):
    """
    Extract (start, end) from names like:
    ..._fullvideo_frame_split_1270_1450_points_binary_masks(.npz)
    """
    base = os.path.basename(filename)
    m = re.search(r"_frame_split_(\d+)_(\d+)_points_binary_masks", base)
    if m is None:
        raise ValueError(f"Could not parse frame range from filename: {filename}")
    start = int(m.group(1))
    end = int(m.group(2))
    return start, end


def make_gif(masks, output_path):
    """Create a GIF from a sequence of binary masks."""
    scale = 0.75
    pil_frames = [
        Image.fromarray((frame.astype(np.uint8) * 255), mode="L").resize(
            (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        )
        for frame in masks
    ]

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=50,
        loop=0,
        optimize=True,
    )

    print(f"Saved GIF to {output_path}")


def process_all_masks_OLD(filenames_, target_date, num_masks):
    """
    Main processing loop - extracts features from all frames in the mask file.

    Parameters:
    filenames_: list of str
        Paths to .npz mask files
    target_date: str
        Date string for split lookup
    num_masks: int
        Number of masks per frame (usually 1 for single-worm videos)

    Returns:
    list: List of metric dictionaries, one per frame
    """
    split_csv = "hand_scored_datasheets/video_splits.csv"
    frame_split_starts, frame_split_ends = get_values_for_date(split_csv, target_date)
    all_metrics = []
    for filename_ in filenames_:
        with np.load(filename_) as data:
            loaded_all_masks = data["masks"].squeeze(axis=1)

            print(f"Loaded mask file with shape: {loaded_all_masks.shape}")
            print(f"Processing {loaded_all_masks.shape[0]} frames...")

            for frame in range(loaded_all_masks.shape[0]):
                if frame % 100 == 0 or frame == loaded_all_masks.shape[0] - 1:
                    progress_pct = (frame + 1) / loaded_all_masks.shape[0] * 100
                    print(
                        f"\r  Processing frame {frame}/{loaded_all_masks.shape[0]} "
                        f"({progress_pct:.1f}%)",
                        end="",
                        flush=True,
                    )

                for mask_index in range(num_masks):
                    mask = loaded_all_masks[frame]
                    metrics = analyze_shape(mask)
                    all_metrics.append(metrics)

            print()

        return all_metrics


def process_final_mask(final_path, num_masks=1):
    """
    Load an existing FINAL mask file and extract per-frame features,
    bypassing the stitching step entirely.

    Parameters
    ----------
    final_path : str
        Path to the *_FINAL_*_binary_masks.npz file.
    num_masks : int
        Number of masks per frame (usually 1 for single-worm videos).

    Returns
    -------
    all_metrics : list of dict
        List of metric dictionaries, one per frame.
    """
    print(f"Loading pre-existing FINAL mask: {final_path}")
    with np.load(final_path) as data:
        masks = data["masks"]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(axis=1)
    print(f"  Mask shape: {masks.shape}")

    all_metrics = []
    T = masks.shape[0]
    print(f"Processing {T} frames...")

    for frame in range(T):
        if frame % 100 == 0 or frame == T - 1:
            progress_pct = (frame + 1) / T * 100
            print(
                f"\r  Processing frame {frame}/{T} ({progress_pct:.1f}%)",
                end="",
                flush=True,
            )
        for mask_index in range(num_masks):
            mask = masks[frame]
            metrics = analyze_shape(mask)
            all_metrics.append(metrics)

    print()
    return all_metrics


def process_all_masks(filenames_, num_masks):
    """
    Stitch multiple mask files into one continuous sequence, using the file
    with range 0..N as the base, and replacing interior intervals with the
    corresponding masks from the other files.

    Parameters
    ----------
    filenames_ : list of str
        Paths to .npz mask files.
    num_masks : int
        Number of masks per frame (usually 1 for single-worm videos).

    Returns
    -------
    all_metrics : list of dict
        List of metric dictionaries, one per frame (per mask_index).
    """
    split_csv = "hand_scored_datasheets/video_splits.csv"
    target_date = str(filenames_[0]).split("_fullvideo")[0].split("Masks/")[1]
    frame_split_starts, frame_split_ends = get_values_for_date(split_csv, target_date)

    # --- Parse all ranges ---
    ranges = []
    for fn in filenames_:
        if "FINAL" in os.path.basename(fn):
            continue
        start, end = parse_frame_range_from_name(fn)
        ranges.append((fn, start, end))

    if len(ranges) > 1:
        allowed = set(zip(frame_split_starts, frame_split_ends))
        allowed.add((0, 1799))
        filtered_ranges = [
            (p, s, e) for (p, s, e) in ranges if (s, e) in allowed
        ]
        ranges = filtered_ranges

    # --- Find base file: the one starting at 0 with the max end ---
    base_candidates = [r for r in ranges if r[1] == 0]
    if not base_candidates:
        raise ValueError("No base file found (no file with start == 0).")

    base_fn, base_start, base_end = max(base_candidates, key=lambda x: x[2])
    total_last_frame = base_end
    print(f"Using base file: {base_fn} with range {base_start}-{base_end}")

    # --- Load base masks ---
    with np.load(base_fn) as data:
        base_masks = data["masks"].squeeze(axis=1)
    print(f"Base masks shape: {base_masks.shape}")

    expected_T = base_end - base_start
    if base_masks.shape[0] != expected_T:
        raise ValueError(
            f"Base masks first dimension {base_masks.shape[0]} != expected {expected_T}"
        )

    stitched_masks = base_masks.copy()

    # --- Overwrite interior intervals from other files ---
    for fn, start, end in ranges:
        if fn == base_fn:
            continue

        if start <= 0 or end > total_last_frame:
            print(f"Skipping edge interval {start}-{end} from {fn}")
            continue

        print(f"Overwriting interval {start}-{end} from {fn}")
        with np.load(fn) as data:
            seg_masks = data["masks"].squeeze(axis=1)

        L = end - start
        if seg_masks.shape[0] != L:
            raise ValueError(
                f"Segment {fn} has {seg_masks.shape[0]} frames, "
                f"but expected {L} for range {start}-{end}"
            )

        stitched_masks[start:end] = seg_masks

    # --- Save stitched masks to a "FINAL" file ---
    base_dir = os.path.dirname(base_fn)
    base_name = os.path.basename(base_fn)

    final_base_name = re.sub(
        r"_frame_split_\d+_\d+_points_binary_masks",
        "_frame_split_FINAL_points_binary_masks",
        base_name,
    )
    final_path = os.path.join(base_dir, final_base_name)

    np.savez_compressed(final_path, masks=stitched_masks)
    print(f"Saved stitched masks to: {final_path}")

    # --- Now run metrics on the stitched sequence ---
    all_metrics = []
    T = stitched_masks.shape[0]
    print(f"Processing stitched sequence of {T} frames...")

    for frame in range(T):
        if frame % 100 == 0 or frame == T - 1:
            progress_pct = (frame + 1) / T * 100
            print(
                f"\r  Processing frame {frame}/{T} ({progress_pct:.1f}%)",
                end="",
                flush=True,
            )

        for mask_index in range(num_masks):
            mask = stitched_masks[frame]
            metrics = analyze_shape(mask)
            all_metrics.append(metrics)

    print()
    return all_metrics


# ============================================================================
# STEP 2.5: Planam PCA functions
# ============================================================================

def referencePCA(
    folder,
    raw_pickle_folder,
    reset_reference,
    ncoord=20,
    scale=True,
    verbose=False,
    tp=1,
    limFrame=False,
    retScaling=False,
    ref_mean=False,
):
    """Define reference PCA space for states and return PCA fit."""
    raw_pickle_folder = Path(raw_pickle_folder)
    folder = Path(folder)

    if retScaling and not scale:
        raise ValueError("retScaling=True requires scale=True in referencePCA.")

    if reset_reference:
        for item in folder.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"Failed to delete {item}: {e}")

    copied_files = []
    for item in raw_pickle_folder.iterdir():
        if item.is_file():
            if "FINAL" not in item.name:
                continue
            dest_file = folder / item.name
            if not dest_file.exists():
                print(f"Copying {item.name} -> {folder}")
                shutil.copy2(item, dest_file)
                copied_files.append(item.name)
            else:
                print(f"Already exists, skipping: {item.name}")

    if len(copied_files) > 0:
        theta = loadAllWorms_theta_only(folder, limFrame=limFrame)
        X, X_nan = defineStates_noLight(theta, tp=tp)
        if ref_mean:
            mean = X.mean(axis=0)
            mean_nan = X_nan.mean(axis=0)
            np.save(folder / "REF_MEAN.npy", mean)
            np.save(folder / "REF_MEAN_NAN.npy", mean_nan)
        if scale:
            X = preprocessing.scale(X, with_std=False)
            X_nan = preprocessing.scale(X_nan, with_std=False)
            SCALING = preprocessing.StandardScaler(with_mean=False).fit(X)
            SCALING_NAN = preprocessing.StandardScaler(with_mean=False).fit(X_nan)
            X = SCALING.transform(X)
            X_nan = SCALING_NAN.transform(X_nan)
            with open(folder / "SCALING.pickle", "wb") as f:
                pickle.dump(SCALING, f)
            with open(folder / "SCALING_NAN.pickle", "wb") as f:
                pickle.dump(SCALING_NAN, f)
        pca = PCA(n_components=ncoord, random_state=0)
        pca.fit(X)
        pca_nan = PCA(n_components=ncoord, random_state=0)
        pca_nan.fit(X_nan)
        with open(folder / "PCA_.pickle", "wb") as f:
            pickle.dump(pca, f)
        with open(folder / "PCA_NAN.pickle", "wb") as f:
            pickle.dump(pca_nan, f)
    else:
        print("Skipping recreation of PCA REF bceause already contains all 'FINAL' files.")
        with open(folder / "PCA_.pickle", "rb") as f:
            pca = pickle.load(f)
        if retScaling:
            with open(folder / "SCALING.pickle", "rb") as f:
                SCALING = pickle.load(f)
        if ref_mean:
            mean = np.load(folder / "REF_MEAN.npy")
        print("Loaded the pca, retscaling and retmean transformations")

    if ref_mean:
        if verbose:
            eig_val, eig_vector = np.linalg.eig(np.cov(X.T))
            eig_val = pca.explained_variance_
            eig_vector = pca.components_
            if retScaling:
                return pca, pca.explained_variance_ratio_, eig_val, eig_vector, SCALING, mean
            else:
                return pca, pca.explained_variance_ratio_, eig_val, eig_vector, mean
        if retScaling:
            return pca, SCALING, mean
        return pca, mean
    else:
        if verbose:
            eig_val, eig_vector = np.linalg.eig(np.cov(X.T))
            eig_val = pca.explained_variance_
            eig_vector = pca.components_
            if retScaling:
                return pca, pca.explained_variance_ratio_, eig_val, eig_vector, SCALING
            else:
                return pca, pca.explained_variance_ratio_, eig_val, eig_vector
        if retScaling:
            return pca, SCALING
        return pca


def saveThetaResults(save_filepath, thetas, com):
    theta_np = np.asarray(thetas)
    if theta_np.ndim == 1 and len(theta_np) == 179900:
        theta_np = theta_np.reshape(1799, 100)
    Results = (theta_np, com)
    with open(str(save_filepath), "wb") as f:
        pickle.dump(Results, f)


def extract_key(f, pattern):
    match = re.search(pattern, f)
    if match:
        trial_num = int(match.group(1))
        region_nums = tuple(map(int, match.group(2).split("_")))
        return (trial_num,) + region_nums
    return (float("inf"),)


def loadAllWorms_theta_only(folder, interpolate=False, limFrame=False):
    """Load and concatenate only the shape states (theta) from all worm pickles."""
    pattern = r".*trial_(\d+).*regions_([\d_]+)_fullvideo.*\.pickle"

    theta = []
    if not limFrame:
        limFrame = -1
    sorted_files = sorted(os.listdir(folder), key=lambda f: extract_key(f, pattern))
    for i, filename in enumerate(sorted_files):
        if filename in (
            ".DS_Store",
            "PCA_NAN.pickle",
            "SCALING.pickle",
            "PCA_.pickle",
            "SCALING_NAN.pickle",
        ):
            continue
        R = loadResults(str(folder / filename))
        if len(R[0]) != 1799:
            continue
        theta.extend(R[0])

    return theta


def defineStates_noLight(theta, tp=1, lowpass_filter=False, smooth=0, ref_shape=None):
    """
    Build stacked state matrix P from per-frame shape vectors theta,
    without any light metadata.

    Returns
    -------
    P : 2D array, NaN-rows removed, L1-normalized
    P_nan : 2D array, all rows kept, NaN-safe L1-normalized
    """
    Theta = np.zeros((len(theta), len(theta[0])))

    for i, val in enumerate(theta):
        if lowpass_filter:
            val = lowpass(val, lowpass_filter)
        Theta[i, :] = val

    P = np.zeros((len(theta) - tp, len(theta[0]) * tp))
    WN = np.zeros(P.shape[0])

    for i in range(P.shape[0]):
        P[i, :] = np.reshape(Theta[i : i + tp, :], P.shape[1])

    flat = np.sum(P, axis=1)
    ind_real = np.isfinite(flat)

    P_nan = P.copy()
    P = P[ind_real, :]

    if np.all(np.isnan(flat)):
        P_norm = P
        P_nan_norm = P_nan
    else:
        P_norm = preprocessing.normalize(P, norm="l1", axis=1)
        P_no_nans = np.nan_to_num(P_nan, nan=0.0)
        P_nan_norm = preprocessing.normalize(P_no_nans, norm="l1", axis=1)

    return P_norm, P_nan_norm


def lowpass(sig, freq):
    fft_filt = np.ones(sig.size)
    fft_filt[freq:-freq] = 0
    return np.fft.ifft(np.fft.fft(sig * fft_filt))


def loadResults(name):
    with open(name, "rb") as input_file:
        e = pickle.load(input_file)
    return e


def performPCA(
    folder, reference, the_video, LABEL_TYPE, scale=True, verbose=False,
    ncoord=20, tp=1, smooth=0,
):
    """Load PCA reference and transform a single video's theta into PC space."""
    with open(folder + "PCA_.pickle", "rb") as f:
        R = pickle.load(f)

    X_all = np.array([])
    X_nan_all = np.array([])
    all_masks = []

    worm = loadResults(folder + the_video + ".pickle")
    theta = worm[0]
    filename_masks = the_video + "_binary_masks.npz"
    with np.load("data/Masks/" + filename_masks) as data:
        masks = data["masks"]
    all_masks.append(masks)
    P, P_nan = defineStates_noLight(theta, tp=tp, smooth=smooth)

    if len(P) == 0:
        print("no Data for worm: ", the_video)
        return None, None

    X = P.copy()
    X_nan = P_nan.copy()
    if scale:
        X = preprocessing.scale(X)
        X_nan = preprocessing.scale(X_nan)

    X = R.transform(X)
    X_nan = R.transform(X_nan)

    if X_all.size == 0:
        X_all = X.copy()
        X_nan_all = X_nan.copy()
    else:
        X_all = np.append(X_all, X.copy(), 0)
        X_nan_all = np.append(X_nan_all, X_nan.copy(), 0)

    X_pc1 = X_nan[:, 0]
    X_pc2 = X_nan[:, 1]

    if verbose:
        return X_all, X_nan_all, expVar, eig_val, eig_vector, all_masks

    return X_pc1, X_pc2


# ============================================================================
# STEP 3: Interpolation functions
# ============================================================================

def interpolate_angles_with_wrapping(angles, max_gap=30):
    """
    Interpolate NaN values in angle array with wrapping awareness (0 deg = 180 deg).

    Parameters:
    -----------
    angles : numpy array
        Array of angles in degrees (0-180 range)
    max_gap : int
        Maximum number of consecutive NaNs to interpolate

    Returns:
    --------
    numpy array : Interpolated angles
    """
    angles = angles.copy()
    n = len(angles)

    valid_indices = np.where(~np.isnan(angles))[0]
    if len(valid_indices) == 0:
        return angles

    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]

    if first_valid_idx > 0:
        angles[:first_valid_idx] = angles[first_valid_idx]

    if last_valid_idx < n - 1:
        angles[last_valid_idx + 1 :] = angles[last_valid_idx]

    i = 0
    while i < n:
        if np.isnan(angles[i]):
            gap_start = i
            gap_end = i
            while gap_end < n and np.isnan(angles[gap_end]):
                gap_end += 1

            gap_length = gap_end - gap_start

            if gap_length <= max_gap and gap_start > 0 and gap_end < n:
                angle_before = angles[gap_start - 1]
                angle_after = angles[gap_end]

                diff = angle_after - angle_before

                if abs(diff) > 90:
                    if diff > 90:
                        angle_after_unwrapped = angle_after - 180
                    else:
                        angle_after_unwrapped = angle_after + 180

                    for j in range(gap_start, gap_end):
                        weight = (j - gap_start + 1) / (gap_length + 1)
                        interpolated = angle_before + weight * (
                            angle_after_unwrapped - angle_before
                        )
                        if interpolated < 0:
                            interpolated += 180
                        elif interpolated >= 180:
                            interpolated -= 180
                        angles[j] = interpolated
                else:
                    for j in range(gap_start, gap_end):
                        weight = (j - gap_start + 1) / (gap_length + 1)
                        angles[j] = angle_before + weight * (
                            angle_after - angle_before
                        )

            i = gap_end
        else:
            i += 1

    return angles


def interpolate_linear(values, max_gap=30):
    """
    Interpolate NaN values using linear interpolation.

    Parameters:
    -----------
    values : numpy array
        Array with NaN values to interpolate
    max_gap : int
        Maximum number of consecutive NaNs to interpolate

    Returns:
    --------
    numpy array : Interpolated values
    """
    values = values.copy()
    n = len(values)

    valid_indices = np.where(~np.isnan(values))[0]
    if len(valid_indices) == 0:
        return values

    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]

    if first_valid_idx > 0:
        values[:first_valid_idx] = values[first_valid_idx]

    if last_valid_idx < n - 1:
        values[last_valid_idx + 1 :] = values[last_valid_idx]

    i = 0
    while i < n:
        if np.isnan(values[i]):
            gap_start = i
            gap_end = i
            while gap_end < n and np.isnan(values[gap_end]):
                gap_end += 1

            gap_length = gap_end - gap_start

            if gap_length <= max_gap and gap_start > 0 and gap_end < n:
                value_before = values[gap_start - 1]
                value_after = values[gap_end]

                for j in range(gap_start, gap_end):
                    weight = (j - gap_start + 1) / (gap_length + 1)
                    values[j] = value_before + weight * (value_after - value_before)

            i = gap_end
        else:
            i += 1

    return values


def interpolate_all_features(feature_dict, max_gap=30):
    """
    Apply appropriate interpolation to all features in the dictionary.

    Returns:
    --------
    dict : Dictionary with interpolated features
    dict : Dictionary with before/after NaN counts
    """
    interpolated = {}
    nan_report = {}

    for feature_name, feature_array in feature_dict.items():
        nans_before = np.sum(np.isnan(feature_array))
        if feature_name == "Angles":
            interpolated[feature_name] = interpolate_angles_with_wrapping(
                feature_array, max_gap
            )
        if feature_name == "PC1":
            interpolated[feature_name] = feature_array
        if feature_name == "PC2":
            interpolated[feature_name] = feature_array
        else:
            interpolated[feature_name] = interpolate_linear(feature_array, max_gap)

        nans_after = np.sum(np.isnan(interpolated[feature_name]))
        nans_filled = nans_before - nans_after

        nan_report[feature_name] = {
            "before": nans_before,
            "after": nans_after,
            "filled": nans_filled,
            "total": len(feature_array),
        }

    return interpolated, nan_report


def print_nan_report(nan_report, video_name):
    """Print a formatted NaN report for a video."""
    print(f"\n{'=' * 70}")
    print(f"NaN INTERPOLATION REPORT: {video_name}")
    print(f"{'=' * 70}")

    print(f"\n{'Feature':<25} {'Before':<15} {'After':<15} {'Filled':<15}")
    print(f"{'-' * 70}")

    for feature_name, stats in nan_report.items():
        before_pct = 100 * stats["before"] / stats["total"]
        after_pct = 100 * stats["after"] / stats["total"]

        before_str = f"{stats['before']} ({before_pct:.1f}%)"
        after_str = f"{stats['after']} ({after_pct:.1f}%)"
        filled_str = f"{stats['filled']}"

        print(f"{feature_name:<25} {before_str:<15} {after_str:<15} {filled_str:<15}")

    total_before = sum(s["before"] for s in nan_report.values())
    total_after = sum(s["after"] for s in nan_report.values())
    total_filled = total_before - total_after

    print(f"{'-' * 70}")
    print(f"{'TOTAL':<25} {total_before:<15} {total_after:<15} {total_filled:<15}")
    print()


# ============================================================================
# STEP 3.5: NaN export functions
# ============================================================================

def extract_region_from_video_name(video_name):
    """
    Extract region coordinates from video name using string splitting.
    Example: '2025_10_14_10_47_52_trial_1_TP_regions_100_200_300_400_fullvideo'
    Returns: '100_200_300_400' (or None if not found)
    """
    try:
        after_regions = video_name.split("_regions_")[1]
        region = after_regions.split("_fullvideo")[0]
        return region
    except (IndexError, AttributeError):
        return None


def update_nan_csv(session_name, video_name, features_dict, csv_dir):
    """
    Update the NaN tracking CSV with NaN counts per feature for a video.

    Args:
        session_name: Session identifier
        video_name: Video name (full name from results)
        features_dict: Dictionary of features where each value is a numpy array
        csv_dir: Directory to store the CSV
    """
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "nan_counts.csv")

    region = extract_region_from_video_name(video_name)
    if region is None:
        print(f"Warning: Could not extract region from {video_name}")
        return

    nan_counts = {}
    for feature_name, feature_array in features_dict.items():
        if isinstance(feature_array, np.ndarray):
            nan_count = int(np.sum(np.isnan(feature_array)))
            nan_counts[feature_name] = nan_count

    if not region:
        print("Can't make csv cause region not made.")
    row_data = {"Session": session_name, "Region": region, **nan_counts}

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        mask = (existing_df["Session"] == session_name) & (
            existing_df["Region"] == region
        )
        existing_df = existing_df[~mask]
        new_row_df = pd.DataFrame([row_data])
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        updated_df = pd.DataFrame([row_data])

    column_order = ["Session", "Region"] + sorted(
        [col for col in updated_df.columns if col not in ["Session", "Region"]]
    )
    updated_df = updated_df[column_order]
    updated_df.to_csv(csv_path, index=False)

    print(f"\nNaN CSV updated: {session_name} | Region {region}")


# ============================================================================
# STEP 4: Feature aggregation with filtering
# ============================================================================

def plot_feature_histograms(
    feature_data_dict, title_suffix="", normal_filter_threshold=2.5
):
    """Plot histograms for all features with filtering boundaries marked."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    axes = axes.flatten()

    feature_names = [
        "Areas", "Area_percentages", "Perimeters", "Area_perimeter_ratios",
        "Circularities", "Hull_areas", "Centroidxs", "Centroidys",
        "Angles", "Concavities", "PC1", "PC2",
    ]

    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        feature_array = feature_data_dict[feature_name]

        clean_data = feature_array[~np.isnan(feature_array)]
        ax.hist(clean_data, bins=30, alpha=0.7, edgecolor="black")

        mean_val = np.mean(clean_data)
        std_val = np.std(clean_data)

        for sigma in [1, 2, 3, 4, 5, 6]:
            upper_sigma = mean_val + sigma * std_val
            lower_sigma = mean_val - sigma * std_val
            ax.axvline(upper_sigma, color="gray", linestyle="-", alpha=0.5, linewidth=1)
            ax.axvline(lower_sigma, color="gray", linestyle="-", alpha=0.5, linewidth=1)
            if sigma == 1:
                ax.text(
                    upper_sigma, ax.get_ylim()[1] * 0.05, f"+{sigma}\u03c3",
                    rotation=90, fontsize=8, color="gray", ha="right", va="bottom",
                )
                ax.text(
                    lower_sigma, ax.get_ylim()[1] * 0.05, f"-{sigma}\u03c3",
                    rotation=90, fontsize=8, color="gray", ha="left", va="bottom",
                )

        ax.axvline(
            mean_val, color="black", linestyle="-", alpha=0.7, linewidth=1.5,
            label="Mean",
        )

        upper_bound = mean_val + normal_filter_threshold * std_val
        lower_bound = mean_val - normal_filter_threshold * std_val
        ax.axvline(
            upper_bound, color="red", linestyle="--", alpha=0.8, linewidth=2,
            label=f"{normal_filter_threshold}\u03c3 filter bounds",
        )
        ax.axvline(lower_bound, color="red", linestyle="--", alpha=0.8, linewidth=2)

        ax.set_title(f"{feature_name}", fontsize=18, pad=15)
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)

        if i == 0:
            legend_elements = [
                Line2D([0], [0], color="black", linewidth=1.5, label="Mean"),
                Line2D(
                    [0], [0], color="gray", linewidth=1, alpha=0.5,
                    label="1\u03c3, 2\u03c3, 3\u03c3",
                ),
                Line2D(
                    [0], [0], color="red", linestyle="--", linewidth=2,
                    label=f"{normal_filter_threshold}\u03c3 filter bounds",
                ),
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    for i in range(len(feature_names), len(axes)):
        axes[i].set_visible(False)

    plt.subplots_adjust(
        left=0.06, right=0.94, top=0.96, bottom=0.04, wspace=0.35, hspace=0.5,
    )

    fig.suptitle(f"Feature Histograms{title_suffix}", fontsize=16, y=0.98)
    plt.show()


def find_prev_near_level(t, A, spike_idxs, tol=50.0):
    """
    For each spike index s, take the post-spike area A[s_post] as the target,
    look backwards for the nearest earlier time where |A - target| <= tol.
    """
    t = np.asarray(t)
    A = np.asarray(A)
    spike_idxs = np.asarray(spike_idxs, dtype=int)

    N = len(A)
    prev_times = []
    prev_areas = []

    for s in spike_idxs:
        s_post = min(s + 1, N - 1)
        target = A[s_post]

        A_seg = A[:s_post]
        t_seg = t[:s_post]

        close_mask = np.abs(A_seg - target) <= tol
        close_idxs = np.where(close_mask)[0]

        if len(close_idxs) == 0:
            prev_times.append(np.nan)
            prev_areas.append(np.nan)
            continue

        i = close_idxs[-1]
        prev_times.append(t_seg[i])
        prev_areas.append(A_seg[i])

    return np.array(prev_times), np.array(prev_areas)


def compute_nan_intervals(t, prev_times, pop_indices, max_frames=50):
    """
    For each event, find the index corresponding to prev_times[k],
    compare with pop_indices[k], and produce NaN intervals.
    """
    t = np.asarray(t)
    prev_times = np.asarray(prev_times)
    pop_indices = np.asarray(pop_indices, dtype=int)

    N = len(t)
    intervals = []

    for tau, pop_idx in zip(prev_times, pop_indices):
        if np.isnan(tau):
            continue
        if pop_idx < 0 or pop_idx >= N:
            continue

        idx0 = np.searchsorted(t, tau, side="right") - 1
        idx0 = np.clip(idx0, 0, N - 1)

        idx1 = pop_idx

        if idx0 > idx1:
            idx0, idx1 = idx1, idx0

        span = idx1 - idx0

        if span <= max_frames:
            intervals.append((idx0, idx1))
        else:
            intervals.append((idx1, idx1))

    return intervals


def get_nans_from_areas(areas):
    """Detect occlusion/outlier indices and pop-back interpolation intervals."""
    areas = np.asarray(areas, dtype=float)
    areas = np.array(areas, dtype=float).flatten()

    threshold_elim = 2.5
    ub = np.nanmean(areas) + threshold_elim * np.nanstd(areas)
    lb = np.nanmean(areas) - threshold_elim * np.nanstd(areas)
    is_outlier = ~((areas >= lb) & (areas <= ub))
    is_outlier = np.where(is_outlier)[0]

    smooth = median_filter(areas, size=5)
    baseline = median_filter(smooth, size=60)
    ratio = smooth / baseline
    is_nan = np.isnan(areas)
    min_ratio = 0.55
    min_abs_drop = 0.15 * np.nanmedian(areas)
    occluded = is_nan | ((ratio < min_ratio) & ((baseline - smooth) > min_abs_drop))
    occlusion_indices = np.where(occluded)[0]

    occlusion_indices_ = np.concatenate([occlusion_indices, is_outlier])
    print("Occlusion indices:", occlusion_indices_)

    t = np.arange(len(areas))
    dA_dt = np.gradient(areas, t)
    d2A_dt2 = np.gradient(dA_dt, t)
    threshold = 8 * np.nanstd(d2A_dt2)
    spike_idxs = np.where(d2A_dt2 > threshold)[0]

    nan_to_real_idxs = (
        np.where(np.isnan(areas[:-1]) & ~np.isnan(areas[1:]))[0] + 1
    )

    spike_idxs = np.unique(np.concatenate([spike_idxs, nan_to_real_idxs]))

    prev_times, prev_areas = find_prev_near_level(t, areas, spike_idxs)
    pop_indices = np.minimum(spike_idxs + 1, len(areas) - 1)
    interpolation_intervals = compute_nan_intervals(
        t, prev_times, pop_indices, max_frames=150
    )

    return occlusion_indices_, interpolation_intervals


def interpolate_intervals(feature_np, intervals):
    """
    Linearly interpolate inside each interval [i0, i1] (inclusive),
    using the values at i0-1 and i1+1 as anchors.
    """
    feature_np = np.asarray(feature_np, dtype=float).reshape(-1)
    n = feature_np.shape[0]

    for i0, i1 in intervals:
        if i0 < 0 or i1 >= n or i0 > i1:
            continue

        left_idx = i0 - 1
        right_idx = i1 + 1

        if left_idx < 0 or right_idx >= n:
            continue

        y_left = feature_np[left_idx]
        y_right = feature_np[right_idx]

        if np.isnan(y_left) or np.isnan(y_right):
            continue

        xs = np.arange(i0, i1 + 1)
        feature_np[xs] = np.interp(xs, [left_idx, right_idx], [y_left, y_right])

    return feature_np


def aggregate_filtered_features(results):
    """Aggregate features from all frames and apply statistical outlier filtering."""
    print(f"=== FUNCTION CALLED - Results length: {len(results)} ===")
    print(
        f"First few areas from results: "
        f"{[results[i]['area'] for i in range(min(5, len(results)))]}"
    )

    areas = []
    area_percentages = []
    perimeters = []
    thetas = []
    area_perimeter_ratios = []
    circularities = []
    hull_areas = []
    centroidxs = []
    centroidys = []
    angles = []

    for i, mask_result in enumerate(results):
        areas.append(results[i]["area"])
        area_percentages.append(results[i]["area_percentage"])
        perimeters.append(results[i]["perimeter"])
        thetas.append(results[i]["theta"])
        area_perimeter_ratios.append(results[i]["area_perimeter_ratio"])
        circularities.append(results[i]["circularity"])
        hull_areas.append(results[i]["hull_area"])
        centroidxs.append(results[i]["centroid_x"])
        centroidys.append(results[i]["centroid_y"])
        angles.append(results[i]["orientation_angle"])

    all_features = {
        "areas": areas,
        "area_percentages": area_percentages,
        "perimeters": perimeters,
        "thetas": thetas,
        "area_perimeter_ratios": area_perimeter_ratios,
        "circularities": circularities,
        "hull_areas": hull_areas,
        "centroidxs": centroidxs,
        "centroidys": centroidys,
        "angles": angles,
    }

    unfiltered_features = all_features

    nans_to_insert, intervals_to_interpolate = get_nans_from_areas(areas)

    filtered_features = {}

    for feature_idx, (feature_name, feature_list) in enumerate(all_features.items()):
        feature_np = np.array(feature_list)

        if nans_to_insert is not None:
            idxs_to_nan = np.asarray(nans_to_insert, dtype=int)
            idxs_to_nan = idxs_to_nan[
                (idxs_to_nan >= 0) & (idxs_to_nan < feature_np.shape[0])
            ]
            idxs_to_nan = np.unique(idxs_to_nan)
            feature_np[idxs_to_nan] = np.nan

        if feature_name != "thetas" and intervals_to_interpolate is not None:
            feature_np = interpolate_intervals(feature_np, intervals_to_interpolate)

        filtered_features[feature_name] = feature_np
        print(
            f"NaNs after filtering/interpolating: "
            f"\t\t{np.sum(np.isnan(feature_np))}\t\t{feature_name}"
        )

    Areas = filtered_features["areas"]
    Area_percentages = filtered_features["area_percentages"]
    Perimeters = filtered_features["perimeters"]
    Thetas = filtered_features["thetas"]
    Area_perimeter_ratios = filtered_features["area_perimeter_ratios"]
    Circularities = filtered_features["circularities"]
    Hull_areas = filtered_features["hull_areas"]
    Centroidxs = filtered_features["centroidxs"]
    Centroidys = filtered_features["centroidys"]
    Angles = filtered_features["angles"]

    Concavities = []
    if len(Areas) != len(Hull_areas):
        print("ERROR: Length of Areas and Hull_areas not equal.")
    else:
        for i in range(len(Areas)):
            Concavities.append((Hull_areas[i] - Areas[i]) / Hull_areas[i])

    Concavities = np.array(Concavities)

    return (
        Areas, Area_percentages, Perimeters, Thetas,
        Area_perimeter_ratios, Circularities, Hull_areas,
        Centroidxs, Centroidys, Angles, Concavities, unfiltered_features,
    )


# ============================================================================
# STEP 5: CS-on vs CS-off plotting
# ============================================================================

def plot_CSon_vs_CSoff_features(
    Areas, Perimeters, Circularities, trial_definitions_df, session_prefix
):
    """Plot features comparing CS-on (during trials) vs CS-off (between trials) periods."""
    if trial_definitions_df is None:
        print("Trial definitions not available - skipping CS-on/CS-off comparison plots")
        return

    print("\nCreating CS-on vs CS-off comparison plots...")

    CSon_areas = []
    CSon_perimeters = []
    CSon_circularities = []

    for idx, row in trial_definitions_df.iterrows():
        trial_num = row["trial_num"]
        start_frame = row["start_frame"]
        end_frame = row["end_frame"]

        trial_areas = Areas[start_frame : end_frame + 1]
        trial_perimeters = Perimeters[start_frame : end_frame + 1]
        trial_circularities = Circularities[start_frame : end_frame + 1]

        if np.sum(np.isnan(trial_areas)) > len(trial_areas) * 0.5:
            print(f"  Skipping trial {trial_num} (CS-on) - too many NaNs")
            continue

        CSon_areas.append(trial_areas)
        CSon_perimeters.append(trial_perimeters)
        CSon_circularities.append(trial_circularities)

    CSoff_areas = []
    CSoff_perimeters = []
    CSoff_circularities = []

    for idx in range(len(trial_definitions_df) - 1):
        current_trial = trial_definitions_df.iloc[idx]
        next_trial = trial_definitions_df.iloc[idx + 1]

        interval_start = current_trial["end_frame"] + 1
        interval_end = next_trial["start_frame"] - 1

        if interval_end <= interval_start:
            print(
                f"  Skipping interval after trial {current_trial['trial_num']} - no frames"
            )
            continue

        interval_areas = Areas[interval_start : interval_end + 1]
        interval_perimeters = Perimeters[interval_start : interval_end + 1]
        interval_circularities = Circularities[interval_start : interval_end + 1]

        if np.sum(np.isnan(interval_areas)) > len(interval_areas) * 0.5:
            print(
                f"  Skipping interval after trial {current_trial['trial_num']} "
                f"(CS-off) - too many NaNs"
            )
            continue

        CSoff_areas.append(interval_areas)
        CSoff_perimeters.append(interval_perimeters)
        CSoff_circularities.append(interval_circularities)

    def plot_comparison(CSon_data, CSoff_data, feature_name, normalize=False):
        """Plot CS-on vs CS-off with color coding."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        if normalize:
            CSon_normalized = []
            for trace in CSon_data:
                min_val = np.nanmin(trace)
                max_val = np.nanmax(trace)
                if max_val > min_val:
                    normalized = (trace - min_val) / (max_val - min_val)
                    CSon_normalized.append(normalized)
            CSon_data = CSon_normalized

            CSoff_normalized = []
            for trace in CSoff_data:
                min_val = np.nanmin(trace)
                max_val = np.nanmax(trace)
                if max_val > min_val:
                    normalized = (trace - min_val) / (max_val - min_val)
                    CSoff_normalized.append(normalized)
            CSoff_data = CSoff_normalized

        for i, trace in enumerate(CSon_data):
            x_vals = np.arange(len(trace))
            ax.plot(x_vals, trace, color="cyan", alpha=0.3, linewidth=1)

        if CSon_data:
            max_len = max(len(trace) for trace in CSon_data)
            padded_CSon = []
            for trace in CSon_data:
                padded = np.full(max_len, np.nan)
                padded[: len(trace)] = trace
                padded_CSon.append(padded)
            CSon_mean = np.nanmean(padded_CSon, axis=0)
            x_vals = np.arange(len(CSon_mean))
            ax.plot(
                x_vals, CSon_mean, color="cyan", linewidth=3,
                label="CS-on (trials) - Mean",
            )

        for i, trace in enumerate(CSoff_data):
            x_vals = np.arange(len(trace))
            ax.plot(x_vals, trace, color="orange", alpha=0.3, linewidth=1)

        if CSoff_data:
            max_len = max(len(trace) for trace in CSoff_data)
            padded_CSoff = []
            for trace in CSoff_data:
                padded = np.full(max_len, np.nan)
                padded[: len(trace)] = trace
                padded_CSoff.append(padded)
            CSoff_mean = np.nanmean(padded_CSoff, axis=0)
            x_vals = np.arange(len(CSoff_mean))
            ax.plot(
                x_vals, CSoff_mean, color="orange", linewidth=3,
                label="CS-off (inter-trial) - Mean",
            )

        title_suffix_norm = " (Normalized)" if normalize else ""
        ax.set_title(
            f"{feature_name}: CS-on vs CS-off Comparison{title_suffix_norm}",
            fontsize=14,
        )
        ax.set_xlabel("Frame (relative to period start)", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print("\nPlotting raw value comparisons...")
    plot_comparison(CSon_areas, CSoff_areas, "Areas", normalize=False)
    plot_comparison(CSon_perimeters, CSoff_perimeters, "Perimeters", normalize=False)
    plot_comparison(
        CSon_circularities, CSoff_circularities, "Circularities", normalize=False
    )

    print("\nPlotting normalized comparisons...")
    plot_comparison(CSon_areas, CSoff_areas, "Areas", normalize=True)
    plot_comparison(CSon_perimeters, CSoff_perimeters, "Perimeters", normalize=True)
    plot_comparison(
        CSon_circularities, CSoff_circularities, "Circularities", normalize=True
    )

    print("\n=== CS-on vs CS-off Summary ===")
    print(f"CS-on periods: {len(CSon_areas)}")
    print(f"CS-off periods: {len(CSoff_areas)}")

    CSon_area_mean = np.nanmean([np.nanmean(trace) for trace in CSon_areas])
    CSoff_area_mean = np.nanmean([np.nanmean(trace) for trace in CSoff_areas])
    print(f"\nAreas:")
    print(f"  CS-on mean: {CSon_area_mean:.2f} px")
    print(f"  CS-off mean: {CSoff_area_mean:.2f} px")
    print(
        f"  Difference: {CSon_area_mean - CSoff_area_mean:.2f} px "
        f"({100 * (CSon_area_mean - CSoff_area_mean) / CSoff_area_mean:.1f}%)"
    )

    CSon_perim_mean = np.nanmean([np.nanmean(trace) for trace in CSon_perimeters])
    CSoff_perim_mean = np.nanmean([np.nanmean(trace) for trace in CSoff_perimeters])
    print(f"\nPerimeters:")
    print(f"  CS-on mean: {CSon_perim_mean:.2f} px")
    print(f"  CS-off mean: {CSoff_perim_mean:.2f} px")
    print(
        f"  Difference: {CSon_perim_mean - CSoff_perim_mean:.2f} px "
        f"({100 * (CSon_perim_mean - CSoff_perim_mean) / CSoff_perim_mean:.1f}%)"
    )

    CSon_circ_mean = np.nanmean([np.nanmean(trace) for trace in CSon_circularities])
    CSoff_circ_mean = np.nanmean([np.nanmean(trace) for trace in CSoff_circularities])
    print(f"\nCircularities:")
    print(f"  CS-on mean: {CSon_circ_mean:.4f}")
    print(f"  CS-off mean: {CSoff_circ_mean:.4f}")
    print(
        f"  Difference: {CSon_circ_mean - CSoff_circ_mean:.4f} "
        f"({100 * (CSon_circ_mean - CSoff_circ_mean) / CSoff_circ_mean:.1f}%)"
    )

    print("\nCS-on vs CS-off comparison complete.")


# ============================================================================
# STEP 7: Master control function
# ============================================================================

def run_feature_extraction_pipeline(
    THETA_PICKLE_DIR,
    PCA_REF_DIR,
    LABEL_TYPE,
    reset_reference,
    video_list,
    feature_folder=None,
    plot_individual=True,
    plot_combined=True,
    plot_trials=True,
    assume_final=False,
):
    """
    Master function to run entire feature extraction pipeline.

    Parameters:
    -----------
    THETA_PICKLE_DIR : str
        Directory for theta pickle output.
    PCA_REF_DIR : str
        Directory for PCA reference space.
    LABEL_TYPE : str
        'points' or 'boxes'.
    reset_reference : bool
        Whether to reset the PCA reference.
    video_list : list of str
        List of video names to process (without .npz extension).
    feature_folder : str, optional
        Where to save .npy files (default: HOLYLABS/Features).
    plot_individual : bool
        Whether to plot histograms for each video.
    plot_combined : bool
        Whether to plot combined histogram across all videos.
    plot_trials : bool
        Whether to create trial-aligned plots.
    assume_final : bool
        If True, skip stitching and load the existing *_FINAL_* mask file
        directly. Raises an error if no FINAL file is found for a video.

    Returns:
    --------
    dict: Dictionary with processed feature data for each video.
    """
    start_time = time.time()

    if feature_folder is None:
        feature_folder = os.path.join("data", "Features")
    os.makedirs(feature_folder, exist_ok=True)

    print(f"{'=' * 30}")
    print("FEATURE EXTRACTION PIPELINE")
    print(f"{'=' * 30}")
    print(f"Videos to process: {len(video_list)}")
    print(f"Output folder: {feature_folder}")
    if assume_final:
        print("Mode: ASSUME_FINAL (skipping stitching, loading existing FINAL masks)")
    print(f"{'=' * 30}\n")

    all_video_features = {}
    processed_data = {}

    if assume_final:
        video_list_ = list(video_list)
    else:
        video_list_ = []
        for v in video_list:
            if check_date_in_splits(
                "hand_scored_datasheets/video_splits.csv", v
            ):
                video_list_.append(v)

    if not video_list_:
        print("WARNING: No videos remain after filtering. Nothing to process.")
        return {}

    for video_idx, current_video in enumerate(reversed(video_list_)):
        print(f"\n{'=' * 30}")
        print(f"[{video_idx + 1}/{len(video_list)}] Processing: {current_video}")
        print(f"{'=' * 30}")

        try:
            curr_vid_pre = current_video.split("_fullvideo")[0]
            mask_files = list(
                Path(MASKS_DIR).glob(f"{curr_vid_pre}*_binary_masks*.npz")
            )

            if not mask_files:
                print(f"ERROR: No base mask file found for {current_video}")
                continue

            filename = str(sorted(mask_files)[-1])

            if "_regions_" in current_video:
                session_prefix = current_video.split("_regions_")[0]
            else:
                session_prefix = None

            trial_definitions_df = None
            if session_prefix:
                trial_csv_path = os.path.join(
                    "data", "Raw_data", session_prefix, "trial_definitions.csv"
                )
                if os.path.exists(trial_csv_path):
                    trial_definitions_df = pd.read_csv(trial_csv_path)
                    print(
                        f"  Loaded trial definitions: {len(trial_definitions_df)} trials"
                    )

            print("\nExtracting features from all frames...")
            if assume_final:
                final_files = [
                    f for f in mask_files if "FINAL" in os.path.basename(str(f))
                ]
                if not final_files:
                    print(
                        f"ERROR: assume_final=True but no FINAL mask file "
                        f"found for {current_video}. Skipping."
                    )
                    continue
                results = process_final_mask(str(final_files[0]), num_masks=1)
            else:
                results = process_all_masks(mask_files, num_masks=1)
            print(f"  Extracted features from {len(results)} frames")

            print("\nApplying outlier filtering...")
            (
                Areas, Area_percentages, Perimeters, Thetas,
                Area_perimeter_ratios, Circularities, Hull_areas,
                Centroidxs, Centroidys, Angles, Concavities, unfiltered_features,
            ) = aggregate_filtered_features(results)

            Centroid_Coords = list(zip(Centroidxs, Centroidys))

            nan_csv_dir = os.path.join("data", "Features", "NaN_analysis")
            session_for_csv = current_video.split("_fullvideo")[0]

            update_nan_csv(
                session_name=session_for_csv,
                video_name=current_video,
                features_dict={
                    "Areas": Areas,
                    "Area_percentages": Area_percentages,
                    "Perimeters": Perimeters,
                    "Area_perimeter_ratios": Area_perimeter_ratios,
                    "Circularities": Circularities,
                    "Hull_areas": Hull_areas,
                    "Centroidxs": Centroidxs,
                    "Centroidys": Centroidys,
                    "Angles": Angles,
                    "Concavities": Concavities,
                },
                csv_dir=nan_csv_dir,
            )

            video_name = filename.split("Masks/")[-1].split("_binary")[0]
            PICKLE_NAME = f"{THETA_PICKLE_DIR}/{video_name}.pickle"
            saveThetaResults(PICKLE_NAME, Thetas, Centroid_Coords)
            print(f"Curent video_name:\t{video_name}")
            print(f"\nPLANAM WHEEL: Thetas and Centroids saved to:\t{PICKLE_NAME}")

            ref_exists = referencePCA(
                PCA_REF_DIR, THETA_PICKLE_DIR, reset_reference
            )

            if ref_exists:
                print("PERFORMING PCA ON " + video_name)
                PC1, PC2 = performPCA(
                    PCA_REF_DIR, THETA_PICKLE_DIR, video_name, LABEL_TYPE
                )
            else:
                print(
                    "Can't include PC1 and PC2 because a PCA space doesn't exist yet"
                )
                PC1, PC2 = None, None

            video_features = {
                "Areas": Areas,
                "Area_percentages": Area_percentages,
                "Perimeters": Perimeters,
                "Area_perimeter_ratios": Area_perimeter_ratios,
                "Circularities": Circularities,
                "Hull_areas": Hull_areas,
                "Centroidxs": Centroidxs,
                "Centroidys": Centroidys,
                "Angles": Angles,
                "Concavities": Concavities,
                "PC1": PC1,
                "PC2": PC2,
            }

            if plot_individual:
                print("\nPlotting feature distributions...")
                plot_feature_histograms(
                    video_features,
                    f" - {current_video}",
                    normal_filter_threshold=2.5,
                )

            if video_idx == 0:
                for feature_name, feature_array in video_features.items():
                    all_video_features[feature_name] = feature_array.copy()
            else:
                for feature_name, feature_array in video_features.items():
                    all_video_features[feature_name] = np.concatenate(
                        [all_video_features[feature_name], feature_array]
                    )

            if plot_trials and trial_definitions_df is not None:
                print("\nCreating CS-on vs CS-off comparison plots...")
                plot_CSon_vs_CSoff_features(
                    Areas, Perimeters, Circularities,
                    trial_definitions_df, session_prefix,
                )

            PC1 = np.append(PC1, np.nan)
            PC2 = np.append(PC2, np.nan)
            Feature_vector = np.vstack((
                Areas, Area_percentages, Perimeters, Area_perimeter_ratios,
                Circularities, Hull_areas, Centroidxs, Centroidys,
                Angles, Concavities, PC1, PC2,
            ))

            vid_base = current_video.split("_fullvideo")[0]
            output_filename = vid_base + "_FINAL_Feature_vector.npy"
            output_path = os.path.join(feature_folder, output_filename)
            np.save(output_path, Feature_vector)

            print(f"\n  Saved: {output_filename}")
            print(f"  Shape: {Feature_vector.shape}")

            processed_data[current_video] = {
                "features": video_features,
                "feature_vector": Feature_vector,
                "trial_definitions": trial_definitions_df,
                "session_prefix": session_prefix,
            }

        except Exception as e:
            print(f"\nERROR processing {current_video}: {e}")
            traceback.print_exc()
            continue

    if plot_combined and len(video_list) > 1:
        print(f"\n{'=' * 30}")
        print(f"Creating combined histogram for all {len(video_list)} videos")
        print(f"{'=' * 30}")
        plot_feature_histograms(
            all_video_features, " - Combined Videos", normal_filter_threshold=3.5
        )

    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 30}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 30}")
    print(f"Processed: {len(processed_data)}/{len(video_list)} videos")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    if not processed_data:
        print("Average time per video: N/A - no videos processed.")
    else:
        print(f"Average per video: {elapsed_time / len(processed_data):.2f} seconds")
    print(f"\nFeature vectors saved to: {feature_folder}")

    return processed_data


# ============================================================================
# Batch processing helpers
# ============================================================================

def normalize_filter(filter_value):
    """Convert filter to list format. None/empty -> None, string -> [string], list -> list."""
    if filter_value is None or filter_value == "" or filter_value == []:
        return None
    if isinstance(filter_value, str):
        return [filter_value]
    return filter_value


def should_exclude(name, exclusion_patterns):
    """Check if name should be excluded based on substring matching."""
    if not exclusion_patterns:
        return False, None
    for pattern in exclusion_patterns:
        if pattern in name:
            return True, pattern
    return False, None


def matches_prefix(name, prefix_patterns):
    """Check if name starts with ANY of the prefix patterns."""
    if not prefix_patterns:
        return True
    return any(name.startswith(prefix) for prefix in prefix_patterns)


def matches_group(session_name, group_patterns):
    """Check if session name ends with _{ANY} of the group patterns (TC/TP)."""
    if not group_patterns:
        return True
    return any(session_name.endswith(f"_{group}") for group in group_patterns)


def extract_session_from_video_name(video_name):
    """
    Extract session name from video name.
    Example: '2025_10_17_14_30_05_trial_1_TC_regions_100_200_fullvideo'
    Returns: '2025_10_17_14_30_05_trial_1_TC'
    """
    if "_regions_" in video_name:
        return video_name.split("_regions_")[0]
    return video_name


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Run the full batch feature extraction pipeline."""
    global MASKS_DIR, FEATURES_DIR

    MASKS_DIR = "data/Masks/"
    FEATURES_DIR = "data/Features/"
    THETA_PICKLE_DIR = "data/Planam_Wheels/"
    PCA_REF_DIR = "data/PCA_REF/"
    os.makedirs(FEATURES_DIR, exist_ok=True)

    SESSION_SUMMARIES_DIR = os.path.join(FEATURES_DIR, "session_summaries")
    os.makedirs(SESSION_SUMMARIES_DIR, exist_ok=True)

    SHOW_GRAPHS = False
    PLOT_INDIVIDUAL = False
    PLOT_COMBINED = False
    PLOT_TRIALS = False
    SAVE_SESSION_SUMMARY = False
    ASSUME_FINAL = True

    prefix_filter = normalize_filter(VIDEO_PREFIX_FILTER)
    group_filter = normalize_filter(VIDEO_GROUP_FILTER)
    exclusion_filter = normalize_filter(EXCLUSION_FILTER)

    print("=" * 70)
    print("FEATURE EXTRACTION BATCH PROCESSING")
    print("=" * 70)
    print("\nFilter Configuration:")
    if prefix_filter:
        print(f"  Prefix filter: {prefix_filter}")
    else:
        print("  Prefix filter: None (all prefixes)")
    if group_filter:
        print(f"  Group filter: {group_filter}")
    else:
        print("  Group filter: None (all groups)")
    if exclusion_filter:
        print(f"  Exclusion patterns: {exclusion_filter}")
    else:
        print("  Exclusion patterns: None (no exclusions)")

    # --- Discover mask files ---
    print("\n" + "=" * 70)
    print("DISCOVERING MASK FILES")
    print("=" * 70)

    all_mask_files = list(
        Path(MASKS_DIR).glob(f"*{LABEL_TYPE}_binary_masks*.npz")
    )
    print(f"\nFound {len(all_mask_files)} total mask files in {MASKS_DIR}")

    all_video_names = []
    for mask_file in all_mask_files:
        video_name = mask_file.stem.split("_binary_masks")[0]
        all_video_names.append(video_name)

    print(f"Extracted {len(all_video_names)} video names from mask files")

    all_sessions_set = set()
    for video_name in all_video_names:
        session = extract_session_from_video_name(video_name)
        all_sessions_set.add(session)

    all_sessions = sorted(list(all_sessions_set))
    print(f"Found {len(all_sessions)} unique sessions")

    if group_filter:
        group_filtered_sessions = [
            s for s in all_sessions if matches_group(s, group_filter)
        ]
        excluded_by_group = len(all_sessions) - len(group_filtered_sessions)
        if excluded_by_group > 0:
            group_list = ", ".join([f"'_{g}'" for g in group_filter])
            print(
                f"FILTERING: Excluded {excluded_by_group} sessions "
                f"not ending with {group_list}"
            )
        all_sessions = group_filtered_sessions

    excluded_sessions = []
    filtered_sessions = []
    for session in all_sessions:
        should_excl, matched = should_exclude(session, exclusion_filter)
        if should_excl:
            excluded_sessions.append((session, matched))
        else:
            filtered_sessions.append(session)

    if excluded_sessions:
        print(
            f"FILTERING: Excluded {len(excluded_sessions)} sessions "
            f"matching exclusion patterns:"
        )
        for session, pattern in excluded_sessions:
            print(f"  - {session} (matched '{pattern}')")

    all_sessions = filtered_sessions

    SESSIONS_TO_PROCESS = []
    excluded_videos = []

    for session in all_sessions:
        session_videos = [
            v for v in all_video_names
            if extract_session_from_video_name(v) == session
        ]

        filtered_videos = []
        for video_name in session_videos:
            if not matches_prefix(video_name, prefix_filter):
                continue
            should_excl, matched = should_exclude(video_name, exclusion_filter)
            if should_excl:
                excluded_videos.append((session, video_name, matched))
                continue
            filtered_videos.append(video_name)

        if filtered_videos:
            SESSIONS_TO_PROCESS.append(session)

    if excluded_videos:
        print(
            f"\nFILTERING: Excluded {len(excluded_videos)} videos "
            f"matching exclusion patterns:"
        )
        for session, video, pattern in excluded_videos:
            print(f"  - {video}")
            print(f"    (from session: {session}, matched '{pattern}')")

    if not SESSIONS_TO_PROCESS:
        print("\nERROR: No sessions found matching filters")
        print(f"  Prefix: {prefix_filter}")
        print(f"  Group: {group_filter}")
        print(f"  Exclusions: {exclusion_filter}")
        print(f"  Searched in: {MASKS_DIR}")
        raise ValueError("No matching sessions found")

    print(f"\n{'=' * 70}")
    print(f"SESSIONS TO PROCESS: {len(SESSIONS_TO_PROCESS)}")
    print(f"{'=' * 70}")
    for idx, session in enumerate(SESSIONS_TO_PROCESS, 1):
        print(f"  {idx}. {session}")
    print()

    # --- Process each session ---
    all_sessions_results = {}

    for session_idx, SESSION in enumerate(SESSIONS_TO_PROCESS, 1):
        print("\n" + "=" * 70)
        print(f"SESSION {session_idx}/{len(SESSIONS_TO_PROCESS)}: {SESSION}")
        print("=" * 70 + "\n")

        session_videos = [
            v for v in all_video_names
            if extract_session_from_video_name(v) == SESSION
        ]

        videos_to_process = []
        for video_name in session_videos:
            if not matches_prefix(video_name, prefix_filter):
                continue
            should_excl, matched = should_exclude(video_name, exclusion_filter)
            if should_excl:
                continue
            videos_to_process.append(video_name)

        if ASSUME_FINAL:
            # Prefer FINAL mask files; fall back to _0_1799 if no FINAL exists
            final_vids = [v for v in videos_to_process if "_FINAL_" in v]
            if final_vids:
                videos_to_process = final_vids
            else:
                videos_to_process = [
                    v for v in videos_to_process if "_0_1799" in v
                ]
        else:
            videos_to_process = [v for v in videos_to_process if "_0_1799" in v]

        if not videos_to_process:
            print(f"WARNING: No videos found for {SESSION} after filtering")
            continue

        print(f"Found {len(videos_to_process)} videos to process:")

        for video in videos_to_process:
            regions = (
                video.split("_regions_")[1] if "_regions_" in video else "unknown"
            )
            print(f"  - regions_{regions}")
        print()

        try:
            plot_individual = SHOW_GRAPHS and PLOT_INDIVIDUAL
            plot_combined = SHOW_GRAPHS and PLOT_COMBINED
            plot_trials = SHOW_GRAPHS and PLOT_TRIALS

            results = run_feature_extraction_pipeline(
                THETA_PICKLE_DIR,
                PCA_REF_DIR,
                LABEL_TYPE,
                reset_reference,
                video_list=videos_to_process,
                feature_folder=FEATURES_DIR,
                plot_individual=plot_individual,
                plot_combined=plot_combined,
                plot_trials=plot_trials,
                assume_final=ASSUME_FINAL,
            )

            all_sessions_results[SESSION] = results

            if SAVE_SESSION_SUMMARY and results:
                print(f"\n{'=' * 70}")
                print(f"SAVING SESSION SUMMARY: {SESSION}")
                print(f"{'=' * 70}")

                summary_rows = []
                for video_name, video_data in results.items():
                    features = video_data["features"]
                    feature_vector = video_data["feature_vector"]

                    if "_regions_" in video_name:
                        regions = video_name.split("_regions_")[1].split(
                            "_fullvideo"
                        )[0]
                    else:
                        regions = "unknown"

                    row = {
                        "session": SESSION,
                        "video_name": video_name,
                        "regions": regions,
                        "n_frames": feature_vector.shape[1],
                    }

                    for feature_idx, feature_name in enumerate([
                        "Areas", "Area_percentages", "Perimeters",
                        "Area_perimeter_ratios", "Circularities",
                        "Hull_areas", "Centroidxs", "Centroidys",
                        "Angles", "Concavities",
                    ]):
                        feature_data = features[feature_name]
                        row[f"{feature_name}_mean"] = np.nanmean(feature_data)
                        row[f"{feature_name}_std"] = np.nanstd(feature_data)
                        row[f"{feature_name}_nan_pct"] = (
                            100 * np.sum(np.isnan(feature_data)) / len(feature_data)
                        )

                    summary_rows.append(row)

                session_df = pd.DataFrame(summary_rows)
                session_csv_path = os.path.join(
                    SESSION_SUMMARIES_DIR, f"{SESSION}_feature_summary.csv"
                )
                session_df.to_csv(session_csv_path, index=False)

                print(f" Saved session summary: {session_csv_path}")
                print(f"  Videos in summary: {len(summary_rows)}")
                print()

            print(f"\n{'=' * 70}")
            print(f"SESSION {SESSION} COMPLETE")
            print(f"{'=' * 70}")

        except Exception as e:
            print(f"\nERROR processing session {SESSION}: {e}")
            traceback.print_exc()
            continue

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(
        f"\nSessions processed: "
        f"{len(all_sessions_results)}/{len(SESSIONS_TO_PROCESS)}"
    )
    print(f"Feature vectors saved to: {FEATURES_DIR}")
    if SAVE_SESSION_SUMMARY:
        print(f"Session summaries saved to: {SESSION_SUMMARIES_DIR}")

    total_videos = sum(len(r) for r in all_sessions_results.values())
    print(f"Total videos processed: {total_videos}")

    print(f"Sessions processed: {len(SESSIONS_TO_PROCESS)}")
    for idx, session in enumerate(SESSIONS_TO_PROCESS, 1):
        print(f"  {idx}. {session}")


# Module-level variable needed by run_feature_extraction_pipeline
MASKS_DIR = "data/Masks/"
FEATURES_DIR = "data/Features/"


if __name__ == "__main__":
    main()
