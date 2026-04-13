"""KL divergence analysis pipeline for Tasmanian conditioning experiments.

Computes KL divergence of feature means across trial and inter-trial intervals
from full-length mask feature vectors. Supports batch processing across sessions,
per-feature KL decomposition, optional null distribution testing, and saves
per-video, per-session, and combined results.

Usage::

    python LF_5_Calculate_KL.py
"""

import csv
import datetime
import glob
import json
import os
import re
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Configuration constants
# ============================================================================

"""
VIDEO_PREFIX_FILTER = [
    # TC-6 sessions
    "2025_10_14_10_25_19_trial_1_TC",
    "2025_10_14_14_13_15_trial_1_TC",
    "2025_10_15_10_09_28_trial_1_TC",
    "2025_10_15_14_16_21_trial_1_TC",
    "2025_10_16_09_59_15_trial_1_TC",
    "2025_10_16_14_12_17_trial_1_TC",
    "2025_10_17_10_05_03_trial_1_TC",
    "2025_10_17_14_18_23_trial_1_TC",
    # TC-7 sessions
    "2025_10_14_10_35_03_trial_1_TC",
    "2025_10_14_14_25_09_trial_1_TC",
    "2025_10_15_10_20_58_trial_1_TC",
    "2025_10_15_14_30_16_trial_1_TC",
    "2025_10_16_10_10_37_trial_1_TC",
    "2025_10_16_14_28_16_trial_1_TC",
    "2025_10_17_10_18_17_trial_1_TC",
    "2025_10_17_14_30_05_trial_1_TC",
    # TP-3 sessions
    "2025_10_14_10_47_52_trial_1_TP",
    "2025_10_14_14_35_32_trial_1_TP",
    "2025_10_15_10_37_01_trial_1_TP",
    "2025_10_15_14_43_14_trial_1_TP",
    "2025_10_16_10_24_06_trial_1_TP",
    "2025_10_16_14_48_01_trial_1_TP",
    "2025_10_17_10_34_57_trial_1_TP",
    "2025_10_17_14_45_34_trial_1_TP",
    # TP-4 sessions
    "2025_10_14_11_00_58_trial_1_TP",
    "2025_10_14_14_49_36_trial_1_TP",
    "2025_10_15_10_52_34_trial_1_TP",
    "2025_10_15_14_57_00_trial_1_TP",
    "2025_10_16_10_37_16_trial_1_TP",
    "2025_10_16_15_00_56_trial_1_TP",
    "2025_10_17_10_48_37_trial_1_TP",
    "2025_10_17_14_54_47_trial_1_TP",
]
"""

VIDEO_GROUP_FILTER = None
EXCLUSION_FILTER = []
LABEL_TYPE = "points"

FEATURE_NAMES = [
    "Areas", "Area_percentages", "Perimeters", "Area_perimeter_ratios",
    "Circularities", "Hull_areas", "Centroidxs", "Centroidys",
    "Angles", "Concavities", "PC1", "PC2",
]


# ============================================================================
# KL divergence functions
# ============================================================================

def kl_gaussian_diag(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute KL divergence KL(A || B) assuming diagonal (independent) Gaussians.

    Parameters
    ----------
    A : np.ndarray, shape (n_samples, n_features)
        Samples from distribution A
    B : np.ndarray, shape (m_samples, n_features)
        Samples from distribution B
    eps : float
        Small value added to variances for numerical stability

    Returns
    -------
    kl_total : float
        KL(A || B) = sum of 1D Gaussian KL divergences
    kl_per_feature : np.ndarray
        Per-feature KL contributions
    """
    mu_A = np.nanmean(A, axis=0)
    mu_B = np.nanmean(B, axis=0)

    var_A = np.nanvar(A, axis=0, ddof=1) + eps
    var_B = np.nanvar(B, axis=0, ddof=1) + eps

    kl_per_feature = 0.5 * (
        np.log(var_B / var_A) + (var_A + (mu_A - mu_B) ** 2) / var_B - 1.0
    )
    kl_total = np.sum(kl_per_feature)

    return kl_total, kl_per_feature


def kl_gaussian_full(A: np.ndarray, B: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute KL divergence KL(A || B) for full multivariate Gaussians.

    Parameters
    ----------
    A : np.ndarray, shape (n_samples, n_features)
    B : np.ndarray, shape (m_samples, n_features)
    eps : float
        Regularisation added to covariance diagonal

    Returns
    -------
    float : KL(A || B)
    """
    n_samples, d = A.shape

    mu_A = np.nanmean(A, axis=0)
    mu_B = np.nanmean(B, axis=0)

    Sigma_A = np.cov(A, rowvar=False) + eps * np.eye(d)
    Sigma_B = np.cov(B, rowvar=False) + eps * np.eye(d)

    inv_Sigma_B = np.linalg.inv(Sigma_B)
    diff = mu_B - mu_A

    trace_term = np.trace(inv_Sigma_B @ Sigma_A)
    mahalanobis_term = diff.T @ inv_Sigma_B @ diff
    logdet_term = np.log(np.linalg.det(Sigma_B) / np.linalg.det(Sigma_A))

    kl_div = 0.5 * (trace_term + mahalanobis_term - d + logdet_term)

    return kl_div


# ============================================================================
# CS-on / CS-off frame extraction
# ============================================================================

def extract_CSon_CSoff_frames(
    Feature_vector, trial_definitions_df, feature_names, verbose=True
):
    """
    Extract CS-on and CS-off frames from full feature vector.

    Parameters
    ----------
    Feature_vector : np.ndarray, shape (n_features, n_frames)
    trial_definitions_df : pd.DataFrame
        Must contain columns 'trial_num', 'start_frame', 'end_frame'.
    feature_names : list
    verbose : bool

    Returns
    -------
    CS_on_samples : np.ndarray, shape (n_CS_on_frames, n_features)
    CS_off_samples : np.ndarray, shape (n_CS_off_frames, n_features)
    extraction_info : dict
    """
    if verbose:
        print("=" * 30)
        print("Extracting CS-on frames (during trials)...")

    CS_on_frames = []
    CS_on_frame_indices = []

    for idx, row in trial_definitions_df.iterrows():
        trial_num = row["trial_num"]
        start_frame = row["start_frame"]
        end_frame = row["end_frame"]

        trial_frames = Feature_vector[:, start_frame : end_frame + 1]
        CS_on_frames.append(trial_frames)
        CS_on_frame_indices.extend(range(start_frame, end_frame + 1))

        if verbose:
            print(
                f"  Trial {trial_num}: frames {start_frame}-{end_frame} "
                f"({end_frame - start_frame + 1} frames)"
            )

    CS_on_data = np.concatenate(CS_on_frames, axis=1)
    if verbose:
        print(f"Total CS-on frames: {CS_on_data.shape[1]}")

    if verbose:
        print("\nExtracting CS-off frames (inter-trial intervals only)...")

    CS_off_frames = []
    CS_off_frame_indices = []

    off_sess_num = len(trial_definitions_df) - 1
    if (
        int(trial_definitions_df["end_frame"][len(trial_definitions_df) - 1])
        != len(Feature_vector[0])
    ):
        off_sess_num = off_sess_num + 1

    for idx in range(off_sess_num):
        if idx == (off_sess_num - 1):
            interval_start = trial_definitions_df.iloc[idx]["end_frame"] + 1
            interval_end = len(Feature_vector[0])
        else:
            current_trial = trial_definitions_df.iloc[idx]
            next_trial = trial_definitions_df.iloc[idx + 1]
            interval_start = current_trial["end_frame"] + 1
            interval_end = next_trial["start_frame"] - 1

        if interval_end >= interval_start:
            interval_frames = Feature_vector[:, interval_start : interval_end + 1]
            CS_off_frames.append(interval_frames)
            CS_off_frame_indices.extend(range(interval_start, interval_end + 1))
            if verbose:
                print(
                    f"  After trial {idx + 1}: frames {interval_start}-{interval_end} "
                    f"({interval_end - interval_start + 1} frames)"
                )
        else:
            if verbose:
                print(f"  After trial {idx + 1}: No interval (trials adjacent)")

    if CS_off_frames:
        CS_off_data = np.concatenate(CS_off_frames, axis=1)
        if verbose:
            print(f"Total CS-off frames: {CS_off_data.shape[1]}")
    else:
        raise ValueError(
            "No CS-off frames found! Trials may be adjacent with no inter-trial intervals."
        )

    if verbose:
        print("\nTransposing to (samples, features) format...")
    CS_on_samples = CS_on_data.T
    CS_off_samples = CS_off_data.T

    if verbose:
        print(f"  CS-on:  {CS_on_samples.shape}")
        print(f"  CS-off: {CS_off_samples.shape}")

    CS_on_nans = np.sum(np.isnan(CS_on_samples), axis=0)
    CS_off_nans = np.sum(np.isnan(CS_off_samples), axis=0)

    if verbose:
        print("\nNaN analysis:")
        print(
            f"{'Feature':<20} {'CS-on NaNs':<15} {'CS-off NaNs':<15} "
            f"{'CS-on %':<10} {'CS-off %':<10}"
        )
        print("-" * 70)
        for i, fname in enumerate(feature_names):
            cs_on_pct = 100 * CS_on_nans[i] / CS_on_samples.shape[0]
            cs_off_pct = 100 * CS_off_nans[i] / CS_off_samples.shape[0]
            print(
                f"{fname:<20} {CS_on_nans[i]:<15} {CS_off_nans[i]:<15} "
                f"{cs_on_pct:<10.2f} {cs_off_pct:<10.2f}"
            )

    extraction_info = {
        "n_CS_on_frames": CS_on_samples.shape[0],
        "n_CS_off_frames": CS_off_samples.shape[0],
        "CS_on_frame_indices": CS_on_frame_indices,
        "CS_off_frame_indices": CS_off_frame_indices,
        "CS_on_nans_per_feature": CS_on_nans.tolist(),
        "CS_off_nans_per_feature": CS_off_nans.tolist(),
    }

    return CS_on_samples, CS_off_samples, extraction_info


# ============================================================================
# KL computation wrapper
# ============================================================================

def compute_kl_divergence(CS_on_samples, CS_off_samples, feature_names):
    """
    Compute KL divergence between CS-on and CS-off distributions.

    Returns
    -------
    results_dict : dict
        Contains 'kl_all' (float) and 'kl_per_feature' (np.ndarray).
    """
    print("=" * 30)
    print("Computing KL divergence...")

    kl, kl_per_feature = kl_gaussian_diag(CS_on_samples, CS_off_samples)
    print(f"\nKL(CS-on || CS-off) = {kl:.6f}")

    print(f"\n{'=' * 30}")
    print("Per-feature KL contributions:")
    print(f"{'=' * 30}")
    print(f"{'Feature':<20} {'KL contrib':<15} {'% of total':<12}")
    print("-" * 50)

    results_dict = {
        "kl_all": float(kl),
        "kl_per_feature": kl_per_feature,
    }

    return results_dict


# ============================================================================
# CSV split helpers
# ============================================================================

def _parse_start_end_from_row(row):
    rest = ",".join(row[1:]).strip()
    groups = re.findall(r"\[(.*?)\]", rest)
    starts, ends = [], []
    if len(groups) >= 1:
        starts = [int(x) for x in groups[0].split(",") if x.strip()]
    if len(groups) >= 2:
        ends = [int(x) for x in groups[1].split(",") if x.strip()]
    return starts, ends


def get_session_splits(csv_path, session_name):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            date = row[0].strip()
            if date != session_name:
                continue
            return _parse_start_end_from_row(row)
    raise ValueError(f"Session {session_name!r} not found in CSV")


def remove_poor_values(feature_vector, session):
    csv_path = "hand_scored_datasheets/video_splits.csv"
    start_frames, end_frames = get_session_splits(csv_path, session)

    for s, e in zip(start_frames, end_frames):
        feature_vector[:, s : e + 1] = np.nan

    return feature_vector


# ============================================================================
# Results dataframe builder
# ============================================================================

def create_results_dataframe(
    video_name, session_prefix, results_dict, feature_names, extraction_info
):
    """Create a DataFrame with KL divergence results for one video."""
    try:
        if "_regions_" in video_name and "_fullvideo" in video_name:
            regions_part = video_name.split("_regions_")[1].split("_fullvideo")[0]
            worm_regions = regions_part
        else:
            worm_regions = "unknown"
    except Exception as e:
        print(f"Warning: Could not extract worm_regions from video name: {e}")
        worm_regions = "unknown"

    main_results = {
        "video": video_name,
        "session": session_prefix,
        "Troupe": "",
        "Day": "",
        "Block": "",
        "worm_regions": worm_regions,
        "n_CSon_frames": extraction_info["n_CS_on_frames"],
        "n_CSoff_frames": extraction_info["n_CS_off_frames"],
        "KL": results_dict["kl_all"],
        "KL_per_feature": results_dict["kl_per_feature"],
    }

    for i, feature_data in enumerate(results_dict["kl_per_feature"]):
        main_results[f"{feature_names[i]}_KL"] = feature_data

    df = pd.DataFrame([main_results])
    return df


# ============================================================================
# Master analysis function
# ============================================================================

def analyze_video_kl_divergence(
    video_name,
    home_dir,
    holylabs_dir,
    LABEL_TYPE,
    save_results=True,
    output_dir=None,
    compute_null=False,
    show_null_diagnostic=True,
):
    """
    Master function to compute KL divergence between CS-on and CS-off for a
    single video.

    Parameters
    ----------
    video_name : str
        Video name (without _Feature_vector.npy extension).
    home_dir : str
        Path to HOME directory.
    holylabs_dir : str
        Path to holylabs directory.
    LABEL_TYPE : str
        'points' or 'boxes'.
    save_results : bool
        Whether to save results to JSON/CSV.
    output_dir : str, optional
        Where to save results (default: holylabs/KL_Results).
    compute_null : bool
        Whether to compute circshift null distribution.
    show_null_diagnostic : bool
        Whether to display diagnostic plot of KL vs shift.

    Returns
    -------
    results_df : pd.DataFrame
    full_results : dict
    """
    print("=" * 70)
    print("KL DIVERGENCE ANALYSIS")
    print("=" * 70)
    print(f"Video: {video_name}")

    feature_names = FEATURE_NAMES

    features_folder = os.path.join("data", "Features")
    raw_data_folder = os.path.join("data", "Raw_data")

    feature_file = os.path.join(features_folder, f"{video_name}")

    if "_regions_" in video_name:
        session_prefix = video_name.split("_regions_")[0]
    else:
        raise ValueError(f"Could not extract session prefix from: {video_name}")

    # Hard-coded path overrides for sessions stored in a different location
    _HL_ALT = (
        "data/Raw_data/Stuff_already_on_HL"
    )
    _ALT_SESSIONS = {
        "2025_10_15_14_16_21_trial_1_TC",
        "2025_10_16_14_12_17_trial_1_TC",
        "2025_10_17_10_05_03_trial_1_TC",
        "2025_10_16_09_59_15_trial_1_TC",
        "2025_10_17_14_18_23_trial_1_TC",
    }

    if session_prefix in _ALT_SESSIONS:
        trial_csv_path = os.path.join(
            _HL_ALT, session_prefix, "trial_definitions.csv"
        )
    else:
        trial_csv_path = os.path.join(
            raw_data_folder, session_prefix, "trial_definitions.csv"
        )

    print(f"Session: {session_prefix}")

    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    if not os.path.exists(trial_csv_path):
        raise FileNotFoundError(f"Trial definitions not found: {trial_csv_path}")

    print("\nLoading data...")
    Feature_vector = np.load(feature_file)
    print(f"  Feature vector: {Feature_vector.shape}")

    trial_definitions_df = pd.read_csv(trial_csv_path)
    print(f"  Trial definitions: {len(trial_definitions_df)} trials")

    CS_on_samples, CS_off_samples, extraction_info = extract_CSon_CSoff_frames(
        Feature_vector, trial_definitions_df, feature_names
    )

    results_dict = compute_kl_divergence(
        CS_on_samples, CS_off_samples, feature_names
    )

    results_df = create_results_dataframe(
        video_name, session_prefix, results_dict, feature_names, extraction_info
    )

    full_results = {
        "video": video_name,
        "session": session_prefix,
        "extraction_info": extraction_info,
        "kl_results": results_dict,
    }

    if save_results:
        if output_dir is None:
            output_dir = os.path.join("data", "KL_Results")
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, f"{video_name}_KL_results.json")
        with open(json_path, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\n{'=' * 70}")
        print(f"Results saved to: {json_path}")

        csv_path = os.path.join(output_dir, f"{video_name}_KL_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")

    print(f"\n{'=' * 70}")
    print("Analysis complete")
    print("=" * 70)

    return results_df, full_results


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


def extract_session_from_feature_file(filename, label_type):
    """
    Extract session name from feature file.
    Example: '..._trial_1_TC_regions_100_200_fullvideo_Feature_vector.npy'
    Returns: '2025_10_17_14_30_05_trial_1_TC'
    """
    basename = os.path.basename(filename)
    basename = basename.replace(f"_{label_type}_Feature_vector.npy", "")
    if "_regions_" in basename:
        return basename.split("_regions_")[0]
    return basename


def convert_numpy(obj):
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Run the full KL divergence batch pipeline."""

    KL_RESULTS_DIR = os.path.join("data", "KL_divergence_results")
    INDIVIDUAL_METADATA_DIR = os.path.join(KL_RESULTS_DIR, "individual_video_KL_metadata")
    INDIVIDUAL_SESSION_DIR = os.path.join(KL_RESULTS_DIR, "individual_session_KL_results")

    os.makedirs(KL_RESULTS_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_METADATA_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_SESSION_DIR, exist_ok=True)

    SAVE_RESULTS = True
    COMPUTE_NULL = False
    SHOW_NULL_DIAGNOSTIC = False

    prefix_filter = normalize_filter(VIDEO_PREFIX_FILTER)
    group_filter = normalize_filter(VIDEO_GROUP_FILTER)
    exclusion_filter = normalize_filter(EXCLUSION_FILTER)

    print("=" * 70)
    print("KL DIVERGENCE BATCH PROCESSING")
    print("=" * 70)

    # --- Discover feature files ---
    print("\n" + "=" * 70)
    print("DISCOVERING FEATURE FILES")
    print("=" * 70)

    features_folder = os.path.join("data", "Features")
    all_feature_files = glob.glob(
        os.path.join(features_folder, "*_FINAL_Feature_vector.npy")
    )
    print(f"\nFound {len(all_feature_files)} total feature files in {features_folder}")

    matched_files = [
        path
        for path in all_feature_files
        if any(sid in path for sid in VIDEO_PREFIX_FILTER)
    ]

    if matched_files:
        print(f"Found {len(matched_files)} matching files:")
        for file in matched_files:
            print(f" - {file}")
    else:
        print("No files found matching the VIDEO_PREFIX_FILTER.")

    all_sessions = []
    for feature_file in matched_files:
        session = extract_session_from_feature_file(feature_file, LABEL_TYPE)
        all_sessions.append(session)

    all_sessions = sorted([s for s in all_sessions if s.startswith("2025")])
    print(f"Found {len(all_sessions)} unique sessions")

    SESSIONS_TO_PROCESS = all_sessions

    print(f"\n{'=' * 70}")
    print(f"SESSIONS TO PROCESS: {len(SESSIONS_TO_PROCESS)}")
    print(f"{'=' * 70}")
    for idx, session in enumerate(SESSIONS_TO_PROCESS, 1):
        print(f"  {idx}. {session}")
    print()

    # --- Process each session ---
    all_sessions_results = []
    SESSIONS_TO_PROCESS = np.unique(SESSIONS_TO_PROCESS).tolist()

    for session_idx, SESSION in enumerate(SESSIONS_TO_PROCESS, 1):
        print("\n" + "=" * 70)
        print(f"SESSION {session_idx}/{len(SESSIONS_TO_PROCESS)}: {SESSION}")
        print("=" * 70 + "\n")

        pattern = os.path.join(
            "data", "Features",
            f"{SESSION}_regions_*_FINAL_Feature_vector.npy",
        )
        feature_files = glob.glob(pattern)

        session_results = []

        for idx, feature_file in enumerate(feature_files, 1):
            if "FINAL" not in feature_file:
                continue

            video = os.path.basename(feature_file).replace(
                f"{LABEL_TYPE}_Feature_vector.npy", ""
            )
            regions = video.split("_regions_")[1].split("_fullvideo")[0]
            print(f"  [{idx}/{len(feature_files)}] regions_{regions}")

            try:
                df, results = analyze_video_kl_divergence(
                    video_name=video,
                    home_dir="data",
                    holylabs_dir="data",
                    LABEL_TYPE=LABEL_TYPE,
                    save_results=False,
                    compute_null=COMPUTE_NULL,
                    show_null_diagnostic=SHOW_NULL_DIAGNOSTIC,
                )
                session_results.append(df)
                all_sessions_results.append(df)

                kl = results["kl_results"]["kl_all"]
                print(f"      KL = {kl:.4f}")

                if COMPUTE_NULL and results.get("null_statistics") is not None:
                    p_val = results["null_statistics"]["on_given_off"][
                        "p_value_one_tailed"
                    ]
                    print(f"      p-value (one-tailed) = {p_val:.4f}")

                if SAVE_RESULTS:
                    individual_json = os.path.join(
                        INDIVIDUAL_METADATA_DIR, f"{video}_KL_results.json"
                    )
                    results_serializable = convert_numpy(results)
                    with open(individual_json, "w") as f:
                        json.dump(results_serializable, f, indent=2)
                    print(
                        f"      Metadata saved: {os.path.basename(individual_json)}"
                    )

            except Exception as e:
                print(f"      ERROR: {e}")
                traceback.print_exc()
                continue

        if session_results and SAVE_RESULTS:
            session_df = pd.concat(session_results, ignore_index=True)
            print(f"\n  Session {SESSION} summary:")
            print(f"    Videos processed: {len(session_df)}")

            if (
                COMPUTE_NULL
                and "p_one_tailed_CSon_given_CSoff" in session_df.columns
            ):
                n_sig = np.sum(
                    session_df["p_one_tailed_CSon_given_CSoff"] < 0.05
                )
                print(
                    f"    Significant (p < 0.05): {n_sig}/{len(session_df)} worms"
                )

            session_csv = os.path.join(
                INDIVIDUAL_SESSION_DIR, f"{SESSION}_KL_results.csv"
            )
            session_df.to_csv(session_csv, index=False)
            print(f"    Saved: {session_csv}")

    # --- Combined results ---
    if all_sessions_results:
        print("\n" + "=" * 70)
        print("COMBINED RESULTS - ALL SESSIONS")
        print("=" * 70)

        combined_df = pd.concat(all_sessions_results, ignore_index=True)

        print(f"Total videos: {len(combined_df)}")
        print(f"Total sessions: {len(SESSIONS_TO_PROCESS)}")

        if (
            COMPUTE_NULL
            and "p_one_tailed_CSon_given_CSoff" in combined_df.columns
        ):
            print("\nNull hypothesis testing:")
            n_sig_05 = np.sum(
                combined_df["p_one_tailed_CSon_given_CSoff"] < 0.05
            )
            n_sig_01 = np.sum(
                combined_df["p_one_tailed_CSon_given_CSoff"] < 0.01
            )
            print(
                f"  Significant at p < 0.05: {n_sig_05}/{len(combined_df)} "
                f"({100 * n_sig_05 / len(combined_df):.1f}%)"
            )
            print(
                f"  Significant at p < 0.01: {n_sig_01}/{len(combined_df)} "
                f"({100 * n_sig_01 / len(combined_df):.1f}%)"
            )

        print("\nPer-session summary:")

        if SAVE_RESULTS:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            master_filename = (
                f"Tasmanian_Conditioning_KL_Results_COMPILED_{timestamp}.csv"
            )
            master_csv = os.path.join(KL_RESULTS_DIR, master_filename)
            combined_df.to_csv(master_csv, index=False)

            print(f"\n{'=' * 70}")
            print("MASTER FILE SAVED")
            print(f"{'=' * 70}")
            print(f"Filename: {master_filename}")
            print(f"Full path: {master_csv}")
            print("\nUse this file for learning curve plotting (KL vs. Day)")
    else:
        print(
            "\nNo results to combine - all sessions failed or had no videos."
        )


if __name__ == "__main__":
    main()
