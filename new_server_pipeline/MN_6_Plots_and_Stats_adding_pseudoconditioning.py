"""KL divergence plotting and statistical analysis pipeline.

Analyses KL divergence CSV outputs from Notebook 5, enriches them with
experimental metadata, generates per-troupe learning-curve plots (TC vs TP),
and runs linear mixed model (LMM) statistics to test for day-over-day trends
and condition differences.

Usage::

    python LF_6_Plots_and_Stats_adding_pseudoconditioning.py
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Optional: statsmodels may not be available in all environments
try:
    import statsmodels.formula.api as smf

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ============================================================================
# Configuration constants
# ============================================================================

TROUPE_LIST = ["TC-6", "TC-7", "TP-3", "TP-4"]

FEATURE_NAMES = [
    "Areas", "Area_percentages", "Perimeters", "Area_perimeter_ratios",
    "Circularities", "Hull_areas", "Centroidxs", "Centroidys",
    "Angles", "Concavities",
]


# ============================================================================
# STEP 1: Metadata enrichment
# ============================================================================

def update_kl_with_log_info(kl_csv_path, log_csv_path, output_csv_path=None):
    """
    Update KL divergence CSV with experimental log information.

    Args:
        kl_csv_path: Path to KL divergence CSV file.
        log_csv_path: Path to experimental log CSV file.
        output_csv_path: Path where updated CSV should be saved (None to skip).

    Returns:
        Updated DataFrame with log information added.
    """
    kl_df = pd.read_csv(kl_csv_path)
    log_df = pd.read_csv(log_csv_path)

    log_dict = {}
    for _, row in log_df.iterrows():
        data_folder = row["Data_Folder"]
        log_dict[data_folder] = {
            "Run": data_folder,
            "Troupe": row["Troupe"],
            "Day": row["Day"],
            "Block": row["Block"],
        }

    kl_df["Run"] = kl_df["session"].map(
        lambda x: log_dict.get(x, {}).get("Run", None)
    )
    kl_df["Troupe"] = kl_df["session"].map(
        lambda x: log_dict.get(x, {}).get("Troupe", None)
    )
    kl_df["Day"] = kl_df["session"].map(
        lambda x: log_dict.get(x, {}).get("Day", None)
    )
    kl_df["Block"] = kl_df["session"].map(
        lambda x: log_dict.get(x, {}).get("Block", None)
    )

    if output_csv_path is not None:
        kl_df.to_csv(output_csv_path, index=False)
        print(f"Updated KL CSV saved to: {output_csv_path}")

    print(f"Total rows: {len(kl_df)}")
    print(
        f"Rows with complete log info: "
        f"{kl_df[['Run', 'Troupe', 'Day', 'Block']].notna().all(axis=1).sum()}"
    )

    return kl_df


# ============================================================================
# STEP 2: Worm position extraction and dataframe creation
# ============================================================================

def extract_worm_positions_from_session(session_data):
    """
    Extract worm positions for all videos in a session.
    Videos are sorted by the first coordinate in their region, and assigned
    W1, W2, etc.
    """
    video_regions = []

    for idx, row in session_data.iterrows():
        video_name = row["video"]
        region_pattern = r"regions_(\d+)_(\d+)_(\d+)_(\d+)"
        match = re.search(region_pattern, video_name)

        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            video_regions.append((x1, y1, x2, y2, idx))
        else:
            print(f"Warning: Could not find region in {video_name}")
            video_regions.append((float("inf"), 0, 0, 0, idx))

    sorted_regions = sorted(video_regions, key=lambda x: x[0])

    session_data = session_data.copy()
    session_data["Worm"] = None
    for worm_num, (x1, y1, x2, y2, idx) in enumerate(sorted_regions, 1):
        session_data.loc[idx, "Worm"] = worm_num

    return session_data


def create_worm_kl_dataframe_from_df(kl_df):
    """
    Create a dataframe with columns for each worm's KL values.

    Returns:
        DataFrame with columns: Run, Troupe, Day, Block,
        W1_KL, W2_KL, ...
    """
    processed_dfs = []
    for session_name, session_data in kl_df.groupby("session"):
        session_with_worms = extract_worm_positions_from_session(session_data)
        processed_dfs.append(session_with_worms)

    kl_df = pd.concat(processed_dfs, ignore_index=True)

    pivot_data = []
    for session_name, session_data in kl_df.groupby("session"):
        row_data = {
            "Run": session_data["Run"].iloc[0],
            "Troupe": session_data["Troupe"].iloc[0],
            "Day": session_data["Day"].iloc[0],
            "Block": session_data["Block"].iloc[0],
        }
        for _, worm_data in session_data.iterrows():
            worm_num = worm_data["Worm"]
            if pd.notna(worm_num):
                row_data[f"W{int(worm_num)}_KL"] = worm_data["KL"]
        pivot_data.append(row_data)

    return pd.DataFrame(pivot_data)


def create_dataframe_for_LMM(kl_df):
    """Create a tidy dataframe suitable for LMM analysis."""
    processed_dfs = []
    for session_name, session_data in kl_df.groupby("session"):
        session_with_worms = extract_worm_positions_from_session(session_data)
        processed_dfs.append(session_with_worms)

    kl_df = pd.concat(processed_dfs, ignore_index=True)

    pivot_data = []
    for session_name, session_data in kl_df.groupby("session"):
        row_data = {
            "Run": session_data["Run"].iloc[0],
            "Troupe": session_data["Troupe"].iloc[0],
            "Day": session_data["Day"].iloc[0],
            "Block": session_data["Block"].iloc[0],
        }
        for _, worm_data in session_data.iterrows():
            worm_num = worm_data["Worm"]
            if pd.notna(worm_num):
                row_data[f"W{int(worm_num)}_KL"] = worm_data["KL"]
                for feature in FEATURE_NAMES:
                    row_data[f"W{int(worm_num)}_{feature}_KL"] = worm_data[
                        f"{feature}_KL"
                    ]
        pivot_data.append(row_data)

    return pd.DataFrame(pivot_data)


def build_long_format(updated_kl_df):
    """Melt the wide worm-KL dataframe into long format for plotting / LMMs.

    Returns
    -------
    df_long : pd.DataFrame
        One row per worm-day-block observation with columns including
        'Day', 'KL', 'Subject_ID', 'Troupe', etc.
    """
    worm_kl_df = create_dataframe_for_LMM(updated_kl_df)

    id_cols = ["Run", "Troupe", "Day", "Block"]
    df_melted = pd.melt(
        worm_kl_df, id_vars=id_cols, var_name="Full_Metric", value_name="Value"
    )

    df_split = df_melted["Full_Metric"].str.split("_", n=1, expand=True)
    df_melted["Subject_ID"] = df_split[0]
    df_melted["Metric"] = df_split[1]

    df_long = (
        df_melted.pivot_table(
            index=id_cols + ["Subject_ID"],
            columns="Metric",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
    )

    df_long["Subject_ID"] = (
        df_long["Subject_ID"].str.replace("W", "").astype(int).astype(str)
    )
    df_long["Day"] = pd.to_numeric(df_long["Day"], errors="coerce")
    df_long["KL"] = pd.to_numeric(df_long["KL"], errors="coerce")
    df_long = df_long.dropna(subset=["Day", "KL"])

    return df_long


# ============================================================================
# STEP 3: Plotting functions
# ============================================================================

def plot_troupe_kl_over_days(
    worm_kl_df,
    troupe_list,
    plot_null=True,
    figsize=(10, 6),
    plot_individual_lines=False,
):
    """
    Plot average KL divergence over days for TC and TP troupes on separate
    lines, with optional individual worm traces.

    Returns:
        matplotlib figure
    """
    tc_troupes = [t for t in troupe_list if "TC" in t]
    tp_troupes = [t for t in troupe_list if "TP" in t]

    fig, ax = plt.subplots(figsize=figsize)

    tc_color = "#1f4788"
    tp_color = "#9e778a"

    def get_worm_cols(df):
        return [c for c in df.columns if c.startswith("W") and c.endswith("_KL")]

    # --- TC troupes ---
    if tc_troupes:
        filtered_df_tc = worm_kl_df[worm_kl_df["Troupe"].isin(tc_troupes)].copy()
        kl_columns = get_worm_cols(filtered_df_tc)
        filtered_df_tc["Mean_KL"] = filtered_df_tc[kl_columns].mean(axis=1)

        if plot_individual_lines:
            for troupe in tc_troupes:
                troupe_df = filtered_df_tc[filtered_df_tc["Troupe"] == troupe]
                for worm_col in kl_columns:
                    worm_day_means = troupe_df.groupby("Day")[worm_col].mean()
                    ax.plot(
                        worm_day_means.index, worm_day_means.values,
                        color=tc_color, linewidth=2, alpha=0.35,
                        marker="o", markersize=4, zorder=1,
                    )

        days_tc = sorted(filtered_df_tc["Day"].unique())
        means_tc, ci_lower_tc, ci_upper_tc = [], [], []

        for day in days_tc:
            day_data = filtered_df_tc[filtered_df_tc["Day"] == day][
                "Mean_KL"
            ].dropna()
            if len(day_data) > 0:
                mean = day_data.mean()
                means_tc.append(mean)
                if len(day_data) > 1:
                    sem = stats.sem(day_data)
                    ci_lower_tc.append(mean - sem)
                    ci_upper_tc.append(mean + sem)
                else:
                    ci_lower_tc.append(mean)
                    ci_upper_tc.append(mean)
            else:
                means_tc.append(np.nan)
                ci_lower_tc.append(np.nan)
                ci_upper_tc.append(np.nan)

        ax.plot(
            days_tc, means_tc, color=tc_color, linewidth=4,
            marker="o", markersize=10, label="TC Troupes", zorder=3,
        )
        ax.fill_between(
            days_tc, ci_lower_tc, ci_upper_tc, color=tc_color, alpha=0.2, zorder=2,
        )

    # --- TP troupes ---
    if tp_troupes:
        filtered_df_tp = worm_kl_df[worm_kl_df["Troupe"].isin(tp_troupes)].copy()
        kl_columns = get_worm_cols(filtered_df_tp)
        filtered_df_tp["Mean_KL"] = filtered_df_tp[kl_columns].mean(axis=1)

        if plot_individual_lines:
            for troupe in tp_troupes:
                troupe_df = filtered_df_tp[filtered_df_tp["Troupe"] == troupe]
                for worm_col in kl_columns:
                    worm_day_means = troupe_df.groupby("Day")[worm_col].mean()
                    ax.plot(
                        worm_day_means.index, worm_day_means.values,
                        color=tp_color, linewidth=2, alpha=0.35,
                        marker="o", markersize=4, zorder=1,
                    )

        days_tp = sorted(filtered_df_tp["Day"].unique())
        means_tp, ci_lower_tp, ci_upper_tp = [], [], []

        for day in days_tp:
            day_data = filtered_df_tp[filtered_df_tp["Day"] == day][
                "Mean_KL"
            ].dropna()
            if len(day_data) > 0:
                mean = day_data.mean()
                means_tp.append(mean)
                if len(day_data) > 1:
                    sem = stats.sem(day_data)
                    ci_lower_tp.append(mean - sem)
                    ci_upper_tp.append(mean + sem)
                else:
                    ci_lower_tp.append(mean)
                    ci_upper_tp.append(mean)
            else:
                means_tp.append(np.nan)
                ci_lower_tp.append(np.nan)
                ci_upper_tp.append(np.nan)

        ax.plot(
            days_tp, means_tp, color=tp_color, linewidth=4,
            marker="o", markersize=10, label="TP Troupes", zorder=3,
        )
        ax.fill_between(
            days_tp, ci_lower_tp, ci_upper_tp, color=tp_color, alpha=0.2, zorder=2,
        )

    # --- Formatting ---
    ax.set_xlabel("Day", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean KL Divergence", fontsize=12, fontweight="bold")
    ax.set_title("KL Divergence Over Days - TC vs TP", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    all_days = sorted(
        set(
            (days_tc if tc_troupes else [])
            + (days_tp if tp_troupes else [])
        )
    )
    ax.set_xticks(all_days)
    ax.legend(loc="best", framealpha=0.9, fontsize=11)
    plt.tight_layout()

    return fig


def plot_individual_trends(df_long):
    """Plot individual subject KL trajectories for TC, TP, and pooled."""
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    troupe_groups = [
        ("TC", df_long["Troupe"].str.contains("TC"), "#1f4788"),
        ("TP", df_long["Troupe"].str.contains("TP"), "#9467bd"),
        (
            "All (Pooled)",
            pd.Series([True] * len(df_long), index=df_long.index),
            "black",
        ),
    ]

    for ax, (label, filter_mask, color) in zip(axes, troupe_groups):
        df_filtered = df_long[filter_mask].copy()

        sns.lineplot(
            data=df_filtered, x="Day", y="KL", hue="Subject_ID",
            marker="o", palette="tab10", alpha=0.5, ax=ax, legend=False,
        )

        sns.regplot(
            data=df_filtered, x="Day", y="KL", scatter=False, color=color,
            label=f"{label} Trend",
            line_kws={"linewidth": 3, "linestyle": "--"}, ax=ax,
        )

        ax.set_title(
            f"{label} - KL Value Progression", fontsize=13, fontweight="bold"
        )
        ax.set_ylabel("KL Value", fontsize=11)
        ax.set_xlabel("Day", fontsize=11)
        ax.set_xticks([1, 2, 3, 4])
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Individual Subject Trends: TC vs TP vs Pooled",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.show()
    return fig


def plot_log_kl_with_qq(df_long):
    """Visualise log-KL divergence separating TC and TP, with Q-Q plots."""
    if not HAS_STATSMODELS:
        print("statsmodels not available — skipping log-KL Q-Q plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    tc_color = "#1f4788"
    tp_color = "#9467bd"

    for idx, (troupe_type, color, troupe_filter) in enumerate([
        ("TC", tc_color, df_long["Troupe"].str.contains("TC")),
        ("TP", tp_color, df_long["Troupe"].str.contains("TP")),
    ]):
        df_filtered = df_long[troupe_filter].copy()
        df_filtered["log_KL"] = np.log(df_filtered["KL"])

        sns.lineplot(
            data=df_filtered, x="Day", y="log_KL", hue="Subject_ID",
            marker="o", palette="viridis", alpha=0.4, ax=axes[0, idx],
        )
        sns.regplot(
            data=df_filtered, x="Day", y="log_KL", scatter=False,
            color=color, line_kws={"linewidth": 3, "ls": "--"},
            ax=axes[0, idx],
        )
        axes[0, idx].set_title(
            f"{troupe_type} Troupes: Upward Trend (Log-Scale)", fontsize=14
        )
        axes[0, idx].set_ylabel("log(KL Divergence)")
        axes[0, idx].set_xticks([1, 2, 3, 4])

        model_log = smf.mixedlm(
            "log_KL ~ Day", df_filtered, groups=df_filtered["Subject_ID"]
        )
        result_log = model_log.fit()
        stats.probplot(result_log.resid, dist="norm", plot=axes[1, idx])
        axes[1, idx].set_title(f"{troupe_type} Troupes: Q-Q Plot", fontsize=14)

    plt.tight_layout()
    plt.show()
    return fig


# ============================================================================
# STEP 4: Statistical analysis functions
# ============================================================================

def run_pooled_lmm(df_long):
    """Fit log-KL ~ Day LMM on pooled data and print summary + Q-Q plot."""
    if not HAS_STATSMODELS:
        print("statsmodels not available — skipping pooled LMM.")
        return

    df_long = df_long.copy()
    df_long["log_KL"] = np.log(df_long["KL"])

    model_log = smf.mixedlm(
        "log_KL ~ Day", df_long, groups=df_long["Subject_ID"]
    )
    result_log = model_log.fit()
    print("--- Log-Transformed Model Summary ---")
    print(result_log.summary())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.lineplot(
        data=df_long, x="Day", y="log_KL", hue="Subject_ID",
        marker="o", palette="viridis", alpha=0.4, ax=axes[0],
    )
    sns.regplot(
        data=df_long, x="Day", y="log_KL", scatter=False, color="black",
        line_kws={"linewidth": 3, "ls": "--"}, ax=axes[0],
    )
    axes[0].set_title("Upward Trend (Log-Scale)", fontsize=14)
    axes[0].set_ylabel("log(KL Divergence)")
    axes[0].set_xticks([1, 2, 3, 4])

    stats.probplot(result_log.resid, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot: Normality of Residuals", fontsize=14)

    plt.tight_layout()
    plt.show()


def run_categorical_day_lmm(df_long):
    """Fit log(KL) ~ C(Day) LMM and compare to null with LRT."""
    if not HAS_STATSMODELS:
        print("statsmodels not available — skipping categorical-Day LMM.")
        return

    model_full = smf.mixedlm(
        "np.log(KL) ~ C(Day)", df_long, groups=df_long["Subject_ID"]
    )
    result_full = model_full.fit(reml=False)
    print(result_full.summary())

    model_null = smf.mixedlm(
        "KL ~ 1", df_long, groups=df_long["Subject_ID"]
    )
    result_null = model_null.fit(reml=False)

    lr_stat = 2 * (result_full.llf - result_null.llf)
    df_diff = len(result_full.params) - len(result_null.params)

    if df_diff > 0:
        p_value = stats.chi2.sf(lr_stat, df_diff)
        print(f"Likelihood Ratio Statistic: {lr_stat:.4f}")
        print(f"Degrees of Freedom: {df_diff}")
        print(f"p-value: {p_value:.4f}")
        if p_value > 0.05:
            print("Day is not a significant predictor of KL")
    else:
        print(
            "Error: Models have the same number of parameters. "
            "Check your formula."
        )


def run_per_troupe_type_lmm(df_long):
    """Fit log(KL) ~ C(Day) LMM separately for TC and TP groups."""
    if not HAS_STATSMODELS:
        print("statsmodels not available — skipping per-troupe LMM.")
        return

    for troupe_type, troupe_filter in [
        ("TC", df_long["Troupe"].str.contains("TC")),
        ("TP", df_long["Troupe"].str.contains("TP")),
    ]:
        print(f"\n{'=' * 60}")
        print(f"ANALYSIS FOR {troupe_type} TROUPES")
        print(f"{'=' * 60}")

        df_filtered = df_long[troupe_filter].copy()

        model_full = smf.mixedlm(
            "np.log(KL) ~ C(Day)", df_filtered,
            groups=df_filtered["Subject_ID"],
        )
        result_full = model_full.fit(reml=False)
        print(result_full.summary())

        model_null = smf.mixedlm(
            "KL ~ 1", df_filtered, groups=df_filtered["Subject_ID"]
        )
        result_null = model_null.fit(reml=False)

        lr_stat = 2 * (result_full.llf - result_null.llf)
        df_diff = len(result_full.params) - len(result_null.params)

        if df_diff > 0:
            p_value = stats.chi2.sf(lr_stat, df_diff)
            print(f"Likelihood Ratio Statistic: {lr_stat:.4f}")
            print(f"Degrees of Freedom: {df_diff}")
            print(f"p-value: {p_value:.4f}")
            if p_value > 0.05:
                print(f"Day is not a significant predictor of KL for {troupe_type}")
            else:
                print(f"Day IS a significant predictor of KL for {troupe_type}")
        else:
            print(
                "Error: Models have the same number of parameters. "
                "Check your formula."
            )


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Run the complete KL divergence analysis and plotting pipeline."""
    # --- Configuration ---
    kl_csv_path = (
        "/n/holylabs/gershman_lab/Users/zkelso/KL_divergence_results/"
        "Tasmanian_Conditioning_KL_Results_COMPILED_2026_03_25_14_36_31.csv"
    )
    log_csv_path = (
        "/n/holylabs/gershman_lab/Users/zkelso/KL_divergence_results/"
        "utils/Planarian_Experiments_Log.csv"
    )
    output_dir = (
        "/n/holylabs/gershman_lab/Users/zkelso/KL_divergence_results/figures/"
    )

    troupe_list = TROUPE_LIST
    save_figures = True
    figure_format = "svg"
    figure_dpi = 300
    plot_null = True
    plot_individual_lines = False

    print("=" * 80)
    print("STARTING KL DIVERGENCE ANALYSIS PIPELINE")
    print("=" * 80)

    # Step 1: Update KL CSV with experimental log information
    print("\n[1/4] Updating KL CSV with experimental metadata...")
    updated_kl_df = update_kl_with_log_info(
        kl_csv_path=kl_csv_path,
        log_csv_path=log_csv_path,
        output_csv_path=None,
    )

    # Step 2: Create worm-specific dataframe
    print("\n[2/4] Creating worm-specific KL dataframe...")
    worm_kl_df = create_worm_kl_dataframe_from_df(updated_kl_df)
    print(f"Created dataframe with {len(worm_kl_df)} videos")
    print(f"Columns: {list(worm_kl_df.columns)}")

    # Step 3: Generate KL divergence plot across days
    fig_onoff = plot_troupe_kl_over_days(
        worm_kl_df, troupe_list, plot_null=plot_null, figsize=(10, 6),
    )

    if plot_individual_lines:
        fig_onoff = plot_troupe_kl_over_days(
            worm_kl_df, troupe_list, plot_null=plot_null,
            figsize=(10, 6), plot_individual_lines=True,
        )

    if save_figures:
        os.makedirs(output_dir, exist_ok=True)
        if plot_individual_lines:
            filename = (
                f"kl_onoff_{'_'.join(troupe_list)}_individual_FINAL"
                f".{figure_format}"
            )
        else:
            filename = (
                f"kl_onoff_{'_'.join(troupe_list)}_FINAL.{figure_format}"
            )
        filepath = os.path.join(output_dir, filename)
        fig_onoff.savefig(filepath, dpi=figure_dpi, bbox_inches="tight")
        print(f"Saved KL divergence across days plot to: {filepath}")

    plt.show()

    # Step 4: Build long-format dataframe and run stats
    print("\n[3/4] Building long-format dataframe for statistics...")
    df_long = build_long_format(updated_kl_df)

    print("\n[4/4] Plotting individual trends...")
    plot_individual_trends(df_long)

    # LMM analyses
    print("\n--- Pooled log-KL LMM ---")
    run_pooled_lmm(df_long)

    print("\n--- Categorical Day LMM (pooled) ---")
    run_categorical_day_lmm(df_long)

    print("\n--- Per-troupe-type LMM ---")
    run_per_troupe_type_lmm(df_long)

    # Log-KL Q-Q visualisation
    print("\n--- Log-KL with Q-Q plots ---")
    plot_log_kl_with_qq(df_long)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nDataframe summary:")
    print(f"  - Total videos: {len(worm_kl_df)}")
    print(f"  - Troupes: {sorted(worm_kl_df['Troupe'].unique())}")
    print(f"  - Days: {sorted(worm_kl_df['Day'].unique())}")
    print(
        f"  - Runs ({len(worm_kl_df['Run'].unique())}): "
        f"{sorted(worm_kl_df['Run'].unique())}"
    )


if __name__ == "__main__":
    main()
