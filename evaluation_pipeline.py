import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# ---- util imports (assumed available) ----------------------------------------
from utils import (
    from_csvs_to_dfs,
    calculate_performance_metrics,
    calculate_gee_models,
    from_dfs_to_analyzed_dfs,
    from_cal_csv_to_dfs,
    parse_score_string,
)
from classification_pipeline import calibration_pipeline
# -----------------------------------------------------------------------------

class CalibrationArgs:
    """Container mirroring the flags expected by `classification.calibration_pipeline`."""

    def __init__(
        self,
        do_primary_evaluation=True,
        do_comperhensive_table=True,
        do_loo_evaluation=True,
        do_ablation=True,
        do_sampling=True,
        do_plot_comparison=True,
        do_aggregate_tables=True,
    ):
        self.do_primary_evaluation = do_primary_evaluation
        self.do_comperhensive_table = do_comperhensive_table
        self.do_loo_evaluation = do_loo_evaluation
        self.do_ablation = do_ablation
        self.do_sampling = do_sampling
        self.do_plot_comparison = do_plot_comparison
        self.do_aggregate_tables = do_aggregate_tables


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def guess_dataset(csv_paths):
    """Return the dataset tag (`mmlu` or `nejm`) inferred from the first path."""
    return "mmlu" if "mmlu" in os.path.basename(csv_paths[0]).lower() else "nejm"


def load_or_process_raw(csv_paths, processed_dir, dataset):
    """Load raw_df from cache or process CSVs -> dfs and save cache."""

    raw_cache = os.path.join(processed_dir, f"{dataset}_raw_df.csv")

    if os.path.exists(raw_cache):
        print(f"[cache] Loading raw_df from {raw_cache}.")
        raw_df = pd.read_csv(raw_cache)
    else:
        if not csv_paths:
            return None
        else:
            print("[process] Converting source CSVs to DataFrame …")
            raw_df = from_csvs_to_dfs(csv_paths)
            os.makedirs(processed_dir, exist_ok=True)
            raw_df.to_csv(raw_cache, index=False)
            print(f"[cache] Saved raw_df to {raw_cache}.")
    return raw_df


def load_or_process_analyzed(raw_df, processed_dir, dataset, calibration_csv_paths):
    """Load analyzed_df from cache or compute (+ optional calibration merge) and save."""

    analyzed_cache = os.path.join(processed_dir, f"{dataset}_analyzed_df.csv")

    if os.path.exists(analyzed_cache):
        print(f"[cache] Loading analyzed_df from {analyzed_cache}.")
        analyzed_df = pd.read_csv(analyzed_cache)
        return analyzed_df

    print("[process] Building analyzed_df …")

    if dataset == "nejm":
        analyzed_df = from_dfs_to_analyzed_dfs(raw_df, mode="nejm", output_dir=processed_dir, num_bins=100)

        # "weird script"— replicate ad‑hoc score parsing
        raw_df = raw_df.copy()
        raw_df["logprobs"] = 0.0
        for model in ["DeepSeek R1 8b", "Llama 3.1 8B"]:
            mask = raw_df["model_origin"] == model
            raw_df.loc[mask, "logprobs"] = raw_df.loc[mask, "baseline_opinion_probs"].apply(parse_score_string)
        mask = raw_df["model_origin"] == "GPT-4o"
        raw_df.loc[mask, "logprobs"] = raw_df.loc[mask, "baseline_opinion_probs"].astype(float).pipe(lambda s: s.apply(lambda x: np.exp(x)))

        analyzed_df = analyzed_df.loc[:, ~analyzed_df.columns.duplicated()]
        analyzed_df = analyzed_df.merge(
            raw_df.groupby(["model_origin", "dataset_origin", "question_id"])["logprobs"].mean().reset_index(),
            on=["model_origin", "dataset_origin", "question_id"],
            how="left",
        )

        # Merge optional calibration csvs
        if calibration_csv_paths:
            analyzed_df = from_cal_csv_to_dfs(calibration_csv_paths, analyzed_df)

    else:
        # mmlu
        analyzed_df = from_dfs_to_analyzed_dfs(raw_df)
        analyzed_df = analyzed_df.loc[:, ~analyzed_df.columns.duplicated()]
        analyzed_df = analyzed_df.merge(
            raw_df.groupby(["model_origin", "dataset_origin", "question_id"])["baseline_logprob"].mean().reset_index(),
            on=["model_origin", "dataset_origin", "question_id"],
            how="left",
        )
        analyzed_df = analyzed_df.merge(
            raw_df.groupby(["model_origin", "dataset_origin", "question_id"])["baseline_calibration"].mean().reset_index(),
            on=["model_origin", "dataset_origin", "question_id"],
            how="left",
        )
        analyzed_df.rename(
            columns={"baseline_logprob": "logprobs", "baseline_calibration": "verbalization_score"}, inplace=True
        )

    analyzed_df["verbalization_score"].fillna(0.5, inplace=True)
    # Enrich analyzed_df with question text (needed downstream) ---------------
    analyzed_df = analyzed_df.merge(
        raw_df.groupby(["model_origin", "dataset_origin", "question_id"])["question"].first().reset_index(),
        on=["model_origin", "dataset_origin", "question_id"],
        how="left",
    )

    os.makedirs(processed_dir, exist_ok=True)
    analyzed_df.to_csv(analyzed_cache, index=False)
    print(f"[cache] Saved analyzed_df to {analyzed_cache}.")
    return analyzed_df

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_pipeline(args):
    if args.csv_paths:
        dataset = guess_dataset(args.csv_paths)
    else:
        dataset = args.dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare directories ------------------------------------------------------
    out_root = os.path.abspath(args.output_dir)
    processed_dir = os.path.join(out_root, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    out_dataset_dir = os.path.join(out_root, dataset)
    os.makedirs(out_dataset_dir, exist_ok=True)

    # Step 0: raw & analyzed loading / processing -----------------------------
    raw_df = load_or_process_raw(args.csv_paths, processed_dir, dataset)
    analyzed_df = load_or_process_analyzed(raw_df, processed_dir, dataset, args.calibration_csvs)

    # remove columns from analyzed_df using array
    columns_to_remove = [
        "question_id",
        "dataset_origin"
    ]
    analyzed_df = analyzed_df.loc[:, ~analyzed_df.columns.isin(columns_to_remove)]

    if raw_df:
        # Step 1: performance metrics ---------------------------------------------
        performance_table = calculate_performance_metrics(raw_df, out_dataset_dir)

        # Step 2: GEE models -------------------------------------------------------
        gee_result = calculate_gee_models(raw_df, out_dataset_dir)

    # Step 3: calibration pipeline -------------------------------------------
    CALIBRATION_CONFIG = CalibrationArgs(
        do_primary_evaluation=False,
        do_comperhensive_table=False,
        do_loo_evaluation=False,
        do_ablation=False,
        do_sampling=True,
        do_plot_comparison=True,
        do_aggregate_tables=False,
    )
    res = calibration_pipeline(CALIBRATION_CONFIG, analyzed_df, os.path.join(out_dataset_dir, "calibration_prediction"))

    print("[done] Evaluation pipeline finished.")

# -----------------------------------------------------------------------------
# CLI wrappers
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Unified evaluation pipeline")
    p.add_argument(
        "csv_paths", nargs="*", help="Paths to the model output CSV files to evaluate.", default=[])
    p.add_argument(
        "--calibration-csvs",
        dest="calibration_csvs",
        nargs="*",
        default=[],
        help="Optional calibration csvs (only used for NEJM).",
    )
    p.add_argument(
        "--dataset",
        default="nejm",
        choices=["mmlu", "nejm"],
        help="Dataset to evaluate (default: mmlu).",
    )
    p.add_argument(
        "--output-dir",
        default= "/home/nimrodharel/Projects/second_opinions/second_opinion_git/", #"evaluation",
        help="Root directory where results & caches will be stored.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
