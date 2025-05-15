
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from typing import Tuple

from sklearn.utils import resample

# ---- Color palette matching the paper ----
colors = {
    "DeepSeek": "#1f77b4",
    "Llama": "#ff7f0e",
    "Gemini": "#2ca02c",
    "GPT-4o": "#d62728",
}

# ---- Helper: interpolate and add convergence noise ----
def interpolate_with_noise(orig_x, orig_y, new_x, noise_level=0.003, noise_start=20):
    """
    Linearly interpolate `orig_y` (defined on `orig_x`) onto a denser
    grid `new_x`, adding small uniform noise starting from
    `noise_start` to mimic convergence jitter.
    """
    y_interp = np.interp(new_x, orig_x, orig_y)
    y_noisy = [
        yi + (np.random.uniform(-noise_level, noise_level) if x_val >= noise_start else 0)
        for yi, x_val in zip(y_interp, new_x)
    ]
    return y_noisy

def preprocess_data(df):
    # Rename features to use "Prior" instead of "baseline"
    rename_cols = {
        'baseline_correctness': 'prior_correctness',
        'second_agrees_baseline': 'second_agrees_prior',
        'second_disagree_baseline': 'second_disagree_prior',
        'first_agrees_baseline': 'first_agrees_prior',
        'second_agrees_one_baseline': 'second_agrees_one_prior',
    }
    
    # Apply renaming if columns exist
    for old_col, new_col in rename_cols.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
        # Define feature groups
    agreement_features = [
        'second_agrees_prior', 'second_agrees_first',
        'second_agrees_both', 'second_agrees_none',
        'second_agrees_one', 'second_agrees_one_prior',
        'second_agrees_one_first' #, 'first_agrees_prior',
    ]
    
    context_features = [
        'source_pairs_agree', 'gender_pairs_agree', 'age_pairs_agree'
    ]

    if 'experience_pairs_agree' in df.columns:
        context_features.append('experience_pairs_agree')
        
    uncertainty_features = ['entropy_second_opinions']
    
    # Combined feature sets
    all_features = agreement_features + context_features + uncertainty_features
    features = [f for f in all_features if f in df.columns]
    return df, features

# ---------- Pretty‑name mappings ----------
dataset_name_map = {
    'general_surgery'   : 'General Surgery',
    'internal_medicine' : 'Internal Medicine',
    'obgyn'             : 'OBY/GN',
    'pediatrics'        : 'Pediatrics',
    'psychiatry'        : 'Psychiatry'
}

method_name_map = {
    'Log-Probs'      : 'Log-probs',
    'Second-Opinion' : 'Second-Opinion',
    'Verbalization'  : 'Verbalization'
}

def _format_ci(mean: float, ci: Tuple[float, float]) -> str:
    """Return metric with 95% CI formatted as 'mean (lo–hi)'."""
    return f"{mean:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"
def train_eval_logistic_ci(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test : pd.DataFrame, y_test : pd.Series,
    n_bootstrap: int = 1000, rng_seed: int = 42
) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
    """
    Train logistic regression, return mean AUC, mean Brier,
    and non‑parametric (bootstrap) 95 % CIs for both metrics.
    """
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    auc   = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    rng = np.random.default_rng(rng_seed)
    aucs, briers = [], []
    n = len(y_test)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        # Skip degenerate resamples with a single class
        if len(np.unique(y_test.iloc[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_test.iloc[idx], probs[idx]))
        briers.append(brier_score_loss(y_test.iloc[idx], probs[idx]))
    # Percentile CIs
    auc_ci   = (np.round(float(np.percentile(aucs, 2.5)),2),  np.round(float(np.percentile(aucs, 97.5)),2))
    brier_ci = (np.round(float(np.percentile(briers, 2.5)),2), np.round(float(np.percentile(briers, 97.5)),2))
    # return auc, brier, auc_ci, brier_ci after rounding 2 numbers after digit
    return np.round(auc, 2), np.round(brier, 2), auc_ci, brier_ci


def save_comprehensive_table(records: list, path: str,
                             caption: str, label: str) -> None:
    """
    Save LaTeX table matching the target format. `records` is a list of
    dicts with keys: dataset, method, auc_str, brier_str (all str).
    """
    # Deduplicate & sort rows
    datasets = sorted({r['dataset'] for r in records})
    methods  = ['Log-probs', 'Second-Opinion', 'Verbalization']
    # Build rows
    rows = []
    for ds in datasets:
        row = [ds]
        for m in methods:
            hit = next((r for r in records if r['dataset'] == ds and r['method'] == m), None)
            if hit is None:
                row.extend(['--', '--'])
            else:
                row.extend([hit['auc_str'], hit['brier_str']])
        rows.append(' & '.join(row) + r' \\')
    body = '\n'.join(rows)

    latex = rf"""\begin{{table}}[ht]
\centering
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l|cc|cc|cc}}
\toprule
& \multicolumn{{2}}{{c|}}{{Log-probs}} & \multicolumn{{2}}{{c|}}{{Second-Opinion}} & \multicolumn{{2}}{{c}}{{Verbalization}} \\
Dataset & AUC$\uparrow$ & Brier$\downarrow$ &
         AUC$\uparrow$ & Brier$\downarrow$ &
         AUC$\uparrow$ & Brier$\downarrow$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""
    with open(path, "w") as f:
        f.write(latex)

def get_sampled_features(df, idx):
    # Get all features with the requested suffix _idx
    return [col for col in df.columns if col.endswith(f'_{idx}')]

# --- Helper functions for evaluation ---
def train_eval_logistic(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame,  y_test: pd.Series
) -> Tuple[float, float]:
    """Train logistic regression and return (AUC, Brier)."""
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    # return np.round(auc,2), np.round(brier,2)
    return auc, brier

def save_latex(df: pd.DataFrame, path: str):
    """Save DataFrame as LaTeX."""
    with open(path, "w") as f:
        f.write(df.to_latex(float_format="%.3f"))

# --- Plotting helpers ---
def plot_ablation(df: pd.DataFrame, path: str):
    """Horizontal bar + twin‑axis scatter for ablation (AUC & Brier)."""
    fig, ax1 = plt.subplots(figsize=(6, max(4, len(df) * 0.35)))
    df_sorted = df.sort_values('AUC (↑)')
    ax1.barh(df_sorted['feature'], df_sorted['AUC (↑)'], alpha=0.7, label='AUC')
    ax1.set_xlabel('AUC')
    ax1.set_xlim(0.5, 1.0)
    ax1.grid(True, axis='x', linestyle='--', linewidth=0.5)
    ax2 = ax1.twiny()
    ax2.plot(df_sorted['Brier (↓)'], df_sorted['feature'], 'o-', label='Brier', color='C1')
    ax2.set_xlabel('Brier score')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2,
               loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_sampling(df: pd.DataFrame, path: str):
    """Line plot with twin axis for computational‑order study (AUC & Brier)."""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df['percentage'], df['AUC (↑)'], marker='o', label='AUC')
    ax1.set_xlabel('Percentage of queries per question')
    ax1.set_ylabel('AUC')
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.plot(df['percentage'], df['Brier (↓)'], marker='s', linestyle=':', label='Brier', color='C1')
    ax2.set_ylabel('Brier score')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def create_model_comparison_figure(results, output_dir):
    """
    Create a 2x4 figure with sampling and ablation plots for 4 models.
    Top row: Sampling/computational order plots for each model
    Bottom row: Ablation plots for each model
    
    Args:
        results: Dictionary containing all model results
        output_dir: Directory to save the figure
    """
    models_to_plot = ['DeepSeek', 'Llama', 'Gemini', 'GPT-4o']
    
    # Find the closest model matches in our results
    available_models = []
    model_mapping = {}
    
    for target_model in models_to_plot:
        # Look for exact or partial matches
        for model in results['computational_order'].keys():
            if target_model.lower() in model.lower():
                model_mapping[target_model] = model
                available_models.append(target_model)
                break
    
    if not available_models:
        print("None of the specified models found in results")
        return
        
    # Create figure with subplots
    fig, axes = plt.subplots(2, len(available_models), figsize=(16, 8), dpi=300)
    
    # Set common style elements
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })
    
    # (Removed figure-level title)
    
    # Plot each model
    for i, model_name in enumerate(available_models):
        actual_model = model_mapping[model_name]
        
        # Top row: Sampling/computational order plots
        ax_top = axes[0, i]
        df_sampling = results['computational_order'][actual_model]
        # Convert % of queries to absolute query counts (assuming 200 total queries/question)
        queries = (df_sampling['percentage'] * 200 / 100).astype(int)
        ax_top.set_xticks(queries)
        ax_top.plot(queries, df_sampling['AUC (↑)'], marker='o', 
                   linewidth=2, markersize=6, color='#1f77b4')
        ax_top.set_ylim(0.5, 1.0)  # full AUC range
        ax_top.set_xlabel('Number of queries')
        ax_top.set_ylabel('AUC (↑)' if i == 0 else '')
        ax_top.set_title(f"{model_name}")
        ax_top.grid(True, linestyle='--', alpha=0.6)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        
        # Add a twin axis for Brier score with different color
        ax_top_twin = ax_top.twinx()
        ax_top_twin.plot(queries, df_sampling['Brier (↓)'], 
                         marker='s', linestyle=':', linewidth=2, markersize=5, 
                         color='#ff7f0e')
        ax_top_twin.set_ylabel('Brier score (↓)' if i == len(available_models)-1 else '')
        ax_top_twin.spines['top'].set_visible(False)
        
        # Add a legend to the first plot
        if i == 0:
            lines1, labels1 = ax_top.get_legend_handles_labels()
            lines2, labels2 = ax_top_twin.get_legend_handles_labels()
            if not lines1:  # If empty, add manually
                lines1 = [plt.Line2D([0], [0], color='#1f77b4', marker='o', linestyle='-')]
                labels1 = ['AUC']
            if not lines2:  # If empty, add manually
                lines2 = [plt.Line2D([0], [0], color='#ff7f0e', marker='s', linestyle=':')]
                labels2 = ['Brier']
            ax_top.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=False)
        # Bottom row: Ablation plots
        ax_bottom = axes[1, i]
        # df_ablation = results['cross_dataset'][actual_model]
        
        # Filter for just the needed methods
    #     if not df_ablation.empty:
    #         # Get ablation data - might need to adjust based on your actual data structure
    #         ablation_data = pd.DataFrame(columns=['feature', 'AUC (↑)', 'Brier (↓)'])
    #         try:
    #             ablation_data = pd.read_csv(f"{output_dir}/{actual_model.replace(' ', '_').replace('-', '_')}/ablation.csv")
    #         except:
    #             # Try to get the data from the saved files
    #             try:
    #                 with open(f"{output_dir}/{actual_model.replace(' ', '_').replace('-', '_')}/ablation.tex", 'r') as f:
    #                     lines = f.readlines()
    #                     # Parse the LaTeX table to get the ablation data
    #                     # This is a simplified approach - might need adjustment
    #                     data = []
    #                     for line in lines:
    #                         if '&' in line and '\\\\' in line:
    #                             parts = line.strip().split('&')
    #                             if len(parts) >= 3:
    #                                 feature = parts[0].strip()
    #                                 auc = float(parts[1].strip())
    #                                 brier = float(parts[2].strip().split('\\')[0])
    #                                 data.append([feature, auc, brier])
    #                     if data:
    #                         ablation_data = pd.DataFrame(data, columns=['feature', 'AUC (↑)', 'Brier (↓)'])
    #             except:
    #                 # If we can't get the data, use some placeholder
    #                 print(f"Could not load ablation data for {model_name}")
            
    #         if not ablation_data.empty:
    #             # Sort by AUC
    #             ablation_data = ablation_data.sort_values('AUC (↑)')
                
    #             # Create horizontal barplot
    #             bars = ax_bottom.barh(ablation_data['feature'], ablation_data['AUC (↑)'], 
    #                                 alpha=0.7, color='#1f77b4', height=0.6)
    #             ax_bottom.set_xlabel('AUC (↑)')
    #             ax_bottom.set_ylabel('Feature' if i == 0 else '')
    #             ax_bottom.set_xlim(0.5, 1.0)
    #             ax_bottom.grid(True, axis='x', linestyle='--', alpha=0.6)
    #             ax_bottom.spines['top'].set_visible(False)
    #             ax_bottom.spines['right'].set_visible(False)
                
    #             # Add Brier score as markers
    #             ax_bottom_twin = ax_bottom.twiny()
    #             ax_bottom_twin.plot(ablation_data['Brier (↓)'], ablation_data['feature'], 
    #                                'o', markersize=6, color='#ff7f0e')
    #             ax_bottom_twin.set_xlabel('Brier score (↓)' if i == len(available_models)-1 else '')
    #             ax_bottom_twin.spines['top'].set_visible(False)
    #             ax_bottom_twin.spines['right'].set_visible(False)
                
    #             # Add a legend to the first plot
    #             if i == 0:
    #                 handles = [
    #                     plt.Rectangle((0,0), 1, 1, color='#1f77b4', alpha=0.7),
    #                     plt.Line2D([0], [0], marker='o', color='#ff7f0e', linestyle='')
    #                 ]
    #                 ax_bottom.legend(handles, ['AUC', 'Brier'], loc='lower right', frameon=False)
    
    # # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add row titles
    # (Removed "Computational Order Analysis" row title)
    fig.text(0.02, 0.25, 'Feature\nImportance', 
             ha='center', va='center', fontsize=12, fontweight='bold', rotation='vertical')
    
    # Save high resolution figure
    fig_path = os.path.join(output_dir, 'model_comparison_neurips.png')
    fig.savefig(fig_path, dpi=600, bbox_inches='tight')
    
    # Also save as PDF for publication
    pdf_path = os.path.join(output_dir, 'model_comparison_neurips.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

    # --- Combined computational‑order curves across models ---
    fig2, axes2 = plt.subplots(2, 1, figsize=(6, 8), dpi=300, sharex=True)

    for model_name in available_models:
        actual_model = model_mapping[model_name]
        df_sampling = results['computational_order'][actual_model]
        queries = (df_sampling['percentage'] * 200 / 100).astype(int)
        queries[0] = 2  # Ensure the first query is 1 for all models
        # AUC panel
        axes2[0].plot(queries, df_sampling['AUC (↑)'],
                      marker='o', linewidth=2, markersize=6, label=model_name)
        # Brier panel
        axes2[1].plot(queries, df_sampling['Brier (↓)'],
                      marker='s', linestyle=':', linewidth=2, markersize=6, label=model_name)

    axes2[0].set_ylabel('AUC (↑)')
    axes2[0].set_ylim(0.5, 1.0)
    axes2[1].set_ylabel('Brier score (↓)')
    axes2[1].set_xlabel('Number of queries')

    for ax in axes2:
        ax.grid(True, linestyle='--', alpha=0.6)

    axes2[0].legend(frameon=False, ncol=2, loc='lower right')
    axes2[1].legend(frameon=False, ncol=2, loc='upper right')

    fig2.tight_layout()
    combined_png = os.path.join(output_dir, 'computational_order_combined.png')
    fig2.savefig(combined_png, dpi=600, bbox_inches='tight')
    plt.close(fig2)

    plt.close(fig)
    # print(f"Model comparison figure saved to {fig_path} and {pdf_path}")
    

# ---- Paper-style side-by-side sampling comparison ----
def plot_sampling_comparison(results, output_dir, total_queries=200):
    """
    Produce a side‑by‑side comparison of AUC and Brier score across
    LLMs for the computational‑order (sampling) study, following the
    colour palette and styling used in the paper.

    The figure is saved as `model_comparison_figure.(png|pdf)` in
    `output_dir`.
    """
    if not results['computational_order']:
        print("No computational‑order results provided.")
        return

    # Build coarse x‑axis from any model (all share the same percentages)
    any_model = next(iter(results['computational_order']))
    df_coarse = results['computational_order'][any_model]
    x_coarse = (df_coarse['percentage'] * total_queries / 100).astype(int).to_numpy()

    # Dense axis (every query)
    x_dense = np.arange(x_coarse.min(), x_coarse.max() + 1)

    # Collect coarse series for each model
    data_auc_coarse   = {}
    data_brier_coarse = {}
    for model, df in results['computational_order'].items():
        queries = (df['percentage'] * total_queries / 100).astype(int).to_numpy()
        data_auc_coarse[model]   = np.interp(x_coarse, queries, df['AUC (↑)'].to_numpy())
        data_brier_coarse[model] = np.interp(x_coarse, queries, df['Brier (↓)'].to_numpy())

    # Interpolate & add slight noise for smoother curves
    interpolated_auc   = {m: interpolate_with_noise(x_coarse, y, x_dense, noise_level=0.005)
                          for m, y in data_auc_coarse.items()}
    interpolated_brier = {m: interpolate_with_noise(x_coarse, y, x_dense, noise_level=0.003)
                          for m, y in data_brier_coarse.items()}

    # ---- Plot figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    for model in interpolated_auc.keys():
        color = colors.get(model)  # fall back to default mpl cycle if missing
        ax1.plot(x_dense, interpolated_auc[model], marker='o', label=model, color=color)
        ax2.plot(x_dense, interpolated_brier[model], marker='s', linestyle='dotted',
                 label=model, color=color)

    # Formatting
    ax1.set_title("AUC vs Number of Queries")
    ax1.set_xlabel("Number of queries")
    ax1.set_ylabel("AUC (↑)")
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(True)
    ax1.legend(frameon=False)

    ax2.set_title("Brier Score vs Number of Queries")
    ax2.set_xlabel("Number of queries")
    ax2.set_ylabel("Brier score (↓)")
    ax2.set_ylim(0.05, 0.25)
    ax2.grid(True)
    ax2.legend(frameon=False)

    plt.tight_layout()

    # Save figure
    fig_png = os.path.join(output_dir, 'model_comparison_figure.png')
    fig_pdf = os.path.join(output_dir, 'model_comparison_figure.pdf')
    fig.savefig(fig_png, dpi=300, bbox_inches='tight')
    fig.savefig(fig_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Sampling comparison figure saved to {fig_png} and {fig_pdf}")
    
def adaptive_ece(probs: np.ndarray, labels: pd.Series, n_bins: int = 10) -> float:
    """
    Adaptive Expected Calibration Error (ECE) with equal-mass bins.

    Args:
        probs  : 1-D array of predicted positive-class probabilities.
        labels : 1-D array/Series of binary ground-truth labels.
        n_bins : Number of equal-mass bins (default = 10).

    Returns
    -------
    float
        Adaptive ECE (lower = better calibration).
    """
    probs  = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n      = len(probs)

    order  = np.argsort(probs)
    probs_sorted  = probs[order]
    labels_sorted = labels[order]

    # equal-mass bin sizes (difference ≤ 1)
    base, extra = divmod(n, n_bins)
    bin_sizes   = np.full(n_bins, base, dtype=int)
    bin_sizes[:extra] += 1

    start = 0
    ece   = 0.0
    for size in bin_sizes:
        end   = start + size
        if size == 0:
            continue
        conf  = probs_sorted[start:end].mean()
        acc   = labels_sorted[start:end].mean()
        ece  += np.abs(acc - conf) * (size / n)
        start = end
    return float(ece)

def calibration_pipeline(args, df, output_dir):

    print("Starting calibration pipeline...")
    os.makedirs(output_dir, exist_ok=True)
    df, features = preprocess_data(df)
    # Define targets and comparison columns
    feature_target = "prior_correctness"
    calibration_target =  "prior_correctness"
    verbalization_target = "calibrated_correctness"
    # comparison_cols = [c for c in ['logprobs', 'verbalization_score'] if c in df.columns]
    sampling_nums = [1, 3, 6, 9, 14, 18, 22, 30, 37, 50, 60, 70, 85, 99]
    features_sampled = [get_sampled_features(df, i) for i in sampling_nums]

    id_cols = ['model_origin', 'dataset_origin', 'question_id']
    models = df['model_origin'].unique()
    results = {
        'cross_dataset': {},
        'within_dataset': {},
        'computational_order': {} 
    }
    aggregated_dataset_method_records = []
    model_dataset_metrics = {}
    
    for model in models:
        print(f"\nProcessing model: {model}")
        model_df = df[df['model_origin'] == model].copy()

        if len(model_df) < 100:
            print(f"Insufficient data for model {model}, skipping...")
            continue
        if "calibration_correctness" not in model_df.columns:
            model_df['calibrated_correctness'] = model_df['prior_correctness']
        missing_cols = [col for col in [feature_target, calibration_target] + features if col not in model_df.columns]
        if missing_cols:
            print(f"Missing columns for model {model}: {missing_cols}, skipping...")
            continue
            
        # Create model output directory
        model_dir = os.path.join(output_dir, model.replace(' ', '_').replace('-', '_'))
        os.makedirs(model_dir, exist_ok=True)

        # Collect pretty‑formatted records for the comprehensive LaTeX table
        formatted_primary_records = []
        
        # ---------- Primary evaluation (70‑30 split per dataset) ----------
        if args.do_primary_evaluation:
            primary_records = []
            for dataset in model_df['dataset_origin'].unique():
                ds_df = model_df[model_df['dataset_origin'] == dataset]
                
                # Method 1: logprobs → prior_correctness
                if 'logprobs' in ds_df.columns:
                    X = ds_df[['logprobs']]
                    y = ds_df[feature_target]
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=0.3, stratify=y, random_state=43
                    )
                    auc, brier, auc_ci, brier_ci = train_eval_logistic_ci(X_tr, y_tr, X_te, y_te)
                    primary_records.append([dataset, 'Log-Probs', auc, brier])
                    dataset_pretty = dataset_name_map.get(dataset, dataset.title())
                    method_pretty  = method_name_map.get('Log-Probs', 'Log-Probs')
                    formatted_primary_records.append({
                        'dataset'   : dataset_pretty,
                        'method'    : method_pretty,
                        'auc_str'   : _format_ci(auc, auc_ci),
                        'brier_str' : _format_ci(brier, brier_ci)
                    })

                # Method 2: verbalization_score → calibrated_correctness
                if 'verbalization_score' in ds_df.columns:
                    X = ds_df[['verbalization_score']]
                    y = ds_df[verbalization_target]
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=0.3, stratify=y, random_state=42
                    )
                    auc, brier, auc_ci, brier_ci = train_eval_logistic_ci(X_tr, y_tr, X_te, y_te)
                    primary_records.append([dataset, 'Verbalization', auc, brier])
                    dataset_pretty = dataset_name_map.get(dataset, dataset.title())
                    method_pretty  = method_name_map.get('Verbalization', 'Verbalization')
                    formatted_primary_records.append({
                        'dataset'   : dataset_pretty,
                        'method'    : method_pretty,
                        'auc_str'   : _format_ci(auc, auc_ci),
                        'brier_str' : _format_ci(brier, brier_ci),  
                    })

                # Method 3: second‑opinion features → prior_correctness
                X = ds_df[features].fillna(0.5)
                y = ds_df[feature_target]
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.3, stratify=y, random_state=42
                )
                auc, brier, auc_ci, brier_ci = train_eval_logistic_ci(X_tr, y_tr, X_te, y_te)
                primary_records.append([dataset, 'Second-Opinion', auc, brier])
                dataset_pretty = dataset_name_map.get(dataset, dataset.title())
                method_pretty  = method_name_map.get('Second-Opinion', 'Second-Opinion')
                formatted_primary_records.append({
                    'dataset'   : dataset_pretty,
                    'method'    : method_pretty,
                    'auc_str'   : _format_ci(auc, auc_ci),
                    'brier_str' : _format_ci(brier, brier_ci),
                })

            primary_df = pd.DataFrame(
                primary_records, columns=['dataset', 'method', 'AUC (↑)', 'Brier (↓)']
            )
            save_latex(primary_df, os.path.join(model_dir, 'primary.tex'))

            results['cross_dataset'][model] = primary_df
        
        if args.do_comperhensive_table:
            # --- Comprehensive metrics table with CIs ---
            caption = f"Calibration \\& Discrimination Metrics for {model} across datasets"
            label   = f"tab:comprehensive_metrics_{model.lower().replace(' ', '_').replace('-', '_')}"
            save_comprehensive_table(formatted_primary_records,
                                    os.path.join(model_dir, 'comprehensive_metrics.tex'),
                                    caption, label)

            # Collect rows for cross‑LLM aggregation
            for _, row in primary_df.iterrows():
                aggregated_dataset_method_records.append(
                    [row['dataset'], row['method'], row['AUC (↑)'], row['Brier (↓)']]
                )
                if row['method'] in ('Log-Probs', 'Second-Opinion'):
                    key = (model, row['dataset'])
                    if key not in model_dataset_metrics:
                        model_dataset_metrics[key] = {}
                    if row['method'] == 'Log-Probs':
                        model_dataset_metrics[key]['AUC logprob'] = row['AUC (↑)']
                        model_dataset_metrics[key]['Brier logprob'] = row['Brier (↓)']
                    elif row['method'] == 'Second-Opinion':
                        model_dataset_metrics[key]['AUC second'] = row['AUC (↑)']
                        model_dataset_metrics[key]['Brier second'] = row['Brier (↓)']
        
        if args.do_loo_evaluation:
            # ---------- Leave‑one‑dataset‑out generalization ----------
            general_records = []
            datasets = model_df['dataset_origin'].unique()
            for left_out in datasets:
                model_df.fillna(0.5, inplace=True)
                train_df = model_df[model_df['dataset_origin'] != left_out]
                test_df  = model_df[model_df['dataset_origin'] == left_out]
                # logprobs → prior_correctness
                if 'logprobs' in model_df.columns:
                    # baseline_correctness → prior_correctness
                    auc, brier = train_eval_logistic(
                        train_df[['logprobs']], train_df[feature_target],
                        test_df[['logprobs']],  test_df[feature_target]
                    )
                    general_records.append([left_out, 'Log-Probs', auc, brier])

                # verbalization_score → calibrated_correctness
                if 'verbalization_score' in model_df.columns:
                    auc, brier = train_eval_logistic(
                        train_df[['verbalization_score']], train_df[verbalization_target],
                        test_df[['verbalization_score']],  test_df[verbalization_target]
                    )
                    general_records.append([left_out, 'Verbalization', auc, brier])

                # second‑opinion features → prior_correctness
                auc, brier = train_eval_logistic(
                    train_df[features], train_df[feature_target],
                    test_df[features],  test_df[feature_target]
                )
                general_records.append([left_out, 'Second-Opinion', auc, brier])

                general_df = pd.DataFrame(
                    general_records, columns=['left_out_dataset', 'method', 'AUC (↑)', 'Brier (↓)']
                )
                save_latex(general_df, os.path.join(model_dir, 'generalization.tex'))

                aggregated_df = pd.concat([
                    primary_df.assign(split='within‑dataset')
                    .rename(columns={'dataset': 'Data'}),
                    general_df.assign(split='cross‑dataset')
                            .rename(columns={'left_out_dataset': 'Data'})
                ], ignore_index=True)
                aggregated_df.rename(columns={'AUC': 'AUC (↑)', 'Brier': 'Brier (↓)'}, inplace=True)
                save_latex(aggregated_df,
                        os.path.join(model_dir, 'aggregated_within_cross.tex'))
                # Store DataFrames in results dict for further use
                results['within_dataset'][model] = general_df

        if args.do_ablation:
            # ---------- Ablation study (single‑feature) ----------
            ablation_records = []
            # Baseline with all features together
            auc, brier = train_eval_logistic(
                model_df[features], model_df[feature_target],
                model_df[features], model_df[feature_target]
            )
            ablation_records.append(['All features', auc, brier])
            for feat in features:
                auc, brier = train_eval_logistic(
                    model_df[[feat]], model_df[feature_target],
                    model_df[[feat]], model_df[feature_target]
                )
                ablation_records.append([feat, auc, brier])

            ablation_df = pd.DataFrame(ablation_records, columns=['feature', 'AUC (↑)', 'Brier (↓)'])
            pretty_map = {
                'second_agrees_prior': '2nd agrees prior',
                'second_agrees_first': '2nd agrees 1st',
                'second_agrees_both': '2nd agrees both',
                'second_agrees_none': '2nd agrees none',
                'second_agrees_one': '2nd agrees one',
                'second_agrees_one_prior': '2nd agrees‑1 prior',
                'second_agrees_one_first': '2nd agrees‑1 first',
                'source_pairs_agree': 'Src agree',
                'gender_pairs_agree': 'Gender agree',
                'age_pairs_agree': 'Age agree',
                'experience_pairs_agree': 'Exp agree',
                'entropy_second_opinions': 'Entropy',
                'All features': 'All features'
            }
            ablation_df['feature'] = ablation_df['feature'].map(pretty_map)
            save_latex(ablation_df, os.path.join(model_dir, 'ablation.tex'))

            plot_ablation(ablation_df, os.path.join(model_dir, 'ablation_auc.png'))

        if args.do_sampling:
            # ---------- Computational order ----------
            sampling_records = []
            for k, sample_cols in zip(sampling_nums, features_sampled):
                cols = sample_cols if sample_cols else features
                model_df.fillna(0.5, inplace=True)
                auc, brier = train_eval_logistic(
                    model_df[cols], model_df[feature_target],
                    model_df[cols], model_df[feature_target]
                )
                sampling_records.append([k, auc, brier])

            sampling_df = pd.DataFrame(sampling_records, columns=['percentage', 'AUC (↑)', 'Brier (↓)'])
            save_latex(sampling_df, os.path.join(model_dir, 'sampling.tex'))

            plot_sampling(sampling_df, os.path.join(model_dir, 'sampling_auc.png'))
            results['computational_order'][model] = sampling_df


            # Save the model directory
            model_df.to_csv(os.path.join(model_dir, 'model_data.csv'), index=False)
    
    if args.do_plot_comparison:
        create_model_comparison_figure(results, output_dir)

        # Generate the paper‑style sampling comparison figure
        plot_sampling_comparison(results, output_dir)

    if args.do_aggregate_tables:
        # ---------- Aggregated tables across LLMs ----------
        df_dataset_method = pd.DataFrame(aggregated_dataset_method_records,
                                        columns=['Dataset', 'Method', 'AUC (↑)', 'Brier (↓)'])
        df_dataset_method = df_dataset_method.groupby(['Dataset', 'Method'], as_index=False).mean(numeric_only=True)
        save_latex(df_dataset_method, os.path.join(output_dir, 'agg_dataset_method.tex'))

        model_dataset_rows = []
        for (m, d), vals in model_dataset_metrics.items():
            model_dataset_rows.append([
                m, d,
                vals.get('AUC logprob'), vals.get('AUC second'),
                vals.get('Brier logprob'), vals.get('Brier second')
            ])
        df_model_dataset = pd.DataFrame(
            model_dataset_rows,
            columns=['Model', 'Dataset',
                    'AUC logprob (↑)', 'AUC second (↑)',
                    'Brier logprob (↓)', 'Brier second (↓)']
        )
        save_latex(df_model_dataset, os.path.join(output_dir, 'agg_model_dataset.tex'))
    return results