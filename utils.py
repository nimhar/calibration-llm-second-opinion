import os
import pandas as pd
import numpy as np    
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Independence, Exchangeable, Autoregressive
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

import re


def parse_score_string(s, mode='value'):
    # Replace unquoted -inf with float('-inf') safely
    s_clean = re.sub(r'(?<!["\'])-inf(?!["\'])', 'float("-inf")', s)
    try:
        # Eval with safe builtins (only float)
        result = eval(s_clean, {"__builtins__": None}, {"float": float})
        max_key= max(result, key=result.get)
        if mode=='key':
            return max_key.strip()
        else:
            return np.exp(float(result[max_key]))
        # return max_key, max_value
    except Exception as e:
        return 0
    
def calculate_gee_models(all_dfs, output_dir): 
    """Improved GEE model calculation with enhanced visualization"""
    output_dir = os.path.join(output_dir, 'gee_models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Rename columns to follow the requested naming convention
    column_mapping = {
        'baseline_correctness': 'prior_correctness',
        'first_agree_baseline': 'first_agree_prior',
        'second_agree_baseline': 'second_agree_prior'
    }
    
    # Rename columns if they exist
    for old_col, new_col in column_mapping.items():
        if old_col in all_dfs.columns:
            all_dfs[new_col] = all_dfs[old_col]
    
    # Define bias features with improved naming
    bias_features = ['age', 'source', 'gender']
    if 'experience' in all_dfs.columns:
        bias_features += ['experience']
    
    # Define custom features with the new naming
    custom_features = ['second_agree_first', 'prior_correctness'] + bias_features 
    
    # Create unique identifier for clustering
    all_dfs['question_id_and_first_id'] = all_dfs['dataset_origin'] + '_' + all_dfs['model_origin'] + '_' + all_dfs['question_id'].astype(str) + '_' + all_dfs['first_opinion_letter'] 
    
    # Run the GEE model with improved visualization
    build_gee_model(
        df=all_dfs, 
        output_dir=output_dir,
        features=custom_features,
        grouping_variable='question_id_and_first_id',
        target_variable='second_agree_prior',
    )

def build_gee_model(df, output_dir=None, features=None, grouping_variable='question_id_and_first_id', 
                   target_variable='second_opinion_correctness', formula=None):
    """
    Build Generalized Estimating Equation model for each combination of model and dataset.
    
    Args:
        df: Combined dataframe with all models and datasets
        output_dir: Directory to save results
        features: List of feature columns to include in the model (if None, uses default features)
        grouping_variable: Column to use for clustering related observations
        target_variable: Target variable to predict
        formula: Custom formula string (if provided, overrides features parameter)
    
    Returns:
        Dictionary containing GEE model results for each model-dataset combination
    """
    
    print("\nBuilding GEE models for each model and dataset combination...")
    
    if output_dir:
        gee_dir = os.path.join(output_dir, "gee_models")
        os.makedirs(gee_dir, exist_ok=True)
    
    # Results container
    gee_results = {}
    
    # Get unique models and datasets
    models = df['model_origin'].unique()
    datasets = df['dataset_origin'].unique()
    
    # Ensure target variable exists in df (add if needed)
    if target_variable not in df.columns:
        if target_variable == 'second_opinion_correctness':
            df['second_opinion_correctness'] = df['second_opinion_letter'] == df['correct_letter']
        else:
            raise ValueError(f"Target variable '{target_variable}' not found in dataframe")
    
    # Default features if None provided
    if features is None and formula is None:
        features = ['first_agree_prior', 'second_agree_first', 'second_agree_prior']
    
    # Rename features to more presentable versions for display
    feature_display_map = {
        'first_agree_baseline': 'First Agree Prior',
        'second_agree_first': 'Second Agree First',
        'second_agree_baseline': 'Second Agree Prior',
        'baseline_correctness': 'Prior Correctness',
        'first_agree_prior': 'First Agree Prior',
        'second_agree_prior': 'Second Agree Prior'
    }
    
    # Construct formula if not provided
    if formula is None:
        # Verify all features exist in the dataframe
        missing_features = [feat for feat in features if feat not in df.columns]
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in dataframe")
        
        formula = f"{target_variable} ~ " + " + ".join(features)
    
    # Verify grouping variable exists
    if grouping_variable not in df.columns:
        raise ValueError(f"Grouping variable '{grouping_variable}' not found in dataframe")
    
    # Check if results file "gee_all_results.csv" already exist
    if output_dir and os.path.exists(os.path.join(gee_dir, "gee_all_results.csv")):
        print("GEE results file already exists. Loading existing results...")
        # Load existing results
        all_results = pd.read_csv(os.path.join(gee_dir, "gee_all_results.csv"))
        all_results_list = all_results.to_dict('records')
    else:
        all_results_list = []
        # Run GEE for each model-dataset combination
        for model in models:
            gee_results[model] = {}
            
            for dataset in datasets:
                print(f"  Processing GEE for model: {model}, dataset: {dataset}")
                
                # Filter data for this model and dataset
                subset = df[(df['model_origin'] == model) & (df['dataset_origin'] == dataset)]
                
                # Skip if not enough data
                if len(subset) < 30:
                    print(f"    Skipping due to insufficient data: {len(subset)} examples")
                    continue
                
                # Check for missing values in features or target
                all_columns = features + [target_variable, grouping_variable]
                if subset[all_columns].isna().any().any():
                    # Detailed report of missing values by column
                    missing_counts = subset[all_columns].isna().sum()
                    missing_columns = missing_counts[missing_counts > 0]
                    missing_percent = (missing_counts / len(subset) * 100)[missing_counts > 0]
                    
                    print(f"    Warning: Missing values detected in the following columns:")
                    for col, count, percent in zip(missing_columns.index, missing_columns, missing_percent):
                        print(f"      - {col}: {count} missing values ({percent:.1f}%)")
                    
                    # Drop rows with NaNs
                    before_count = len(subset)
                    # subset = subset.dropna(subset=all_columns)
                    # Identify feature types
                    numeric_feats = [c for c in features if pd.api.types.is_numeric_dtype(subset[c])]
                    categorical_feats = [c for c in features if c not in numeric_feats]

                    # Impute numeric: column mean
                    for col in numeric_feats:
                        mean_val = subset[col].mean()
                        subset[col] = subset[col].fillna(mean_val)

                    # Impute categorical: new category "Na"
                    for col in categorical_feats:
                        subset[col] = subset[col].astype("object").fillna("Na")

                    # Finally, ensure target and grouping have no NaNs
                    subset[target_variable] = subset[target_variable].fillna(0)  # or another rule
                    subset[grouping_variable] = subset[grouping_variable].fillna("missing_group")
                    after_count = len(subset)
                    print(f"    Dropped {before_count - after_count} rows with NaN values ({(before_count - after_count) / before_count:.1%} of data)")
                    
                    if len(subset) < 30:
                        print(f"    Skipping after NaN removal: insufficient data ({len(subset)} examples)")
                        continue

                # Check for multicollinearity - Fixed implementation for categorical variables
                X = subset[features]
                if len(features) > 1:  # Only check if we have multiple features
                    try:
                        # First check for numeric-only variables to avoid errors with categorical vars
                        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
                        
                        if len(numeric_features) > 1:  # Need at least 2 numeric features for correlation
                            # Check correlations between numeric features
                            corr_matrix = X[numeric_features].corr().abs()
                            high_corr_pairs = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    if corr_matrix.iloc[i, j] > 0.8:
                                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                                        high_corr_pairs.append(f"{col1}-{col2} ({corr_matrix.iloc[i, j]:.2f})")
                                        
                            if high_corr_pairs:
                                print(f"    Warning: High correlations detected between: {', '.join(high_corr_pairs)}")
                            
                            # Only compute VIF for numeric features
                            if len(numeric_features) > 1:  # VIF requires at least 2 features
                                X_numeric = X[numeric_features]
                                X_with_const = sm.add_constant(X_numeric)
                                vif_data = pd.DataFrame()
                                vif_data["Variable"] = X_with_const.columns
                                vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                                                   for i in range(X_with_const.shape[1])]
                                high_vif = vif_data[vif_data["VIF"] > 5]
                                if not high_vif.empty:
                                    print(f"    Warning: High multicollinearity detected for variables:")
                                    for var, vif in zip(high_vif["Variable"], high_vif["VIF"]):
                                        print(f"      - {var}: VIF = {vif:.2f}")
                        
                        # For categorical variables, use different approach - check associations
                        cat_features = [f for f in features if f not in numeric_features]
                        if len(cat_features) > 1:
                            print(f"    Note: Categorical variables {cat_features} - checking associations instead of VIF")
                            for i, cat1 in enumerate(cat_features):
                                for cat2 in cat_features[i+1:]:
                                    # Chi-square test could be used here for a more precise check
                                    # For now just report the categorical pairs
                                    print(f"      Checking association between categorical variables: {cat1} and {cat2}")
                    
                    except Exception as e:
                        print(f"    Warning: Could not compute multicollinearity metrics: {str(e)}")
                        # Print more debug info to understand what's happening
                        print(f"    Feature types: {X.dtypes}")
                
                try:
                    # Create and fit the GEE model
                    gee_model = smf.gee(
                        formula=formula,
                        data=subset,
                        groups=subset[grouping_variable],
                        family=Binomial(),
                        cov_struct=Independence()
                    )
                    
                    # Fit the model
                    result = gee_model.fit()
                    
                    # Store the result
                    gee_results[model][dataset] = result
                    
                    # Extract key statistics
                    params = result.params.to_dict()
                    pvalues = result.pvalues.to_dict()
                    conf_int = result.conf_int()
                    std_errors = result.bse.to_dict()
                    
                    # Add results to summary dataframe
                    for variable in params.keys():
                        all_results_list.append({
                            'model': model,
                            'dataset': dataset,
                            'variable': variable,
                            'coefficient': params[variable],
                            'p_value': pvalues[variable],
                            'lower_ci': conf_int.loc[variable, 0],
                            'upper_ci': conf_int.loc[variable, 1],
                            'std_err': std_errors[variable],
                            'significant': pvalues[variable] < 0.05
                        })
                    
                    # Save summary to file if output_dir provided
                    if output_dir:
                        with open(os.path.join(gee_dir, f"gee_{model}_{dataset}.txt"), 'w') as f:
                            f.write(result.summary().as_text())
                    
                    print(f"    GEE model successfully built")
                    
                except Exception as e:
                    print(f"    Error building GEE model for {model}, {dataset}: {str(e)}")
    
    # Create a combined results dataframe
    if all_results_list:
        results_df = pd.DataFrame(all_results_list)
        
        # Save combined results if output_dir provided
        if output_dir:
            results_df.to_csv(os.path.join(gee_dir, "gee_all_results.csv"), index=False)
            
            # Create summary statistics table
            create_summary_statistics_table(results_df, gee_dir, target_variable)
    
    return gee_results

def create_summary_statistics_table(results_df, output_dir, target_variable):
    """
    Create summary statistics table highlighting significant effects across models
    """
    # Group by variable and model
    summary = results_df.groupby(['variable', 'model']).agg({
        'coefficient': 'mean',
        'std_err': 'mean',
        'p_value': 'mean',
        'significant': 'sum'
    }).reset_index()
    
    # Add significance level column
    summary['sig_level'] = ''
    summary.loc[summary['p_value'] < 0.05, 'sig_level'] = '*'
    summary.loc[summary['p_value'] < 0.01, 'sig_level'] = '**'
    summary.loc[summary['p_value'] < 0.001, 'sig_level'] = '***'
    
    # Save as CSV
    summary.to_csv(os.path.join(output_dir, f"summary_statistics_{target_variable}.csv"), index=False)

    return summary

def create_publication_plots(results_df, output_dir, target_variable, feature_map=None):
    """
    Create publication-quality plots from GEE model results.
    
    Args:
        results_df: DataFrame containing GEE results
        output_dir: Directory to save plots
        target_variable: Target variable name for title
        feature_map: Dictionary mapping technical feature names to display names
    """
    # Make a copy to avoid modifying the original
    df = results_df.copy()
    
    # Map variable names to more readable format
    if feature_map:
        # Function to transform variable names
        def transform_var_name(var):
            if var.lower() == 'intercept':
                return 'Baseline'
            for old, new in feature_map.items():
                if var.replace('_', ' ').lower() == old.replace('_', ' ').lower():
                    return new
            return var.replace('_', ' ').title()
        
        # Apply transformation
        df['variable_display'] = df['variable'].apply(transform_var_name)
    else:
        # Default transformation (remove underscores, capitalize)
        df['variable_display'] = (df['variable']
                             .str.replace('_', ' ')
                             .str.title()
                             .replace({'Intercept': 'Baseline'}))
    
    # Use consistent model naming
    model_map = {
        'GPT-4o': 'GPT-4o',
        'Gemini_1.5_Pro_030425': 'Gemini 1.5 Pro',
        'Llama 3.1 8B': 'Llama 3.1 (8B)',
        'DeepSeek R1 8b': 'DeepSeek R1 (8B)'
    }
    df['model_label'] = df['model'].map(lambda x: model_map.get(x, x.replace('_', ' ')))

    # Choose an ordered category for consistent legend/order
    order = ['DeepSeek R1 (8B)', 'Llama 3.1 (8B)', 'Gemini 1.5 Pro', 'GPT-4o']
    df['model_label'] = pd.Categorical(df['model_label'], categories=order, ordered=True)

    # ----- COEFFICIENT PLOT -----
    create_coefficient_plot(df, output_dir, target_variable, order)

def create_coefficient_plot(df, output_dir, target_variable, order):
    """Create a publication-quality coefficient bar plot with proper significance reporting"""
    # Set professional publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'DeJavu Serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['svg.fonttype'] = 'none'  # For better text in PDF
    
    # Custom color palette inspired by scientific publications
    palette = sns.color_palette([
        "#0173B2", "#DE8F05", "#029E73", "#D55E00", 
        "#CC78BC", "#CA9161", "#FBAFE4", "#949494"
    ])[0:len(order)]
    
    # Filter out intercept for better focus on predictors
    plot_df = df[df['variable'] != 'Intercept']
    # sort exactly the same way seaborn is going to draw:
    plot_df = plot_df.sort_values(
        by=['model_label']
        ).reset_index(drop=True)
    # assume plot_df has columns ['clean_variable','p_value']
    sig = (
        plot_df
        .groupby('variable')['p_value']
        .min()
        .apply(lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '')
        .to_dict()
    )
    # Get display target variable name
    target_display = target_variable.replace('_', ' ').title()
    
    # IMPROVEMENT 1: More informative title
    title = f"Model Tendencies to Agree with Prior Opinions"
    subtitle = f"Impact of Various Factors on Agreement Likelihood"
    
    # Create figure with higher resolution
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    
    # IMPROVEMENT 2: Better variable names for x-axis
    # Clean up variable display names
    def clean_variable_name(var_name):
        if '[T.' in var_name:
            base, level = var_name.split('[T.')
            level = level.rstrip(']')
            return f"{base.strip()}\n{level.strip()}"
        elif var_name == 'Second Agree First':
            return "Agreement with\nFirst Opinion"
        elif var_name == 'Prior Correctness':
            return "Prior Opinion\nCorrectness"
        return var_name
    
    plot_df['clean_variable'] = plot_df['variable_display'].apply(clean_variable_name)
    
    # Print debug info
    print(f"Number of unique variables: {len(plot_df['clean_variable'].unique())}")
    print(f"Variables: {plot_df['clean_variable'].unique()}")
    print(f"Models: {order}")
    
    # Plot bars
    bar = sns.barplot(
        data=plot_df,
        x='clean_variable',
        y='coefficient',
        hue='model_label',
        hue_order=order,
        palette=palette,
        errorbar=('ci',95),     # ← this will center the 95%-CI bars exactly on each bar
        capsize=0.3,
        ax=ax
    )

    # Improve appearance
    for patch in bar.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    patch_data_map = []
    for i, row in enumerate(plot_df.itertuples()):
        if i < len(bar.patches):  # Safety check
            patch_data_map.append((bar.patches[i], row))

    # Improved styling
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.suptitle(subtitle, fontsize=14, y=0.95)
    plt.xlabel("", fontsize=14)
    plt.ylabel("Effect on Agreement Likelihood (Log-Odds)", fontsize=14, fontweight='bold')
    
    # IMPROVEMENT 2: Make x-axis labels more readable with rotation
    plt.xticks(fontsize=11, rotation=45, ha='right', fontweight='bold')
    
    plt.yticks(fontsize=12)
    
    # Add subtle grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Improve legend
    leg = plt.legend(title="Model", fontsize=12, title_fontsize=14, frameon=True, 
                loc='upper right', edgecolor='black')
    leg.get_frame().set_linewidth(1)
    
    # RESTORE SIGNIFICANCE LEGEND
    fig.text(0.01, 0.01, "* p<0.05  ** p<0.01  *** p<0.001", fontsize=10, ha='left', fontweight='bold')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    
    # Add text explaining the plot
    explanation = ("Positive values indicate increased likelihood to agree with prior opinions.\n"
                  "Negative values show decreased likelihood to agree with prior opinions.")
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=11, fontstyle='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # turn on constrained_layout so titles/footnotes aren't clipped
    fig.set_constrained_layout(True)

    # if you prefer tight_layout, push the bottom down:
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    # or more explicitly:
    fig.subplots_adjust(bottom=0.22)
        # grab the current tick texts (in the same order as plot_df clean_variable unique)
    clean_vars = plot_df['clean_variable'].cat.categories.tolist() \
                if hasattr(plot_df['clean_variable'], 'cat') \
                else list(dict.fromkeys(plot_df['clean_variable']))
    dirty_vars = plot_df['variable'].cat.categories.tolist() \
                if hasattr(plot_df['variable'], 'cat') \
                else list(dict.fromkeys(plot_df['variable']))

    new_labels = [f"{var}\n{sig.get(dir_var,'')}" for var, dir_var in zip(clean_vars, dirty_vars)]

    ax.set_xticklabels(new_labels, rotation=45, ha='right', fontsize=11, fontweight='bold')

    # Save in multiple formats for publication
    plt.savefig(os.path.join(output_dir, f"gee_coefficients_{target_variable}_by_model.png"), dpi=300)
    plt.close()

def create_aggregate_performance_comparison(performance_df, output_dir):
    """
    Create a publication-quality plot comparing model performance aggregated across all datasets.
    
    Args:
        performance_df: DataFrame with columns: model, dataset, prior_accuracy, second_accuracy, entropy
        output_dir: Directory to save the plot
    """
    # Set NeurIPS-style aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'text.usetex': False,  # Set to True if LaTeX is installed
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 150
    })
    
    # Define model order and palette
    order = ['DeepSeek R1 8b', 'Llama 3.1 8B', 'Gemini 1.5', 'GPT-4o']
    base_palette = sns.color_palette(["#0173B2", "#DE8F05", "#029E73", "#D55E00"])
    light_palette = [sns.light_palette(c, n_colors=10)[2] for c in base_palette]
    dark_palette = [sns.dark_palette(c, n_colors=10)[2] for c in base_palette]
    
    # Aggregate metrics
    agg_df = performance_df.groupby('model').agg({
        'prior_accuracy': ['mean', 'std', 'count'],
        'second_accuracy': ['mean', 'std', 'count'],
        'entropy': ['mean', 'std', 'count']
    }).reset_index()
    agg_df.columns = ['_'.join(col).rstrip('_') for col in agg_df.columns.values]
    
    # Standard errors
    for metric in ['prior_accuracy', 'second_accuracy', 'entropy']:
        agg_df[f'{metric}_sem'] = agg_df[f'{metric}_std'] / np.sqrt(agg_df[f'{metric}_count'])
    
    # Sort
    model_order_map = {m: i for i, m in enumerate(order)}
    agg_df['model_order'] = agg_df['model'].map(model_order_map)
    agg_df = agg_df.sort_values('model_order')
    
    # Plot setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={'height_ratios': [1.1, 1]})
    bar_width = 0.35
    index = np.arange(len(agg_df))
    
    # Accuracy Bars
    prior_bars = ax1.bar(index - bar_width/2, agg_df['prior_accuracy_mean'], bar_width,
                         color=light_palette, yerr=agg_df['prior_accuracy_sem'],
                         capsize=3, label='Prior Opinion')
    
    second_bars = ax1.bar(index + bar_width/2, agg_df['second_accuracy_mean'], bar_width,
                          color=dark_palette, yerr=agg_df['second_accuracy_sem'],
                          capsize=3, label='Second Opinion (Majority)')
    
    ax1.set_title('Model Accuracy Comparison (All Datasets)', fontweight='bold')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_xticks(index)
    ax1.set_xticklabels(agg_df['model'], rotation=45, ha='right')
    ax1.set_ylim(0.25, 1.0)
    ax1.axhline(0.25, color='red', linestyle='--', linewidth=1)
    # ax1.text(len(order)-0.5, 0.27, 'Random Chance (0.25)', color='red', fontsize=10)
    ax1.legend(loc='upper right')

    for i, bar in enumerate(prior_bars):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f"{agg_df['prior_accuracy_mean'].iloc[i]:.3f}", ha='center', fontsize=9)
    for i, bar in enumerate(second_bars):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f"{agg_df['second_accuracy_mean'].iloc[i]:.3f}", ha='center', fontsize=9)

    # Entropy Bars
    entropy_bars = ax2.bar(index, agg_df['entropy_mean'], bar_width * 1.7,
                           color=light_palette, yerr=agg_df['entropy_sem'], capsize=3)

    ax2.set_title('Model Response Diversity (All Datasets)', fontweight='bold')
    ax2.set_ylabel('Entropy (bits)')
    ax2.set_xlabel('Model')
    ax2.set_xticks(index)
    ax2.set_xticklabels(agg_df['model'], rotation=45, ha='right')
    ax2.set_ylim(0, 2.05)
    ax2.axhline(2.0, color='blue', linestyle='--', linewidth=1)
    ax2.text(len(order)-0.5, 1.97, 'Maximum Entropy (2.0)', color='blue', fontsize=10)

    for i, bar in enumerate(entropy_bars):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{agg_df['entropy_mean'].iloc[i]:.3f}", ha='center', fontsize=9)

    # Caption
    fig.text(0.5, 0.01,
             "Values averaged across all datasets. Higher accuracy scores are better.\n"
             "Lower entropy indicates more consistent responses across contexts.",
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    fig_path_base = os.path.join(output_dir, "model_aggregate_performance")
    plt.savefig(f"{fig_path_base}.png", dpi=300)
    plt.savefig(f"{fig_path_base}.pdf")
    plt.close()

    print(f"Aggregate performance comparison plot saved to {fig_path_base}.(png/pdf)")
    
    # Summary CSV
    summary_table = agg_df[['model', 
                            'prior_accuracy_mean', 'prior_accuracy_std',
                            'second_accuracy_mean', 'second_accuracy_std',
                            'entropy_mean', 'entropy_std']]
    summary_table.columns = ['Model', 
                             'Prior Acc (Mean)', 'Prior Acc (Std)',
                             'Second Acc (Mean)', 'Second Acc (Std)',
                             'Entropy (Mean)', 'Entropy (Std)']
    summary_table.to_csv(os.path.join(output_dir, "model_aggregate_performance.csv"), index=False)
    
    return agg_df

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV data.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns.")

        df['first_agree_baseline'] = (df.first_opinion_letter==df.baseline_opinion_letter).astype(int)
        df['second_agree_first'] = (df.first_opinion_letter==df.second_opinion_letter).astype(int)
        df['second_agree_baseline'] = (df.second_opinion_letter==df.baseline_opinion_letter).astype(int)

    except Exception as e:
        print(f"Error loading data: {e}")
        df = pd.DataFrame()
    return df

def from_csvs_to_dfs(csv_paths):
    
    model_dfs = {}
    for csv_path in csv_paths:
        model_name = os.path.splitext(os.path.basename(csv_path))[0]
        if 'DeepSeek' in model_name:
            model_name = 'DeepSeek R1 8B'
        elif 'Llama' in model_name:
            model_name = 'Llama 3.1 8B'
        elif 'OpenAI' in model_name:
            model_name = 'GPT-4o'
        else:
            model_name = "Gemini 1.5 Pro"


        print(f"\nAnalyzing model: {model_name}")
        df = load_data(csv_path)
        model_dfs[model_name] = df

    return pd.concat(
        [df.assign(model_origin=name) for name, df in model_dfs.items()],
        ignore_index=True
    )

def find_matched_pairs_pandas(df: pd.DataFrame, feature: str, control_features: list) -> pd.DataFrame:
    """
    Find pairs of rows where only the specified feature differs and all control_features are identical.
    
    Returns:
        DataFrame with columns [question_id, idx1, idx2, feature_val1, feature_val2]
    """
    # Merge dataset on control features to find potential pairs
    merged = df.merge(df, on=['question_id'] + control_features, suffixes=('_1', '_2'))
    
    # Remove self-pairs and ensure the feature differs
    matched_pairs = merged[
        (merged['index_1'] < merged['index_2']) &  # Prevent duplicates
        (merged[f'{feature}_1'] != merged[f'{feature}_2'])  # Ensure only the feature differs
    ][['question_id', 'index_1', 'index_2', f'{feature}_1', f'{feature}_2']]
    
    print(f"Found {len(matched_pairs)} matched pairs for feature '{feature}'")
    return matched_pairs

def analyze_feature_impact(df: pd.DataFrame, features: list, control_features_list: list = None, num_of_bins: int = 20) -> dict:
    """
    Analyze how each feature impacts the model's second opinion using pandas.
    
    Returns:
        Dictionary with feature impact analysis results
    """
    results = pd.DataFrame()
    
    for feature, control_features in zip(features, control_features_list):
        print(f"\nAnalyzing feature: {feature}")
        print(f"Control features: {control_features}")
        
        # Get matched pairs
        pairs_df = find_matched_pairs_pandas(df.reset_index(), feature, control_features)
        
        # Calculate agreement rate
        feature_df = pd.DataFrame({
            'question_id': pairs_df['question_id'],
            'second_opinion_1': df.loc[pairs_df['index_1'], 'second_opinion_letter'].values,
            'second_opinion_2': df.loc[pairs_df['index_2'], 'second_opinion_letter'].values,
            'first_opinion': df.loc[pairs_df['index_2'], 'first_opinion_letter'].values,
            'baseline_opinion': df.loc[pairs_df['index_2'], 'baseline_opinion_letter'].values,
            'baseline_correctness': df.loc[pairs_df['index_2'], 'baseline_correctness'].values,
            'gold': df.loc[pairs_df['index_2'], 'gold_letter'].values,
        })
        feature_df['pairs_agree'] = feature_df['second_opinion_1'] == feature_df['second_opinion_2']
            # Calculate agreement rate
        feature_df['first_agrees_baseline'] = feature_df['baseline_opinion'] == feature_df['first_opinion']
        feature_df['first_true'] = feature_df['first_opinion'] == feature_df['gold']
        all_feature_df = pd.DataFrame(feature_df.groupby('question_id').pairs_agree.mean()).reset_index()
        confirm_first_df = pd.DataFrame(feature_df[feature_df.first_agrees_baseline==True].groupby('question_id').pairs_agree.mean()).reset_index()

        # Add feature_df to results with suffix for feature name
        all_feature_df.columns = [f"{feature}_{col}" if col not in ['question_id', 'baseline_correct_rate'] else col for col in all_feature_df.columns]
        confirm_first_df.columns = [f"first_confirm_{feature}_{col}" if col not in ['question_id', 'baseline_correct_rate'] else col for col in confirm_first_df.columns]
        results = pd.concat([results, all_feature_df, confirm_first_df], axis=1)    
        results = results.loc[:,~results.columns.duplicated()]

    return results

def analyze_levels_of_agreement(dataset_df: pd.DataFrame, num_of_bins: int = 20):
    results = {}
    results['question_id'] = dataset_df['question_id']
    results['second_agrees_baseline'] = (dataset_df['second_opinion_letter'] == dataset_df['baseline_opinion_letter'])
    results['second_disagree_baseline'] = (dataset_df['second_opinion_letter'] != dataset_df['baseline_opinion_letter'])
    results['second_agrees_first'] = (dataset_df['second_opinion_letter'] == dataset_df['first_opinion_letter'])
    results['second_disagree_first'] = (dataset_df['second_opinion_letter'] != dataset_df['first_opinion_letter'])
    results['second_disagree_both'] = (dataset_df['second_opinion_letter'] != dataset_df['first_opinion_letter']) & (dataset_df['second_opinion_letter'] != dataset_df['baseline_opinion_letter'])
    results['second_agrees_one'] = (results['second_agrees_baseline'] | results['second_agrees_first']) & ~(results['second_agrees_baseline'] & results['second_agrees_first'])
    results['second_agrees_both'] = results['second_agrees_baseline'] & results['second_agrees_first']
    results['second_agrees_none'] = (~results['second_agrees_baseline'] & ~results['second_agrees_first'])
    results['second_agrees_one_baseline'] = (results['second_agrees_baseline'] & ~results['second_agrees_first'])
    results['second_agrees_one_first'] = (~results['second_agrees_baseline'] & results['second_agrees_first'])
    results['first_agrees_baseline'] = (dataset_df['first_opinion_letter'] == dataset_df['baseline_opinion_letter'])

    cols = ['second_agrees_baseline', 'second_agrees_first', 'second_agrees_both','second_agrees_none', 'second_agrees_one', 'second_disagree_baseline', 'second_disagree_both', 'second_disagree_first', 'first_agrees_baseline']
    results = pd.DataFrame(results)
    agreement_by_question = results.groupby('question_id').agg({
        'second_agrees_baseline': 'mean',
        'second_agrees_first': 'mean',
        'second_agrees_both': 'mean',
        'second_agrees_none': 'mean',
        'second_agrees_one': 'mean',
        'second_disagree_baseline': 'mean',
        'second_disagree_both': 'mean', 
        'second_disagree_first': 'mean',
        'second_agrees_one_baseline': 'mean',
        'second_agrees_one_first': 'mean',
        'first_agrees_baseline': 'mean'
    }).reset_index()

    return agreement_by_question

def from_dfs_to_analyzed_dfs(df, mode='mmlu', output_dir='output', num_bins=20):
    """
    Run the full analysis pipeline.
    """
    # if file exist load it
    if os.path.exists(os.path.join(output_dir, 'feature_impact_results_combined.csv')):
        print(f"Feature impact results already exist in {output_dir}.")
        return pd.read_csv(os.path.join(output_dir, 'feature_impact_results_combined.csv'))
    else:
        informative_context = ['source'] 
        if mode !='mmlu':
            informative_context+= ['experience']
        non_informative_context = ['gender', 'age']
        first_opinion_context = ['first_opinion_letter']
        all_features = informative_context + non_informative_context + first_opinion_context
        
        model_results={}
        for model, model_df in df.groupby('model_origin'):
            print(f"\n{'='*50}")
            print(f"Analyzing model: {model}")
            dataset_results={}
            for dataset, dataset_df in model_df.groupby('dataset_origin'):
                print(f"\n{'='*50}")
                print(f"Analyzing dataset: {dataset}")
                print(f"{'='*50}")
                
                # Create control feature lists (excluding the feature being analyzed)
                control_features_list = []
                for feature in all_features:
                    control_features_list.append([f for f in all_features if f != feature])

                # 1. Calculate agreement rates for feature manipulation
                second_agreement_results_per_question = analyze_feature_impact(
                    dataset_df, 
                    informative_context + non_informative_context,
                    control_features_list[:-1] # Exclude first_opinion_letter controls,
                )

                # 2. Calculate agreement rates for first opinion manipulation
                inner_row_agreement = analyze_levels_of_agreement(dataset_df, num_of_bins=num_bins)
                results = pd.concat([second_agreement_results_per_question, inner_row_agreement], axis=1)
                results['baseline_correctness'] = dataset_df.groupby('question_id').baseline_correctness.first().values
                results = results.loc[:,~results.columns.duplicated()]

                # sampling
                for idx in range(1, num_bins):
                    sampled_indices = dataset_df.groupby('question_id').sample(frac=idx/num_bins, random_state=42).index
                    sampled_results = dataset_df.loc[sampled_indices]
                    second_agreement_sampled = analyze_feature_impact(sampled_results, informative_context + non_informative_context,
                        control_features_list[:-1] # Exclude first_opinion_letter controls,
                    )
                    inner_row_sampled = analyze_levels_of_agreement(sampled_results)
                    sampled_results = pd.concat([second_agreement_sampled, inner_row_sampled], axis=1)
                    #remove duplicate columns
                    sampled_results = sampled_results.loc[:,~sampled_results.columns.duplicated()]
                    #change columns name to _sampled_idx
                    sampled_results.columns = [f"{col}_sampled_{idx}" if col not in ['question_id'] else col for col in sampled_results.columns]
                    results = pd.merge(results, sampled_results, on='question_id', how='left')
                # 3. Calculate entropy per question
                def entropy(group):
                    # Calculate entropy of second_opinion_letter
                    letter_counts = group['second_opinion_letter'].value_counts()
                    entropy = -np.sum((letter_counts / letter_counts.sum()) * 
                                      np.log2(letter_counts / letter_counts.sum() + 1e-10))
                    return entropy
                # Calculate entropy for each question
                entropy_df = pd.DataFrame()
                entropy_df['entropy'] = dataset_df.groupby('question_id').apply(entropy) #.reset_index(name='entropy')
                # sample entropy_df
                for i in range(1, num_bins):
                    sampled_indices = dataset_df.groupby('question_id').sample(frac=i/num_bins, random_state=42).index
                    entropy_df[f'entropy_sampled_{i}'] = dataset_df.loc[sampled_indices].groupby('question_id').apply(entropy) #.reset_index(name='entropy')['entropy'].values
                # Merge entropy with results
                results['entropy_second_opinions'] = entropy_df['entropy'].values
                # sample entropy_df
                for i in range(1, num_bins):
                    results[f'entropy_second_opinions_sampled_{i}'] = entropy_df[f'entropy_sampled_{i}'].values
                dataset_results[dataset] =  results

            combined_df = pd.concat(
                [df.assign(dataset_origin=name) for name, df in dataset_results.items()],
                ignore_index=True
                )
            # Save combined results for the model
            model_results[model] = combined_df
        # Save all models' results
        model_dfs = pd.concat(
            [df.assign(model_origin=name) for name, df in model_results.items()],
            ignore_index=True)
        return model_dfs

def calculate_performance_metrics(all_dfs, output_dir):
    """
    Calculate performance metrics for each model and dataset, and generate a LaTeX table.
    
    Args:
        all_dfs: DataFrame containing evaluation data with model_origin and dataset_origin
        output_dir: Directory to save the LaTeX table
        
    Returns:
        str: LaTeX formatted table with performance metrics
    """
    results = []
    
    # Get unique dataset origins and model origins
    datasets = all_dfs['dataset_origin'].unique()
    models = all_dfs['model_origin'].unique()
    
    print("\n===== PERFORMANCE METRICS =====")
    print(f"{'Dataset':<20} {'Model':<15} {'Prior Acc':<10} {'Second Acc':<10} {'Entropy':<10}")
    print("="*65)
    
    for dataset in datasets:
        dataset_df = all_dfs[all_dfs['dataset_origin'] == dataset]
        
        for model in models:
            model_df = dataset_df[dataset_df['model_origin'] == model]
            
            if "Gpt" in model or "GPT" in model:
                model_name = "GPT-4o"
            elif "Llama" in model:
                model_name = "Llama 3.1 8B"
            elif "DeepSeek" in model:
                model_name = "DeepSeek R1 8B"
            else:
                model_name = "Gemini 1.5 Pro"
            if len(model_df) == 0:
                continue
            
            # Calculate prior opinion accuracy
            prior_accuracy = model_df['baseline_correctness'].mean()
            
            # Calculate second opinion majority accuracy - improved method
            question_results = []
            for question_id, group in model_df.groupby('question_id'):
                # Find most frequent second_opinion_letter (majority vote)
                letter_counts = group['second_opinion_letter'].value_counts()
                if len(letter_counts) > 0:
                    majority_letter = letter_counts.idxmax()
                    # Get gold_letter (assumes it's the same for all rows with this question_id)
                    gold_letter = group['gold_letter'].iloc[0]
                    # Check if majority letter matches gold letter
                    is_correct = (majority_letter == gold_letter)
                    
                    question_results.append({
                        'question_id': question_id,
                        'is_correct': is_correct,
                        'letter_counts': letter_counts,
                        'entropy': -np.sum((letter_counts / letter_counts.sum()) * 
                                          np.log2(letter_counts / letter_counts.sum() + 1e-10))
                    })
            
            # Calculate accuracy and entropy
            if question_results:
                # Second opinion accuracy using majority vote
                second_accuracies = [int(r['is_correct']) for r in question_results]
                second_accuracy = np.mean(second_accuracies)
                second_std = np.std(second_accuracies)                # Entropy (bits) across questions
                entropy = np.mean([r['entropy'] for r in question_results])
            else:
                second_accuracy = 0
                entropy = 0
            
            # Print performance metrics
            print(f"{dataset:<20} {model_name:<15} {prior_accuracy:.3f}     {second_accuracy:.3f}     {entropy:.3f}")
            
            results.append({
                'dataset': dataset,
                'model': model_name,
                'prior_accuracy': prior_accuracy,
                'second_accuracy': second_accuracy,
                'second_std': second_std,
                'entropy': entropy
            })
    performance_df = pd.DataFrame(results)
    print("\n=== BEST PERFORMERS BY DATASET ===")
    
    # Find best values for each dataset
    best_prior = performance_df.groupby('dataset')['prior_accuracy'].max().to_dict()
    best_second = performance_df.groupby('dataset')['second_accuracy'].max().to_dict()
    # Change dataset names:
    renames = {
        'general_surgery': 'General Surgery',
        'internal_medicine': 'Internal Medicine',
        'psychiatry': 'Psychiatry',
        'pediatrics': 'Pediatrics',
        'obgyn': 'Obstetrics & Gynecology',
    }
    # Print best performers
    for dataset in datasets:
        best_prior_row = performance_df[(performance_df['dataset'] == dataset) & 
                                       (performance_df['prior_accuracy'] == best_prior[dataset])]
        best_second_row = performance_df[(performance_df['dataset'] == dataset) & 
                                        (performance_df['second_accuracy'] == best_second[dataset])]
        
        print(f"\nDataset: {dataset}")
        print(f"  Best Prior Accuracy: {best_prior[dataset]:.3f} ({best_prior_row.iloc[0]['model']})")
        print(f"  Best Second Opinion: {best_second[dataset]:.3f} ({best_second_row.iloc[0]['model']})")
        # Generate LaTeX table
        # Generate LaTeX table
    latex_table = r"""\begin{table}[htbp]
\centering
\caption{
Comparison of models' prior opinion accuracy with the majority vote accuracy of their second opinions. 
Higher values in the Prior and Second Opinion Majority Accuracy columns are better (\textuparrow). 
Entropy (in bits) quantifies the diversity of second-opinion responses — lower values indicate more agreement (\textdownarrow). 
Entropy ranges from 0 (total agreement) to 2.0 (maximum disagreement with 4 options). 
The best accuracy performance for each dataset is highlighted in \textbf{bold}.
}
\label{tab:model_performance_with_entropy}
\begin{tabular}{lccc}
\toprule
\textbf{Dataset} & \textbf{Model} & 
\makecell{\textbf{Prior Opinion} \\ \textbf{Accuracy} (\textuparrow)} & 
\makecell{\textbf{Second Opinion} \\ \textbf{Accuracy} (\textuparrow)} & 
\makecell{\textbf{Entropy} \\ \textbf{(bits)} (\textdownarrow)} \\
\midrule
"""     
    
    # Find best values for each dataset
    best_prior = performance_df.groupby('dataset')['prior_accuracy'].max().to_dict()
    best_second = performance_df.groupby('dataset')['second_accuracy'].max().to_dict()
    best_entropy = performance_df.groupby('dataset')['entropy'].min().to_dict()
    
    # Sort by dataset and then by model
    performance_df = performance_df.sort_values(['dataset', 'model'])
    
    current_dataset = None
    models_per_dataset = 0
    
    for idx, (_, row) in enumerate(performance_df.iterrows()):
        # Check if we're starting a new dataset section
        if current_dataset != row['dataset']:
            if current_dataset is not None:
                latex_table += r"\midrule" + "\n\n"
            current_dataset = row['dataset']
            models_per_dataset = len(performance_df[performance_df['dataset'] == current_dataset])
            latex_table += f"\\multirow{{{models_per_dataset}}}{{*}}{{\\textbf{{{current_dataset}}}}}\n"
            
        # Get sample size (n) for standard error calculation
        dataset_df = all_dfs[(all_dfs['dataset_origin'] == row['dataset']) & 
                             (all_dfs['model_origin'] == row['model'])]
        n_samples = len(dataset_df['question_id'].unique())
        
        # Calculate standard errors
        prior_se = np.sqrt(row['prior_accuracy'] * (1 - row['prior_accuracy']) / n_samples)
        second_se = np.sqrt(row['second_accuracy'] * (1 - row['second_accuracy']) / n_samples)
        
        # Check if model is best in any category
        is_best_prior = abs(row['prior_accuracy'] - best_prior[row['dataset']]) < 1e-6
        is_best_second = abs(row['second_accuracy'] - best_second[row['dataset']]) < 1e-6
        is_best_entropy = abs(row['entropy'] - best_entropy[row['dataset']]) < 1e-6
        
        # Format model name
        model_str = row['model']
        
        # Format accuracies with standard errors, bold if best in dataset
        prior_str = f"{row['prior_accuracy']:.3f} $\\pm$ {prior_se:.3f}"
        if is_best_prior:
            prior_str = f"\\textbf{{{prior_str}}}"
            
        second_str = f"{row['second_accuracy']:.3f} $\\pm$ {second_se:.3f}"
        if is_best_second:
            second_str = f"\\textbf{{{second_str}}}"
        
        entropy_str = f"{row['entropy']:.3f}"
        if is_best_entropy:
            entropy_str = f"\\textbf{{{entropy_str}}}"
        
        latex_table += f"& {model_str} & {prior_str} & {second_str} & {entropy_str} \\\\\n"
    
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}"""
    # Save the LaTeX table to a file
    with open(os.path.join(output_dir, "performance_table.tex"), "w") as f:
        f.write(latex_table)
    return latex_table

def from_cal_csv_to_dfs(calibration_csv_paths, combined_analyzed):
    if calibration_csv_paths:
        print("\nLoading calibration data...")
        calibration_dfs = []
        
        for cal_path in calibration_csv_paths:
            model_name = os.path.splitext(os.path.basename(cal_path))[0]
            if 'DeepSeek' in model_name or 'deepseek' in model_name:
                model_name = 'DeepSeek R1 8B'
            elif 'Llama' in model_name or 'llama' in model_name:
                model_name = 'Llama 3.1 8B'
            elif 'OpenAI' in model_name or 'gpt' in model_name.lower():
                model_name = 'GPT-4o'
            else:
                model_name = model_name.title()
                
            # Load calibration data
            cal_df = pd.read_csv(cal_path)
            cal_df['model_origin'] = model_name
            
            # Rename columns to avoid confusion with existing columns
            if model_name == 'GPT-4o':
                cal_df = cal_df.rename(columns={
                    'answer_letter': 'calibrated_answer_letter',
                    'answer': 'calibrated_answer',
                    'answer_correctness': 'calibrated_correctness',
                    'answer_logprob': 'logprobs',
                    'calibration_score': 'verbalization_score',
                })
            else:
                cal_df = cal_df.rename(columns={
                    'answer_letter': 'calibrated_answer_letter',
                    'answer': 'calibrated_answer',
                    'answer_correctness': 'calibrated_correctness',
                    'answer_logprobs': 'logprobs',
                    'calibration_score': 'verbalization_score',
                })
            
            calibration_dfs.append(cal_df)
        
        # Combine all calibration data
        all_calibration_df = pd.concat(calibration_dfs, ignore_index=True)
        print(f"Merged calibration data for {len(all_calibration_df)} questions")
        offsets = {
            'general_surgery': 0,
            'internal_medicine': 528,
            'obgyn': 389,
            'pediatrics': 140,
            'psychiatry': 239
        }
        combined_analyzed = combined_analyzed.loc[:,~combined_analyzed.columns.duplicated()]
        # Adjust question_id  according to the offset, but just for Llama 3.1 8B
        combined_analyzed.loc[combined_analyzed['model_origin'] == 'Llama 3.1 8B', 'question_id'] = combined_analyzed.loc[combined_analyzed['model_origin'] == 'Llama 3.1 8B'].apply(lambda row: row['question_id'] - offsets[row['dataset_origin']], axis=1)

        # Apply transformation
        all_calibration_df['question_id'] = all_calibration_df.apply(lambda row: row['question_id'] - offsets[row['dataset_origin']], axis=1)

        combined_analyzed = combined_analyzed.merge(
            all_calibration_df[['model_origin', 'dataset_origin', 'question_id',                                 'calibrated_correctness', 'logprobs', 
                                'verbalization_score', 'calibrated_correctness', 'logprobs']],
            on=['model_origin', 'dataset_origin', 'question_id'],
            how='left'
        )
        combined_analyzed = combined_analyzed.loc[:,~combined_analyzed.columns.duplicated()]
        # fill na of calibrated correctness with baseline correctness
        combined_analyzed['calibrated_correctness'] = combined_analyzed['calibrated_correctness'].fillna(combined_analyzed['baseline_correctness'])
    return combined_analyzed
