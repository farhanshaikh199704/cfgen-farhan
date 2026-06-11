#!/usr/bin/env python3
"""
Analyze labeled claims from human evaluators.

This script:
1. Loads completed labeling forms
2. Calculates inter-rater agreement on the overlap set
3. Analyzes quality by model and method
4. Generates visualizations and exports results

Usage:
    python analyze_labelled_claims.py --input_dir human_annotations/labelling_samples
"""

import pandas as pd
import numpy as np
import glob
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
import statsmodels.api as sm


def load_labelled_data(input_dir, include_machine_evals=True):
    """Load all labelled data from forms"""
    # Look for files with 'full' in the name (these contain metadata)
    form_files = glob.glob(os.path.join(input_dir, "*full*.csv"))
    
    if not form_files:
        raise ValueError(f"No labelled forms found in {input_dir}")
    
    dfs = []
    
    for file in form_files:
        df = pd.read_csv(file)
        
        # Extract labeler ID from filename if not in data
        if 'labeler_id' not in df.columns:
            filename = os.path.basename(file)
            try:
                labeler_id = int(filename.split('_')[1])
                df['labeler_id'] = labeler_id
            except (IndexError, ValueError):
                # If we can't extract it from filename, use file index
                file_idx = form_files.index(file) + 1
                df['labeler_id'] = file_idx
                print(f"Warning: Could not extract labeler_id from {filename}, using {file_idx}")
        
        # Mark as human evaluation
        df['source'] = 'human'
        dfs.append(df)
    
    # Load machine evaluations if requested
    if include_machine_evals:
        machine_dir = 'machine_evaluations'
        if os.path.exists(machine_dir):
            machine_files = glob.glob(os.path.join(machine_dir, "machine_evaluated_*.csv"))
            
            for file in machine_files:
                try:
                    df = pd.read_csv(file)
                    
                    # Add machine labeler ID
                    df['labeler_id'] = 'machine'
                    df['source'] = 'machine'
                    
                    # Rename columns if needed to match human labels
                    if 'self_containment_score' in df.columns:
                        pass  # Already using the right format
                    else:
                        # Handle potential different column naming
                        column_map = {
                            'self_containment': 'self_containment_score',
                            'factuality': 'factuality_score',
                            'entity_precision': 'entity_precision_score',
                            'factual_alignment': 'factual_alignment_score',
                            'language_matching': 'language_matching_score'
                        }
                        df = df.rename(columns=column_map)
                    
                    dfs.append(df)
                    print(f"Loaded machine evaluations from {file}")
                except Exception as e:
                    print(f"Error loading machine evaluations from {file}: {str(e)}")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert score columns to categorical
    score_columns = [col for col in combined_df.columns if col.endswith('_score')]
    for col in score_columns:
        combined_df[col] = combined_df[col].astype('category')
    
    print(f"Loaded {len(combined_df)} labelled claims")
    return combined_df


def analyze_inter_rater_agreement(df):
    """Analyze agreement between labelers on the overlap set"""
    # Filter to just the overlap claims
    overlap_df = df[df['is_overlap'] == True].copy()
    
    # Get the list of score columns and labelers
    score_columns = [col for col in overlap_df.columns if col.endswith('_score')]
    human_labelers = sorted([l for l in overlap_df['labeler_id'].unique() if l != 'machine'])
    
    # Create a pivot table for each criterion
    agreement_results = []
    
    # Human-Human agreement
    for col in score_columns:
        # Create a pivot of labeler ratings
        pivot = pd.pivot_table(
            overlap_df[overlap_df['source'] == 'human'], 
            values=col, 
            index=['synthetic_claim'],
            columns=['labeler_id'],
            aggfunc=lambda x: x
        )
        
        # Calculate pairwise agreement (Cohen's Kappa) for each pair of labelers
        kappa_values = []
        labeler_pairs = []
        
        for i, labeler1 in enumerate(human_labelers):
            for labeler2 in human_labelers[i+1:]:
                # Get ratings from both labelers and filter out rows with missing values
                ratings = pivot[[labeler1, labeler2]].dropna()
                
                if len(ratings) > 1:  # Need at least 2 ratings to calculate kappa
                    kappa = cohen_kappa_score(
                        ratings[labeler1].apply(lambda x: 1 if x.lower() == 'yes' else 0),
                        ratings[labeler2].apply(lambda x: 1 if x.lower() == 'yes' else 0)
                    )
                    kappa_values.append(kappa)
                    labeler_pairs.append(f"{labeler1}-{labeler2}")
        
        # Average kappa across all pairs
        if kappa_values:
            avg_kappa = np.mean(kappa_values)
            criterion = col.replace('_score', '')
            agreement_results.append({
                'criterion': criterion,
                'agreement_type': 'Human-Human',
                'avg_kappa': avg_kappa,
                'min_kappa': min(kappa_values),
                'max_kappa': max(kappa_values),
                'pairwise_kappas': kappa_values,
                'pairs': labeler_pairs
            })
    
    # Check if we have machine evaluations
    if 'machine' in overlap_df['labeler_id'].unique():
        # Human-Machine agreement
        for col in score_columns:
            # For each human labeler, compare with machine
            kappa_values = []
            labeler_pairs = []
            
            machine_df = overlap_df[overlap_df['labeler_id'] == 'machine']
            
            for labeler in human_labelers:
                human_df = overlap_df[overlap_df['labeler_id'] == labeler]
                
                # Merge by synthetic claim to get pairs of ratings
                merged = pd.merge(
                    human_df[['synthetic_claim', col]], 
                    machine_df[['synthetic_claim', col]],
                    on='synthetic_claim',
                    suffixes=('_human', '_machine')
                )
                
                if len(merged) > 1:
                    kappa = cohen_kappa_score(
                        merged[f"{col}_human"].apply(lambda x: 1 if x.lower() == 'yes' else 0),
                        merged[f"{col}_machine"].apply(lambda x: 1 if x.lower() == 'yes' else 0)
                    )
                    kappa_values.append(kappa)
                    labeler_pairs.append(f"{labeler}-machine")
            
            if kappa_values:
                avg_kappa = np.mean(kappa_values)
                criterion = col.replace('_score', '')
                agreement_results.append({
                    'criterion': criterion,
                    'agreement_type': 'Human-Machine',
                    'avg_kappa': avg_kappa,
                    'min_kappa': min(kappa_values),
                    'max_kappa': max(kappa_values),
                    'pairwise_kappas': kappa_values,
                    'pairs': labeler_pairs
                })
    
    # Create a dataframe of results
    agreement_df = pd.DataFrame(agreement_results)
    
    # Sort by agreement type first, then by average kappa within each type
    agreement_df = agreement_df.sort_values(['agreement_type', 'avg_kappa'], ascending=[True, False])
    
    return agreement_df


def calculate_overall_quality(df):
    """Calculate overall quality based on number of 'yes' scores"""
    # Map the criteria scores to 1 (Yes) or 0 (No)
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    for col in score_columns:
        df[f"{col}_value"] = df[col].apply(
            lambda x: 1 if str(x).lower() == 'yes' else 0
        )
    
    # Calculate the sum of 'yes' scores
    value_columns = [col for col in df.columns if col.endswith('_score_value')]
    df['total_yes'] = df[value_columns].sum(axis=1)
    
    # Map to quality categories
    df['calculated_quality'] = df['total_yes'].apply(
        lambda x: 'High' if x == 5 else ('Medium' if x >= 3 else 'Low')
    )
    
    # Check if the manually assigned quality matches
    if 'overall_quality' in df.columns:
        df['quality_match'] = df['calculated_quality'] == df['overall_quality']
        match_rate = df['quality_match'].mean() * 100
        print(f"Overall quality match rate: {match_rate:.1f}%")
    
    return df


def analyze_quality_by_group(df):
    """Analyze quality metrics by model and method"""
    # Calculate quality metrics
    df = calculate_overall_quality(df)
    
    # Group by model and method
    grouped = df.groupby(['model', 'method']).agg({
        'total_yes': ['mean', 'std'],
        'calculated_quality': lambda x: (x == 'High').mean(),
        'self_containment_score_value': 'mean',
        'factuality_score_value': 'mean',
        'entity_precision_score_value': 'mean',
        'factual_alignment_score_value': 'mean',
        'language_matching_score_value': 'mean'
    }).reset_index()
    
    # Flatten the column multi-index
    grouped.columns = [
        '_'.join(col).strip('_') for col in grouped.columns.values
    ]
    
    # Rename columns for clarity
    column_renames = {
        'calculated_quality_<lambda>': 'high_quality_rate',
        'total_yes_mean': 'avg_yes_count',
        'total_yes_std': 'std_yes_count'
    }
    
    grouped = grouped.rename(columns=column_renames)
    
    # Add a combined group column
    grouped['group'] = grouped['model'] + '_' + grouped['method']
    
    # Sort by average yes count
    grouped = grouped.sort_values('avg_yes_count', ascending=False)
    
    return grouped


def compare_human_machine_quality(df):
    """Compare quality assessment between humans and machine"""
    # Check if we have machine evaluations
    if 'machine' not in df['labeler_id'].unique():
        return None
    
    # Calculate overall quality based on criteria scores
    df = calculate_overall_quality(df)
    
    # Create pivot table with claims as rows and source (human/machine) as columns
    # Calculate average quality scores for each claim by source
    quality_pivot = pd.pivot_table(
        df,
        values=['total_yes', 'calculated_quality'],
        index=['synthetic_claim'],
        columns=['source'],
        aggfunc={'total_yes': 'mean', 'calculated_quality': lambda x: x.mode()[0] if len(x) > 0 else None}
    )
    
    # Flatten column names
    quality_pivot.columns = ['_'.join(col) for col in quality_pivot.columns]
    
    # Calculate agreement rates
    comparison = {}
    
    # Yes count agreement
    diff = abs(quality_pivot['total_yes_human'] - quality_pivot['total_yes_machine'])
    comparison['avg_yes_count_diff'] = diff.mean()
    comparison['exact_yes_count_agreement'] = (diff == 0).mean() * 100
    comparison['close_yes_count_agreement'] = (diff <= 1).mean() * 100
    
    # Quality level agreement (High/Medium/Low)
    quality_agreement = (quality_pivot['calculated_quality_human'] == quality_pivot['calculated_quality_machine']).mean() * 100
    comparison['quality_level_agreement'] = quality_agreement
    
    # Create a confusion matrix for quality levels
    human_quality = df[df['source'] == 'human'].groupby('synthetic_claim')['calculated_quality'].agg(lambda x: x.mode()[0])
    machine_quality = df[df['source'] == 'machine'].groupby('synthetic_claim')['calculated_quality'].agg(lambda x: x.mode()[0])
    
    # Merge them
    quality_comparison = pd.merge(
        human_quality.reset_index(), 
        machine_quality.reset_index(),
        on='synthetic_claim',
        suffixes=('_human', '_machine')
    )
    
    # Create confusion matrix
    quality_levels = ['High', 'Medium', 'Low']
    confusion_matrix = pd.DataFrame(index=quality_levels, columns=quality_levels, data=0)
    
    for _, row in quality_comparison.iterrows():
        human_val = row['calculated_quality_human']
        machine_val = row['calculated_quality_machine']
        confusion_matrix.loc[human_val, machine_val] += 1
    
    # Calculate percentages
    row_sums = confusion_matrix.sum(axis=1)
    confusion_pct = confusion_matrix.divide(row_sums, axis=0) * 100
    
    comparison['confusion_matrix'] = confusion_matrix
    comparison['confusion_matrix_pct'] = confusion_pct
    
    return comparison


def analyze_is_checkable_agreement(annotated_csv, llm_results_csv=None, output_dir='results/is_checkable_agreement'):
    """Analyze inter-labeler agreement for is_checkable binary annotations.

    Treats each annotator (Ruggero, Ilaria, and optionally the LLM) as a labeler
    and computes pairwise Cohen's Kappa, accuracy, and confusion matrices.
    Reports on calibration (first 100) and test (last 100) splits separately.
    Also breaks down by is_factual.
    """
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix, accuracy_score

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load annotated data (ODS merged with metadata)
    df = pd.read_csv(annotated_csv)
    print(f"Loaded {len(df)} annotated claims from {annotated_csv}")

    # Load LLM results if provided
    if llm_results_csv and os.path.exists(llm_results_csv):
        llm_df = pd.read_csv(llm_results_csv)
        print(f"Loaded {len(llm_df)} LLM results from {llm_results_csv}")
        # Merge LLM results into df on claim text
        df = df.merge(
            llm_df[['claim', 'is_checkable_llm', 'is_checkable_reasoning']],
            on='claim', how='left'
        )
        has_llm = df['is_checkable_llm'].notna().sum() > 0
        if has_llm:
            failed = (df['is_checkable_llm'] == -1).sum()
            if failed > 0:
                print(f"Warning: {failed} LLM evaluations failed (marked -1)")
        print(f"LLM labels merged: {df['is_checkable_llm'].notna().sum()}/{len(df)}")
    else:
        has_llm = False

    # Build labeler columns dict
    labelers = {
        'Ruggero': 'is_checkable_ruggero',
        'Ilaria': 'is_checkable_ilaria',
    }
    if has_llm:
        labelers['LLM'] = 'is_checkable_llm'

    def compute_pairwise(subset, labelers_dict, label):
        """Compute pairwise agreement metrics for a subset of data."""
        results = []
        labeler_names = list(labelers_dict.keys())
        for i, name_a in enumerate(labeler_names):
            for name_b in labeler_names[i+1:]:
                col_a, col_b = labelers_dict[name_a], labelers_dict[name_b]
                pair_df = subset[[col_a, col_b]].dropna()
                # Exclude LLM failures
                if 'is_checkable_llm' in [col_a, col_b]:
                    llm_col = col_a if col_a == 'is_checkable_llm' else col_b
                    pair_df = pair_df[pair_df[llm_col] >= 0]

                n = len(pair_df)
                if n < 2:
                    continue

                vals_a = pair_df[col_a].astype(int).values
                vals_b = pair_df[col_b].astype(int).values

                kappa = cohen_kappa_score(vals_a, vals_b)
                acc = accuracy_score(vals_a, vals_b)
                cm = sk_confusion_matrix(vals_a, vals_b, labels=[0, 1])

                results.append({
                    'split': label,
                    'labeler_a': name_a,
                    'labeler_b': name_b,
                    'n': n,
                    'accuracy': acc,
                    'cohens_kappa': kappa,
                    'cm_00': cm[0, 0], 'cm_01': cm[0, 1],
                    'cm_10': cm[1, 0], 'cm_11': cm[1, 1],
                })
        return results

    all_results = []

    # Overall, calibration, and test splits
    for split_name, subset in [
        ('all', df),
        ('calibration', df[df['split'] == 'calibration']),
        ('test', df[df['split'] == 'test']),
    ]:
        all_results.extend(compute_pairwise(subset, labelers, split_name))

    # Breakdown by is_factual within test set
    test_df = df[df['split'] == 'test']
    for factual_val, factual_label in [(True, 'test_factual'), (False, 'test_nonfactual')]:
        subset = test_df[test_df['is_factual'] == factual_val]
        all_results.extend(compute_pairwise(subset, labelers, factual_label))

    results_df = pd.DataFrame(all_results)

    # Print results
    print("\n" + "="*70)
    print("INTER-LABELER AGREEMENT: is_checkable")
    print("="*70)
    for split_name in results_df['split'].unique():
        subset = results_df[results_df['split'] == split_name]
        print(f"\n--- {split_name.upper()} ---")
        for _, row in subset.iterrows():
            print(f"  {row['labeler_a']} vs {row['labeler_b']} (n={row['n']}):")
            print(f"    Accuracy: {row['accuracy']:.3f}")
            print(f"    Cohen's Kappa: {row['cohens_kappa']:.3f}")
            print(f"    Confusion Matrix: [[{row['cm_00']},{row['cm_01']}],[{row['cm_10']},{row['cm_11']}]]")

    # Save results
    results_file = os.path.join(output_dir, f"is_checkable_agreement_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

    # Save full merged data for reference
    full_file = os.path.join(output_dir, f"is_checkable_full_data_{timestamp}.csv")
    cols_to_save = [c for c in df.columns if c != 'article_text']
    df[cols_to_save].to_csv(full_file, index=False)
    print(f"Full data saved to {full_file}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Analyze labeled claims data")
    parser.add_argument('--input_dir', type=str, default='human_annotations/labelling_samples',
                        help='Directory containing labeled form files')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--include_machine', action='store_true', default=True,
                        help='Include machine evaluations in the analysis')
    parser.add_argument('--mode', type=str, default='quality', choices=['quality', 'is_checkable'],
                        help='Analysis mode: quality (5-criteria) or is_checkable (binary)')
    parser.add_argument('--annotated_csv', type=str,
                        default='data/samples/labeling_sample_200_annotated.csv',
                        help='Path to annotated CSV (for is_checkable mode)')
    parser.add_argument('--llm_results', type=str, default=None,
                        help='Path to LLM is_checkable results CSV')
    args = parser.parse_args()

    if args.mode == 'is_checkable':
        analyze_is_checkable_agreement(
            annotated_csv=args.annotated_csv,
            llm_results_csv=args.llm_results,
            output_dir=args.output_dir,
        )
        return

    # --- Original quality analysis below ---
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load labeled data
    df = load_labelled_data(args.input_dir, include_machine_evals=args.include_machine)

    # Set up timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Analyze inter-rater agreement
    print("\nCalculating inter-rater agreement...")
    agreement_df = analyze_inter_rater_agreement(df)

    # Save agreement results
    agreement_file = os.path.join(args.output_dir, f"inter_rater_agreement_{timestamp}.csv")
    agreement_df.to_csv(agreement_file, index=False)
    print(f"Inter-rater agreement results saved to {agreement_file}")

    # Print agreement by criterion and agreement type
    print("\nAgreement by criterion (Cohen's Kappa):")
    for agreement_type in agreement_df['agreement_type'].unique():
        subset = agreement_df[agreement_df['agreement_type'] == agreement_type]
        print(f"\n{agreement_type} Agreement:")
        print(subset[['criterion', 'avg_kappa']].to_string(index=False))

    # Calculate quality metrics
    df = calculate_overall_quality(df)

    # Compare human and machine quality assessments if available
    if args.include_machine and 'machine' in df['labeler_id'].unique():
        print("\nComparing human and machine evaluations...")
        comparison = compare_human_machine_quality(df)

        if comparison:
            comparison_file = os.path.join(args.output_dir, f"human_machine_comparison_{timestamp}.txt")
            with open(comparison_file, 'w') as f:
                f.write("Human-Machine Quality Comparison\n")
                f.write("===============================\n\n")
                f.write(f"Average difference in yes counts: {comparison['avg_yes_count_diff']:.2f}\n")
                f.write(f"Exact yes count agreement: {comparison['exact_yes_count_agreement']:.1f}%\n")
                f.write(f"Close yes count agreement (≤1 difference): {comparison['close_yes_count_agreement']:.1f}%\n")
                f.write(f"Quality level agreement: {comparison['quality_level_agreement']:.1f}%\n\n")
                f.write("Confusion Matrix (rows=human, columns=machine):\n")
                f.write(comparison['confusion_matrix'].to_string())
                f.write("\n\nConfusion Matrix Percentages:\n")
                f.write(comparison['confusion_matrix_pct'].round(1).to_string())
            print(f"Human-machine comparison saved to {comparison_file}")
            print(f"\nHuman-Machine Agreement:")
            print(f"  Quality level agreement: {comparison['quality_level_agreement']:.1f}%")
            print(f"  Exact yes count agreement: {comparison['exact_yes_count_agreement']:.1f}%")
            print(f"  Close yes count agreement (≤1 difference): {comparison['close_yes_count_agreement']:.1f}%")

    # Analyze quality by group
    human_df = df[df['source'] == 'human'].copy()
    quality_human_df = analyze_quality_by_group(human_df)
    quality_file = os.path.join(args.output_dir, f"quality_by_group_human_{timestamp}.csv")
    quality_human_df.to_csv(quality_file, index=False)
    print(f"\nHuman quality results saved to {quality_file}")
    print("\nQuality by model and method (Human evaluations):")
    print(quality_human_df[['group', 'avg_yes_count', 'high_quality_rate']].to_string(index=False))

    plot_quality_by_group(quality_human_df, args.output_dir, suffix="human")
    plot_criteria_by_group(quality_human_df, args.output_dir, suffix="human")

    if args.include_machine and 'machine' in df['labeler_id'].unique():
        machine_df = df[df['source'] == 'machine'].copy()
        quality_machine_df = analyze_quality_by_group(machine_df)
        quality_file = os.path.join(args.output_dir, f"quality_by_group_machine_{timestamp}.csv")
        quality_machine_df.to_csv(quality_file, index=False)
        print(f"\nMachine quality results saved to {quality_file}")
        plot_quality_by_group(quality_machine_df, args.output_dir, suffix="machine")
        plot_criteria_by_group(quality_machine_df, args.output_dir, suffix="machine")

    print("\nAnalyzing claim features associated with quality...")
    quality_stats, regression_summary = analyze_claim_features(human_df)
    stats_file = os.path.join(args.output_dir, f"quality_stats_{timestamp}.csv")
    quality_stats.to_csv(stats_file, index=False)
    print(f"Quality statistics saved to {stats_file}")

    if regression_summary is not None:
        regression_file = os.path.join(args.output_dir, f"regression_results_{timestamp}.txt")
        with open(regression_file, 'w') as f:
            f.write(str(regression_summary))
        print(f"Regression results saved to {regression_file}")

    full_output = os.path.join(args.output_dir, f"labeled_data_analyzed_{timestamp}.csv")
    df.to_csv(full_output, index=False)
    print(f"\nFull analyzed dataset saved to {full_output}")
    print("\nAnalysis complete!")


def plot_quality_by_group(quality_df, output_dir, suffix=""):
    """Create plots showing quality by model and method"""
    suffix_text = f"_{suffix}" if suffix else ""
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=1)
    
    # Plot average yes count by group
    sns.barplot(
        data=quality_df, 
        x='group', 
        y='avg_yes_count',
        ax=axes[0]
    )
    axes[0].set_title(f'Average Number of "Yes" Ratings by Model and Method{suffix_text}')
    axes[0].set_xlabel('Model and Method')
    axes[0].set_ylabel('Average Yes Count (out of 5)')
    axes[0].set_ylim(0, 5)
    axes[0].axhline(y=3, color='r', linestyle='--')  # Line at Medium quality threshold
    
    # Rotate x-axis labels for readability
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot high quality rate by group
    sns.barplot(
        data=quality_df, 
        x='group', 
        y='high_quality_rate',
        ax=axes[1]
    )
    axes[1].set_title(f'Proportion of High Quality Claims by Model and Method{suffix_text}')
    axes[1].set_xlabel('Model and Method')
    axes[1].set_ylabel('Proportion of High Quality Claims')
    axes[1].set_ylim(0, 1)
    
    # Rotate x-axis labels for readability
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(output_dir, f"quality_by_group{suffix_text}_{timestamp}.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Saved quality plot to {plot_file}")
    
    return plot_file


def plot_criteria_by_group(quality_df, output_dir, suffix=""):
    """Create plot showing performance on each criterion by group"""
    suffix_text = f"_{suffix}" if suffix else ""
    
    # Melt the dataframe to get criteria scores in long format
    criteria_cols = [col for col in quality_df.columns if col.endswith('_score_value_mean')]
    
    # Create a copy and rename columns for better display
    plot_df = quality_df.copy()
    for col in criteria_cols:
        new_name = col.replace('_score_value_mean', '')
        plot_df = plot_df.rename(columns={col: new_name})
    
    criteria_names = [col.replace('_score_value_mean', '') for col in criteria_cols]
    
    melted = pd.melt(
        plot_df,
        id_vars=['group'],
        value_vars=criteria_names,
        var_name='criterion',
        value_name='score'
    )
    
    # Create a figure
    plt.figure(figsize=(15, 10))
    
    # Create the grouped bar chart
    ax = sns.catplot(
        data=melted,
        x='group',
        y='score',
        hue='criterion',
        kind='bar',
        height=8,
        aspect=1.5
    )
    
    plt.title(f'Performance on Each Criterion by Model and Method{suffix_text}')
    plt.xlabel('Model and Method')
    plt.ylabel('Average Score (0-1)')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(output_dir, f"criteria_by_group{suffix_text}_{timestamp}.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Saved criteria plot to {plot_file}")
    
    return plot_file


def analyze_claim_features(df):
    """Analyze which features of claims are associated with higher quality"""
    # Calculate claim text length
    df['claim_length'] = df['synthetic_claim'].apply(len)
    df['claim_word_count'] = df['synthetic_claim'].apply(lambda x: len(str(x).split()))
    
    # Group by quality level
    quality_stats = df.groupby('calculated_quality').agg({
        'claim_length': ['mean', 'std', 'min', 'max'],
        'claim_word_count': ['mean', 'std', 'min', 'max'],
    }).reset_index()
    
    # Flatten the column multi-index
    quality_stats.columns = [
        '_'.join(col).strip('_') for col in quality_stats.columns.values
    ]
    
    # Run a logistic regression to see what factors predict high quality
    # Convert to binary outcome (high quality vs not)
    df['is_high_quality'] = (df['calculated_quality'] == 'High').astype(int)
    
    # Select features for the model
    features = ['claim_length', 'claim_word_count']
    factor_vars = ['model', 'method']
    
    # Create dummy variables for categorical predictors
    for var in factor_vars:
        dummies = pd.get_dummies(df[var], prefix=var, drop_first=True)
        for col in dummies.columns:
            features.append(col)
            df[col] = dummies[col]
    
    # Add a constant
    X = sm.add_constant(df[features])
    y = df['is_high_quality']
    
    # Fit the model
    model = sm.Logit(y, X)
    try:
        results = model.fit()
        regression_summary = results.summary()
    except:
        print("Warning: Could not fit logistic regression model")
        regression_summary = None
    
    return quality_stats, regression_summary


if __name__ == "__main__":
    main() 