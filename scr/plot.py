import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from scipy.stats import spearmanr
from utils import *
from analysis import *

import seaborn as sns

def apply_plot_styling():
    """Applies consistent styling to plots."""
    plt.grid(color='#d3d3d3', linestyle='--', linewidth=0.7)
    plt.gca().set_facecolor('#f9f9f9')
    sns.despine(top=True, right=True)


def plot_histogram(data, title, color, ax, bins, x_limit, y_limit):
    """Plots a single histogram."""

    sns.histplot(data, kde=False, color=color, ax=ax, element='bars', bins=bins)
    ax.set_title(title)
    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Frequency')
    ax.set_xlim([0, x_limit])
    ax.set_ylim([0, y_limit])


def plot_mean_distribution(df, title, color, max_y):
    """Plots the distribution of means."""

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['mean'], kde=False, color=color, ax=ax, element='bars', bins=50, stat="count")
    
    ax.set_title(title, fontsize=14, fontweight='normal', color='#333333')
    ax.set_xlabel('Mean for each data point across all 1000 models', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=max_y)
    ax.set_facecolor('#f7f7f7')
    
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    return fig, ax


def plot_multiplicity(all_std, model_name, axes, bins, max_std_value, y_limit, features, colors):
    """Plots model multiplicity for a single model (specific model type and feature set)."""

    titles = [f'Model Multiplicity for {feat}{model_name} Model' for feat in features]
    
    for i, feat in enumerate(features):
        plot_histogram(
            data=all_std[f'std_{feat}'],
            title=titles[i],
            color=colors[i],
            ax=axes[i],
            bins=bins,
            x_limit=max_std_value,
            y_limit=y_limit
        )

def plot_all_multiplicity(model_data, features, colors):
    """Plots model multiplicity for all feature set variations of a specific model type.
    Displays a single figure containing subplots, where each subplot represents the 
    distribution of standard deviations (model multiplicity) for a different feature set."""


    for model_name, all_std in model_data.items():
        max_std_value, bins, y_limit = calculate_limits(all_std, features)
        
        fig, axes = plt.subplots(1, len(features), figsize=(15, 5))
        axes = np.array(axes).flatten()
        
        plot_multiplicity(all_std, model_name, axes, bins, max_std_value, y_limit, features, colors)
        
        plt.suptitle(f"Model Multiplicity of {model_name} Models", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # layout adjustments
        plt.show()


def plot_model_distributions(data_dict, model_name, feature_labels, colors):
    """Plots mean distributions for each model."""

    max_y = calculate_max_y_for_features(data_dict, feature_labels)
    
    for feature, color in zip(feature_labels, colors):
        title = f'Distribution of Means for {feature}{model_name} Models'
        plot_mean_distribution(data_dict[feature], title, color, max_y)
        plt.show()


def plot_all_mean_distributions(means_df, suffixes, model_types, colors):
    """
    Plots the distribution of mean values for each model.
    
    Args:
        - means_df (pd.DataFrame): dataframe containing mean values for each model.
        - suffixes (list of str): List of suffixes used to identify relevant columns in means_df.
        - model_types (str): Name of the model type to include in plot titles.
        - colors (list): List of colors for each plot.
    
    Returns:
        None
    """
    # close any open plots
    plt.close('all')

    # Extract data for columns with the specified suffixes
    data_dict = {f'mean_{suffix}': means_df[f'mean_{suffix}'] for suffix in suffixes if f'mean_{suffix}' in means_df.columns} 

    # helper function to calculate a consistent max_y for all plots
    max_y = calculate_max_y_for_features(data_dict, list(data_dict.keys()))
    
    # Plot each mean distribution 
    for prefix, color in zip(suffixes, colors):
        column = f'mean_{prefix}'
        if column in means_df.columns:
            single_column_df = means_df[[column]].rename(columns={column: 'mean'})
            title = f'Distribution of Means for {prefix}{model_types} Models'
            fig, ax = plot_mean_distribution(single_column_df, title, color, max_y)
            plt.show()
            plt.close(fig)  


def plot_difference_histogram(data, title, color, bins, max_diff, min_diff):
    """
    Plots the histogram for the distribution of mean differences.
    
    Args:
        - data (pd.Series or np.ndarray): Data containing differences of means.
        - title (str): Title for the plot.
        - color (str): Color for the histogram.
        - bins (np.ndarray): Array of bin edges for the histogram.
        - max_diff (float): Maximum x-limit for the histogram.
        - min_diff (float): Minimum x-limit for the histogram.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data, kde=False, color=color, ax=ax, element='bars', bins=bins, legend=False)
    #ax.legend().remove()
    ax.set_title(title)
    ax.set_xlabel('Differences of Means')
    ax.set_ylabel('Frequency')
    ax.set_xlim(min_diff, max_diff)
    plt.show()


def plot_all_differences_histograms(mean_diffs, model_types, differences, plot_description_comp, colors):
    """
    Plots histograms for all specified differences across various model types.

    Args:
        - mean_diffs (dict): Dictionary containing mean differences for each model type.
        - model_types (list): List of model types to plot.
        - differences (list): List of differences to plot.
        - plot_description_comp (list): List of descriptions for each difference.
        - colors (list): List of colors for each plot.

    Returns:
        None
    """

    for model_type, color in zip(model_types, colors):
        if model_type not in mean_diffs:
            print(f"Warning: Model type '{model_type}' not found in mean_diffs.")
            continue  # skip if model type not found

        for diff, description in zip(differences, plot_description_comp):
            if diff not in mean_diffs[model_type]:
                print(f"Warning: Difference '{diff}' not found for model type '{model_type}'.")
                continue  # skip if difference not found
            
            # retrieve data for the current difference
            data = mean_diffs[model_type][diff]
            
            # convert to numpy array if data is a pandas dataframe or series
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                data = data.squeeze().to_numpy()

            if data.size == 0:
                print(f"No data available for {model_type} - {description}. Skipping...")
                continue
            
            max_diff, min_diff = data.max(), data.min()
            if max_diff == min_diff:  # to avoid issues with np.arange on identical min/max
                print(f"Warning: min and max are the same for {model_type} - {description}. Adjusting bins.")
                max_diff += 1e-6  

            bins = np.arange(min_diff, max_diff, 0.01)

            plot_title = f'Differences of Means: {description} ({model_type} Models)'
            
            plot_difference_histogram(
                data,
                plot_title,
                color,
                bins=bins,
                max_diff=max_diff,
                min_diff=min_diff
            )


def plot_ccdf(data, column, diff_type, color, threshold, feature_pair=None, model_type=None):
    """
    Function to plot a single CCDF.
 """

   # title for the plots
    if diff_type == "mean_of_abs_diff":
        title = "Mean Absolute Differences:"
        x_label = "Mean Absolute Differences"
    elif diff_type == "abs_diff_of_means":
        title = "Absolute Differences of Means:"
        x_label = "Absolute Differences of Means"
    else:
        raise ValueError("Invalid diff_type. Use 'mean_of_abs_diff' or 'abs_diff_of_means'.")
    
    if feature_pair:
        title += f" {feature_pair}"
    if model_type:
        title += f" {model_type} Models"
    

    data = data.dropna(subset=[column])
    # print(data.head())
    sorted_data = data[column].sort_values(ascending=False)

    ccdf = sorted_data.reset_index(drop=True).reset_index()
    ccdf.columns = ['Rank', 'Relative difference']
    ccdf['Percentage'] = (ccdf['Rank'] / len(ccdf))
    

    percentage = (data[column] >= threshold).mean() * 100

    plt.figure(figsize=(8, 6))
    plt.plot(ccdf['Relative difference'], ccdf['Percentage'], color=color)
    plt.axvline(x=threshold, color='pink', linestyle='--',label=f'{threshold} threshold', ymin=0, ymax=1)
    plt.ylabel('Proportion of Individuals')  
    # plt.xlim( 0, max(ccdf['Relative difference']))
    plt.xlabel(x_label)
    plt.ylim(0, 1)
    plt.legend()
    plt.title(f'CCDF of {title}')
    plt.show()
    print(f'Percentage of points with at least {threshold} absolute difference in means: {percentage:.2f}%\n')


def plot_model_ccdf(diffs, model_type, diff_type, feature_pairs, colors, thresholds,):
    """
    Plots multiple CCDFs with different thresholds
    for the absolute mean differences/mean of absolute differencenes between two models.
    """

    for feature_pair, color, threshold in zip(feature_pairs, colors, thresholds):

        feature, title = feature_pair
        plot_ccdf(diffs[model_type],column=feature,diff_type=diff_type,color=color,threshold=threshold,
                  feature_pair=title, model_type=model_type)


def correlation_analysis(abs_diff_n, confidence_df, comparison_label='2LP vs. 7LP', index='2vs7', confidence_label='5LP Average Confidence'):
    """
    Plots the relationship between the mean absolute differences and the average confidence.
    """
    merged_df = pd.merge(abs_diff_n, confidence_df, on='ID')
    overall_spearman_corr, _ = spearmanr(merged_df[f'abs_diff_{index}'], merged_df['Average_Confidence'])
    
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        x=merged_df[f'abs_diff_{index}'], 
        y=merged_df['Average_Confidence'], 
        color='#3498db',  
        s=30, 
        marker='.'
    )

    plt.title(
        f'Relationship between Absolute Mean Differences ({comparison_label}) and {confidence_label}\n'
        f'Correlation Coefficient: {overall_spearman_corr:.4f}', 
        fontsize=10, fontweight='normal', color='black', pad=15
    )
    plt.xlabel(f'Absolute Mean Difference ({comparison_label})', fontsize=9)
    plt.ylabel(confidence_label, fontsize=10)
    plt.grid(color='#e1e1e1', linestyle='--', linewidth=0.5)



    sns.despine(top=True, right=True)
    plt.gca().set_facecolor('#f7f7f7')
    plt.tight_layout()
    plt.show()

    return merged_df


def multiplicity_diff_corr_plot(abs_diff_n, multiplicity_df, comparison_label, index, multiplicity_label, model_type, base_model):
    merged_df = pd.merge(abs_diff_n, multiplicity_df, on='ID')
    overall_spearman_corr, _ = spearmanr(merged_df[f'abs_diff_{index}'], merged_df[multiplicity_label])
    print("Correlation coefficient:", overall_spearman_corr)

    plt.figure(figsize=(6, 4))
    plt.scatter(merged_df[f'abs_diff_{index}'], merged_df[multiplicity_label], s=20, marker='.')
    plt.xlabel(f'Mean of Absolute Differences ({comparison_label})')
    plt.ylabel(f'{base_model}{model_type} Model Multiplicity (Measure: {multiplicity_label.capitalize()})') 
    plt.title(f'Relationship between Mean of Absolute Differences ({comparison_label}) and Model Multiplicity ({base_model}{model_type}) \nCorrelation Coefficient={overall_spearman_corr:.4f}', 
              fontsize=9)

    apply_plot_styling()
    plt.tight_layout()
    plt.show()

def correl_multiplicity_differences(df_predictions, combined_mean_of_abs_diff, model_type, std, abs_diff_keys, comparison_labels, primary_model, dynamic_base_model=False):
    """
    Computes and visualizes the relationship between model multiplicity and mean of absolute differences.
    """

    for abs_diff_key, comparison_label in zip(abs_diff_keys, comparison_labels):
        base_model = abs_diff_key.split('vs')[1] if dynamic_base_model else primary_model
        print(f"Correlation between Model Multiplicity ({base_model}{model_type}) and Mean of Absolute Differences ({comparison_label})")
        
        mod_multiplicity = to_dataframe_with_id(compute_row_stats(df_predictions[base_model][model_type])[std], std)
        abs_diff_df = to_dataframe_with_id(combined_mean_of_abs_diff[model_type][f'abs_diff_{abs_diff_key}'], f'abs_diff_{abs_diff_key}')

        multiplicity_diff_corr_plot(
            abs_diff_n=abs_diff_df, 
            multiplicity_df=mod_multiplicity,  
            comparison_label=comparison_label, 
            index=abs_diff_key, 
            model_type=model_type,
            base_model=base_model,
            multiplicity_label=std 
        )


def correl_agreement_differences(model_multiplicity, combined_mean_of_abs_diff_df, model_type, base_model, comparison_key):
    """
    Computes and visualizes the relationship between model agreement and absolute differences.
    """

    agreement_col = model_multiplicity['Agreement']
    abs_diff_col = combined_mean_of_abs_diff_df[model_type][f'abs_diff_{comparison_key}']
    correlation = agreement_col.corr(abs_diff_col)
    print(f"Correlation: {correlation:.4f}")

    plt.figure(figsize=(6, 4))
    plt.scatter(
        agreement_col,
        abs_diff_col,
        color='#1f77b4',  
        alpha=0.9,        
        s=10              
    )

    plt.xlabel(f'Mean of Absolute Differences: 9{model_type} vs. {base_model}{model_type}', fontsize=8)
    plt.ylabel(f'Model Multiplicity of {base_model}{model_type} Models (Measure: Agreement)', fontsize=8)
    plt.title(
        f'Relationship between Mean of Absolute Differences (9{model_type} vs. {base_model}{model_type}) \n'
        f'and Model Multiplicity ({base_model}{model_type}) \n'
        f'Correlation Coefficient: {correlation:.4f}',
        fontsize=11, pad=15
    )

    plt.grid(color='#e1e1e1', linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor('#f9f9f9')
    plt.tight_layout()

    plt.show()

    return correlation


def plot_residuals_vs_differences(mean_diff, total_residual, label_1='Mean Difference', color='#3498db', alpha=0.8, s=30,log_scale=False):
    """
    Plots the relationship between mean differences and total residuals.
    """

    spearman_corr_1, _ = spearmanr(total_residual, mean_diff)

    # Title
    title = f"Relationship between {label_1} and Total Residual\nCorrelation Coefficient: {spearman_corr_1:.4f}"

    plt.figure(figsize=(6, 4))
    plt.scatter(mean_diff, total_residual, alpha=alpha, color=color, s=s, marker='.')

    if log_scale:
        plt.yscale('log')
    
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    plt.ticklabel_format(style='plain', axis='y')

    print(1)
    plt.title(title, fontsize=10, fontweight='normal', color='black', pad=15)
    plt.xlabel(label_1, fontsize=9)  # x-axis label
    plt.ylabel("Total Residual", fontsize=10)  # y-axis label
    plt.grid(color='#e1e1e1', linestyle='--', linewidth=0.5)
    plt.gca().set_facecolor('#f7f7f7')

    plt.tight_layout()
    plt.show()

def plot_std_histogram(std, std_all, title, xlabel='Standard Deviations', ylabel='Frequency', bins=20, color='skyblue'):
    """
    Plots a histogram of standard deviations.
    
    Args:
    - std: List or array of standard deviations.
    - std_all: The combined standard deviation value, displayed as a vertical line.
    - title: The title of the plot.
    - xlabel: Label for the x-axis (default: 'Standard Deviations').
    - ylabel: Label for the y-axis (default: 'Frequency').
    - bins: Number of bins for the histogram (default: 20).
    - color: Color of the bars in the histogram (default: 'skyblue').
    """
   
    plt.hist(std, bins=bins, color=color, density=False)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.axvline(std_all, color='limegreen', linestyle='dashed', linewidth=2, label=f'Combined STD: {std_all:.3f}')
    
    plt.xlim(0, max(max(std), std_all) + 0.05)
    
    plt.legend()
    
    plt.show()
