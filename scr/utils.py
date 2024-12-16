import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from datetime import datetime


def remove_id_column(df):
    """Removes the first ID column from the dataframe."""
    return df.iloc[:, 1:]

#######

def to_dataframe_with_id(series_or_df, column_name):
    """
    Converts a Series to a DataFrame with a specified column name.
    Adds an 'ID' column based on the length of the DataFrame.
    """
    df = series_or_df.to_frame(name=column_name) if isinstance(series_or_df, pd.Series) else series_or_df
    df['ID'] = range(1, len(df) + 1)
    return df



def initialize_model(model_type: str, max_iter: int, random_state: int):
    """Initialize a model based on the model type."""
    if model_type == 'logistic':
        return LogisticRegression(max_iter=max_iter)
    elif model_type == 'random_forest':
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    else:
        raise ValueError("Unsupported model type. Choose 'logistic' or 'random_forest'.")


def calculate_limits(combined_std, features):
    """Calculates histogram limits dynamically based on features."""
    max_std_value = max(combined_std[f'std_{feat}'].max() for feat in features)
    bins = np.arange(0, max_std_value + 0.01, 0.01)
    
    y_limit = 0
    for feat in features:
        hist, _ = np.histogram(combined_std[f'std_{feat}'], bins=bins)
        y_limit = max(y_limit, hist.max())
    
    y_limit *= 1.1  # Add margin to y-limit
    return max_std_value, bins, y_limit



def calculate_max_y(data):
    """Calculate the maximum y-axis value for histograms for a given series without plotting."""
    counts, _ = np.histogram(data, bins=50)  # Calculate histogram counts
    return counts.max()  # Return the maximum count for y-axis limit

def calculate_max_y_for_features(data_dict, features):
    """Calculate maximum y-axis value across multiple features without plotting."""
    return max(calculate_max_y(data_dict[feature]) for feature in features)
    

def save_classification_results(new_df, dataset, prefix, suffix, performance_metrics, all_results, num_iterations, final_threshold, dynamic_threshold=False):
    """
    Processes and saves classification mode results to CSV and performance metrics to a text file.
    
    Parameters:
    - new_df (pd.DataFrame): DataFrame for storing results.
    - dataset (str): Dataset name used for file naming.
    - prefix (str): Prefix for file naming.
    - suffix (str): Suffix for file naming ('B' for binary or 'P' for probability).
    - performance_metrics (list): List of accuracy scores from each iteration.
    - all_results (dict): Dictionary of all model predictions across iterations.
    """

    # Generate timestamp without year
    timestamp = datetime.now().strftime('%m%d_%H%M')

    # Map results for each model iteration to DataFrame columns
    for i in range(len(performance_metrics)):
        col_name = f"{prefix}M{i}"
        new_df[col_name] = new_df['ID'].map(all_results[col_name]).fillna('-')
    
    # Save the DataFrame to a CSV file
    new_df.to_csv(f"../predictions/{dataset}_predictions_{prefix}{suffix}_{timestamp}.csv", index=False)
    
    # Calculate mean and standard deviation of accuracy
    mean_accuracy = np.nanmean(performance_metrics)
    std_accuracy = np.nanstd(performance_metrics)
    print(f"Mean Accuracy ({prefix.upper()}{suffix}): {mean_accuracy:.4f}")
    print(f"Standard Deviation of Accuracy ({prefix}{suffix}): {std_accuracy:.4f}")
   
    # Print the threshold used, dynamically calculated or fixed
    threshold_type = "Mean Threshold" if dynamic_threshold else "Threshold"
    print(f"{threshold_type} used: {final_threshold:.4f}")


# Append metrics with timestamp to performance file in predictions folder
    with open("../predictions/performance_metrics.txt", "a") as f:
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # Add timestamp
        f.write(f"Dataset: {dataset.capitalize()}, Number of Models: {num_iterations}\n")
        f.write(f"{threshold_type}: {final_threshold:.4f}\n")
        f.write(f"Mean Accuracy ({prefix.upper()}{suffix}): {mean_accuracy:.4f}\n")
        f.write(f"Standard Deviation of Accuracy ({prefix}{suffix}): {std_accuracy:.4f}\n")
        f.write("\n")  # Add a newline for separation


def save_confidence_results(new_df, dataset, prefix, suffix, confidence_sums, prediction_counts, num_iterations):
    """
    Processes and saves confidence results to CSV and logs performance metrics.

    Parameters:
    - new_df (pd.DataFrame): DataFrame with sample IDs.
    - dataset (str): Name of the dataset.
    - prefix (str): Prefix for naming.
    - suffix (str): Suffix ('B' or 'P') for output type.
    - confidence_sums (dict): Cumulative confidence scores for each ID.
    - prediction_counts (dict): Number of predictions for each ID.
    - num_iterations (int): Number of iterations for training.
    """

    # Calculate average confidence per ID
    avg_confidence = {
        id_: confidence_sums[id_] / prediction_counts[id_] if prediction_counts[id_] > 0 else '-'
        for id_ in new_df['ID']
    }
    new_df['Average_Confidence'] = new_df['ID'].map(avg_confidence).fillna('-')
    
    # Save to CSV
    timestamp = datetime.now().strftime('%m%d_%H%M')
    new_df.to_csv(f"../predictions/{dataset}_confidence_{prefix}{suffix}_{timestamp}.csv", index=False)

    # Calculate and log mean and standard deviation of confidence
    confidences = [value for value in avg_confidence.values() if isinstance(value, (int, float))]
    mean_confidence = np.nanmean(confidences)
    std_confidence = np.nanstd(confidences)
    print(f"Mean Confidence ({prefix.upper()}{suffix}): {mean_confidence:.4f}")
    print(f"Standard Deviation of Confidence ({prefix.upper()}{suffix}): {std_confidence:.4f}")

    # Append metrics to performance file
    with open("../predictions/performance_metrics.txt", "a") as f:
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset.capitalize()}, Number of Models: {num_iterations}\n")
        f.write(f"Mean Confidence ({prefix.upper()}{suffix}): {mean_confidence:.4f}\n")
        f.write(f"Standard Deviation of Confidence ({prefix.upper()}{suffix}): {std_confidence:.4f}\n")
        f.write("\n")
