import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from threshold_computation import  *
from utils import *

def train_and_evaluate_test(
        dataset: str,
        features: pd.DataFrame,
        target: pd.Series,
        id_column: pd.Series,
        num_iterations: int,
        test_size: float,
        max_iter: int,
        prefix: str,
        model_type: str = 'logistic',
        threshold: float = None,
        quantile_level: int = 70,
        mode: str = 'classification',
        output_type: str = 'probability'
    ):
    """
    Trains and evaluates a model, calculating either confidence metrics 
    or classification results.

    Parameters:
    - features (pd.DataFrame): Dataframe of feature columns.
    - target (pd.Series): Series with target variable.
    - id_column (pd.Series): Series with unique identifier for each row.
    - num_iterations (int): Number of models to train.
    - test_size (float): Proportion of data for the test set.
    - max_iter (int): Maximum iterations for logistic regression.
    - prefix (str): Prefix for model name in outputs.
    - model_type (str): Model type ('logistic' or 'random_forest').
    - threshold (float, optional): Manual threshold; calculated if None.
    - quantile_level (int): Quantile level for threshold calculation.
    - mode (str): 'confidence' for confidence metrics or 'classification'.
    - output_type (str): 'binary' or 'probability' output for classification.

    Outputs:
    - Saves evaluation results to CSV and prints average performance metrics.
    """
    
    # check for valid inputs
    if mode not in ['confidence', 'classification']:
        raise ValueError("Invalid mode. Choose 'confidence' or 'classification'.")
    if output_type not in ['binary', 'probability']:
        raise ValueError("Invalid output_type. Choose 'binary' or 'probability'.")

    performance_metrics = []
    confidence_sums = {id_: 0 for id_ in id_column.tolist()}
    prediction_counts = {id_: 0 for id_ in id_column.tolist()}
    thresholds = [] if mode == 'classification' else None
    all_results = {f"{prefix}M{i}": {} for i in range(num_iterations)}
    suffix = "B" if output_type == 'binary' else "P"

    # dataframe initialization
    new_df = pd.DataFrame({'ID': id_column.tolist()})

    # Training and evaluation loop
    for i in range(num_iterations):
        print(f"Model {i}")
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            features, target, id_column, test_size=test_size, random_state=i
        )
        
        model = initialize_model(model_type, max_iter=max_iter, random_state=i)
        model.fit(X_train, y_train)
        # Predict probabilities and clip extreme values
        raw_probabilities = model.predict_proba(X_test)[:, 1]
        clip_probabilities = np.clip(
            model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6
        )

        if mode == 'confidence':
            confidence_scores = np.abs(clip_probabilities - 0.5)

            for idx, id_ in enumerate(id_test):
                if not pd.isna(y_test.iloc[idx]) and not pd.isna(confidence_scores[idx]):
                    # Accumulate the confidence score and count for this ID
                    confidence_sums[id_] += confidence_scores[idx]
                    prediction_counts[id_] += 1

        elif mode == 'classification':
            actual_threshold = threshold or calculate_quantile_threshold(
                pd.DataFrame(clip_probabilities), quantile_level=quantile_level
            )
            thresholds.append(actual_threshold)

            # Store threshold for saving
            y_pred_binary = (clip_probabilities >= actual_threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred_binary)
            performance_metrics.append(accuracy)

            # Store results based on output type
            results = dict(
                zip(
                    id_test, 
                    y_pred_binary if output_type == 'binary' 
                    else clip_probabilities
                )
            )
            all_results[f"{prefix}M{i}"].update(results)

    if mode == 'classification':
        final_threshold = np.mean(thresholds) if threshold is None else threshold
        dynamic_threshold = threshold is None 


    # call appropriate helper function to save results
    if mode == 'confidence':
        save_confidence_results(
            new_df, dataset, prefix, suffix, confidence_sums, 
            prediction_counts, num_iterations
        )
    elif mode == 'classification':
        save_classification_results(
            new_df, dataset, prefix, suffix, performance_metrics, 
            all_results, num_iterations, final_threshold, dynamic_threshold
        )
