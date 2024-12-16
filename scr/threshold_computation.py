import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def calculate_quantile_threshold(df, quantile_level=70, replace_missing="-"):
    """
    Computes the threshold value for a specified quantile
    (e.g., the 70th percentile)
    from the data in dataframe.

    Parameters:
    - df: DataFrame containing the probabilities
    - quantile_level: The percentile level to calculate
    the threshold for (default is 70)
    - replace_missing: The value used to replace missing
    values (default is '-')

    Returns:
    - threshold: The threshold value for the 70th percentile

    """

    # Convert the DataFrame to a 1D array (flatten) and remove missing values if needed
    data_values = df.replace(
        replace_missing, np.nan
    ).values.flatten()  

    # Remove NaN values (if there are any) and convert to float
    data_values = data_values[~pd.isna(data_values)].astype(float)

    # Sort the data values in ascending order
    data_values = np.sort(data_values)

    # Calculate the threshold for the specified quantile
    threshold = np.percentile(data_values, quantile_level)

    return threshold



def optimize_threshold(features, target, model, n_splits=5, optimization_function='geometric_mean', constrained=True, random_state=42):
    """
    Optimizes threshold based on an optimization function (like geometric mean) and an optional TNR >= TPR constraint.

    Args:
    - features (DataFrame,): Input features
    - target (Series): Target variable
    - model (classifier model): LogisticRegression or RandomForestClassifier
    - n_splits (int): Number of cross-validation splits
    - optimization_function (str): The function to optimize, e.g., 'geometric_mean' or 'f1'
    - constrained (bool): If True applies the TNR >= TPR constraint

    Returns:
    - float, optimized threshold based on cross-validation
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    thresholds = []

    for train_index, test_index in skf.split(features, target):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve and thresholds
        fprs, tprs, thrs = roc_curve(y_test, y_pred_proba)
        tnrs = 1 - fprs  # Calculate True Negative Rates (TNR)

        # Calculate optimization score for each threshold
        scores = []
        for tpr, tnr, threshold in zip(tprs, tnrs, thrs):
             # Apply TNR >= TPR constraint if specified
            if constrained and tpr > tnr:
                continue

            # Optimization based on the specified function
            if optimization_function == 'geometric_mean':
                score = np.sqrt(tpr * tnr)  # Geometric Mean
            elif optimization_function == 'f1':
                precision = tpr / (tpr + fprs[thrs == threshold][0]) if tpr + fprs[thrs == threshold][0] > 0 else 0
                score = 2 * (precision * tpr) / (precision + tpr) if precision + tpr > 0 else 0
            else:
                raise ValueError("Unsupported optimization function. Choose 'geometric_mean' or 'f1'.")

            scores.append((score, threshold))

        # Get the threshold with the maximum score
        if scores:
            best_threshold = max(scores, key=lambda x: x[0])[1]
            thresholds.append(best_threshold)

    # Return the average threshold across folds
    return np.mean(thresholds) if thresholds else 0.5  # Default to 0.5 if no valid thresholds found


def calculate_optimized_threshold(features, target, max_iter=1000, optimization_function='geometric_mean', constrained=True):
    """
    Computes the optimal threshold for both Logistic Regression and Random Forest models
    and averages their results, respecting TPR >= TNR constraint.

    Args:
    - features (DataFrame): Input features
    - target (Series): Target variable
    - max_iter (int): Maximum number of iterations for Logistic Regression
    - optimization_function (str): The function to optimize ('geometric_mean' or 'f1')
    - constrained (bool): If True applies the TPR >= TNR constraint

    Returns:
    - float, averaged optimal threshold across models
    """
    models = [
        LogisticRegression(max_iter=max_iter, class_weight='balanced'),
        RandomForestClassifier(n_estimators=200, class_weight='balanced',random_state=42)
    ]

    optimal_thresholds = [
        optimize_threshold(features, target, model, optimization_function=optimization_function, constrained=constrained, random_state=42)
        for model in models
    ]
    mean_optimal_threshold = np.mean(optimal_thresholds)
    return mean_optimal_threshold