import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings


def compute_row_stats(file):
    """
    Computes mean, standard deviation, and median for each row.
    """
    # Replace '-' with nan and convert all non-numeric values to nan
    file = file.replace('-', np.nan).apply(pd.to_numeric, errors='coerce')
    
    # Remove id column from computation
    numeric_cols = file.columns[1:]
    
    # Compute statistics for each row
    row_stds = file[numeric_cols].std(axis=1, skipna=True)
    row_means = file[numeric_cols].mean(axis=1, skipna=True)
    row_medians = file[numeric_cols].median(axis=1, skipna=True)
    
    # Create a dataframe with the computed statistics
    stats_df = pd.DataFrame({
        'ID': file['ID'],
        'mean': row_means,
        'std': row_stds,
        'median': row_medians
    })
    
    return stats_df


def compute_mean_differences(df, col1='mean_2', col2='mean_9', feature_1='2', feature_2='9'):
    """
    Computes the mean difference between specified mean columns in the dataframe.

    Args:
        df (pd.DataFrame): dataframe with mean columns.
        col1 (str): Name of the first column to compare.
        col2 (str): Name of the second column to compare.
        feature_1 (str): Identifier for the first feature set. 
        feature_2 (str): Identifier for the second feature set. 
    
    Returns:
        pd.DataFrame: dataframe with an additional column for the mean differences.

    """

    df[f'diff_{feature_1}vs{feature_2}'] = df[col1] - df[col2]
    return df

def compute_absolute_difference(df_1, df_2, substring_1='2LM', substring_2='7LM'):
    """
    Computes the absolute difference between corresponding columns in two dataframes

    Args:
        df_1 (pd.DataFrame): First dataframe.
        df_2 (pd.DataFrame): Second dataframe.
        substring_1 (str): Substring to replace in column names of the first dataframe.
        substring_2 (str): Substring to replace in column names of the second dataframe.
    
    Returns:
        pd.DataFrame: A dataframe containing the absolute differences between the two input dataframes.
    """

    df_1.columns = df_1.columns.str.replace(substring_1, 'LM', regex=False)
    df_2.columns = df_2.columns.str.replace(substring_2, 'LM', regex=False)

    id_column = df_1['ID']
    absolute_diff = np.abs(df_1.drop(columns=['ID'], errors='ignore') - df_2.drop(columns=['ID'], errors='ignore'))
    absolute_diff.insert(0, 'ID', id_column.values) 
    
    return absolute_diff



def absolute_mean_differences(df, col1='mean_2', col2='mean_9', feature_1='2', feature_2='9'):
    """
    Computes the absolute difference between specified mean columns in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with mean columns.
        col1 (str): Name of the first column to compare.
        col2 (str): Name of the second column to compare.
        feature_1 (str): Identifier for the first feature set. 
        feature_2 (str):Identifier for the second feature set. 

    Returns:
        pd.DataFrame: Dataframe with an additional column for the absolute differences.
    """
    
    df[f'abs_diff_{feature_1}vs{feature_2}'] = np.abs(df[col1] - df[col2])
    return df


def compute_binary_model_multiplicity(df_predictions, model_number, column='LB'):
    """
    Computes the multiplicity of the binary model predictions.
    """
    
    # convert all values to numeric, ignoring id column
    numeric_df = df_predictions[model_number][column].drop(columns=['ID']).apply(pd.to_numeric, errors='coerce')

    # compute S1, N, and S0
    S1 = numeric_df.sum(axis=1)
    N = numeric_df.notna().sum(axis=1)
    S0 = N - S1
    agreement = np.maximum(S1 / N, S0 / N)

    # add ID column to retain the original reference 
    model_multiplicity = pd.DataFrame({
        'ID': df_predictions[model_number][column]['ID'],
        'S1': S1,
        'S0': S0,
        'N': N,
        'Agreement': agreement
    })

    # minimum agreement value and its frequency 
    min_agreement = agreement.min()
    min_agreement_count = agreement[agreement == min_agreement].count()

    # output results
    print(model_multiplicity.head())
    print("Minimum agreement value: ", min_agreement)
    print("Minimum agreement value count: ", min_agreement_count)
    
    return model_multiplicity, min_agreement, min_agreement_count

def feature_dependency_models(features_B, features_A, model_type="linear"):
    """
    Train models (linear regression or random forest) to predict features in B using features in A as predictors.
    
    Args:
        features_B (pd.DataFrame): Dataframe containing the target features to be predicted B.
        features_A (pd.DataFrame): Datafrane containing the predictor features A.
        model_type (str): Type of model to use ('linear' or 'random_forest').
    
    Returns:
        dict: A dictionary where the keys are the feature names from B and the values are trained models 
              that predict the corresponding feature using features in A.
    """
    
    # Suppress warnings
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    models = {}
    for feature in features_B.columns:
        y = features_B[feature]  # Target variable

        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "random_forest":
            model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Choose 'linear' or 'random_forest'.")
            # Store the trained model
        
        model.fit(features_A, y)
        models[feature] = model  
    return models


def compute_total_residuals(target_features, predictor_data, models_for_B, id_dataset):
    """
    Compute the total residuals for each individual, with each feature residual standardized by its standard deviation.

    Parameters:
        target_features (pd.DataFrame): DataFrame containing the actual values of the target features for individuals.
        predictor_data (pd.DataFrame): DataFrame containing the input predictor features for individuals.
        models_for_B (dict): Dictionary where keys are feature names and values are trained models predicting those features.
        id_dataset (list): List of unique IDs corresponding to individuals in the dataset.

    Returns:
        pd.DataFrame: A DataFrame containing individual IDs and their corresponding total standardized residuals.
    """

    feature_std = target_features.std()

    total_residuals = []

    for i in range(len(target_features)):
        individual_residual = 0
        print(f"Processing individual {i + 1}/{len(target_features)}")

        for feature in target_features.columns:
            actual_value = target_features.iloc[i][feature]

            # predict the value for the current feature using the corresponding model
            x_input = predictor_data.iloc[[i]]
            predicted_value = models_for_B[feature].predict(x_input)[0]

            # unstandardized residual 
            residual = abs(actual_value - predicted_value)

            # standardized residual
            standardized_residual = residual / feature_std[feature]

            # add to total residual for individual
            individual_residual += standardized_residual
            
        total_residuals.append(individual_residual)

    # Dataframe with IDs and corresponding total residuals
    total_residuals_df = pd.DataFrame({
        'id_dataset': id_dataset,
        'total_residual': total_residuals
    })

    return total_residuals_df

