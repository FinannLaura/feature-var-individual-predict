import numpy as np
import pandas as pd
from utils import  *
from threshold_computation import  *
from training import  *
from data_preprocessing import  *
from threshold_computation import *

# Parameters for training
num_iterations = 2
test_size = 0.2
max_iter = 1000
optimization_function = "geometric_mean"

# Load datasets 
def load_compas_data():
    compas_id, compas_target, two_features, seven_features, nine_features, nine_minus_two, five_features = preprocess_compas_data()
    return compas_id, compas_target, two_features, seven_features, nine_features, nine_minus_two, five_features


def load_credit_data():
    id_credit, target_credit, df_9, df_20, df_11 = preprocess_credit_data()
    return id_credit, target_credit, df_9, df_20, df_11 

def load_unemployment_data():
    id_unemployment, target_unemployment,df_30, df_22, df_9, df_8, df_21 = preprocess_unemployment_data()
    return id_unemployment, target_unemployment, df_30, df_22, df_9, df_8, df_21

# Calculate mean threshold based on feature sets
def calculate_mean_threshold(*datasets):
    thresholds = [
        calculate_optimized_threshold(features=df, target=target, optimization_function=optimization_function)
        for df, target in datasets
    ]
    mean_threshold = np.mean(thresholds)
    print(f"Mean threshold calculated from datasets: {mean_threshold}")
    return mean_threshold

# Training function for each dataset with selectable mode, output type, model type, and threshold
def train_dataset(dataset_name, features, target, id_column, threshold, prefix, mode="classification", output_type="probability", model_type="logistic"):
    train_and_evaluate_test(
        dataset=dataset_name,
        features=features,
        target=target,
        id_column=id_column,
        num_iterations=num_iterations,
        test_size=test_size,
        max_iter=max_iter,
        prefix=prefix,
        model_type=model_type, 
        threshold=threshold,
        mode=mode,
        output_type=output_type
    )

def main():
    # User selection for dataset, mode, output type, model type, and threshold
    choice = input("Select dataset to train on (compas, credit, unemployment): ").strip().lower()
    mode = input("Select mode (classification or confidence): ").strip().lower()
    output_type = input("Select output type (probability or binary): ").strip().lower()
    model_type = input("Select model type (logistic or random_forest): ").strip().lower()
    
    if mode not in ["classification", "confidence"]:
        print("Invalid mode selected. Exiting.")
        return


    if choice == "compas":
        compas_id, compas_target, two_features, seven_features, nine_features, nine_minus_two, five_features = load_compas_data()
        feature_choice = input("Select feature set for COMPAS (two, seven, nine, nine_minus_two): ").strip().lower()
        prefix = input("Enter model type for saving results (i.e. 2L, 7L, 9L, 2R, 7R, 9R): ").strip()
    
        
        # Map user choice to the appropriate feature set
        if feature_choice == "two":
            selected_features = two_features
        elif feature_choice == "seven":
            selected_features = seven_features
        elif feature_choice == "nine":
            selected_features = nine_features
        elif feature_choice == "five":
            selected_features = five_features
        elif feature_choice == "nine_minus_two":
            selected_features = nine_minus_two
        else:
            print("Invalid feature set choice. Defaulting to 'two_features'.")
            selected_features = two_features


        train_dataset(
            dataset_name="compas",
            features=selected_features,
            target=compas_target,
            id_column=compas_id,
            threshold=None,
            prefix=prefix,
            mode=mode,
            output_type=output_type,
            model_type=model_type
        )

    if choice == "credit":
        id_credit, target_credit, df_9, df_20, df_11  = load_credit_data()
        feature_choice = input("Choose feature set for Credit (e.g., 9 or 20): ").strip()
        prefix = input("Enter model type for saving results (e.g., 9L, 9R, 20L or 20R): ").strip()

        # Map user choice to the appropriate feature set
        if feature_choice == "9":
            selected_features = df_9
        elif feature_choice == "20":
            selected_features = df_20
        elif feature_choice == "11":
            selected_features = df_11
        else:
            print("Invalid feature set choice for Credit. Defaulting to 'df_9'.")
            selected_features = df_9

        
        # Calculate mean threshold based on feature sets
        mean_threshold = calculate_mean_threshold((df_9, target_credit), (df_20, target_credit))

        # Train 
        train_dataset(
            dataset_name="credit",
            features=selected_features,
            target=target_credit,
            id_column=id_credit,
            threshold=mean_threshold,
            prefix=prefix,
            mode=mode,
            output_type=output_type,
            model_type=model_type
        )   
    
    if choice == "unemployment":
        id_unemployment,target_unemployment, df_30, df_22, df_9, df_8, df_21 = load_unemployment_data()
        feature_choice = input("Choose feature set for Unemployment (e.g., 30, 22, 9): ").strip()
        prefix = input("Enter model type for saving results (e.g., 30L, 30R, 22L, 22R, 9L or 9R): ").strip()

        # Map user choice to the appropriate feature set
        if feature_choice == "30":
            selected_features = df_30
        elif feature_choice == "22":
            selected_features = df_22
        elif feature_choice == "9":
            selected_features = df_9
        elif feature_choice == "8":
            selected_features = df_8
        elif feature_choice == "21":
            selected_features = df_21
        else:
            print("Invalid feature set choice for Unemployment. Defaulting to 'df_30'.")
            selected_features = df_30


        # Train
        train_dataset(
            dataset_name="unemployment",
            features=selected_features,
            target=target_unemployment,
            id_column=id_unemployment,
            threshold=0.5,
            prefix=prefix,
            mode=mode,
            output_type=output_type,
            model_type=model_type
        )

if __name__ == '__main__':
    main()
