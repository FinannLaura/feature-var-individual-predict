import sys
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import *


# --------------------------------------
# Preprocessing Functions for each Dataset
# --------------------------------------

def preprocess_compas_data(filepath="../data/Compas-Scores-Two-Years.csv"):
    df = pd.read_csv(filepath) 

    # Data cleaning
    df = df[(df["days_b_screening_arrest"] <= 30) 
            & (df["days_b_screening_arrest"] >= -30)
            & (df["is_recid"] != -1)
            & (df["c_charge_degree"] != 'O')
            & (df["score_text"] != 'N/A')].reset_index(drop=True)

    # Define compas_id and compas_target after filtering
    compas_id = df['id']
    compas_target = df['is_recid']

    # Define feature sets
    two_features = df[["age", "priors_count"]]

    seven_features = df[["sex", "age", "c_charge_desc", "c_charge_degree", 
                         "juv_misd_count", "juv_fel_count", "priors_count"]]
    

    # Columns to drop
    columns_to_drop = [
        'start', 'end', 'event', 'two_year_recid', 'violent_recid', 'in_custody', 
        'out_custody', 'vr_charge_degree', 'vr_charge_desc', 'r_charge_degree', 
        'r_charge_desc', 'decile_score.1', 'priors_count.1', 'screening_date', 
        'v_screening_date', 'age_cat', 'decile_score', 'v_decile_score', 'score_text', 
        'v_score_text', 'is_violent_recid', 'compas_screening_date', 
        'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number', 
        'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'r_days_from_arrest', 
        'r_offense_date', 'r_jail_in', 'r_jail_out', 'vr_offense_date', 'id', 
        'name', 'first', 'last', 'dob', 'vr_case_number', 'type_of_assessment', 
        'v_type_of_assessment', 'r_case_number', 'is_recid'
    ]

    # Define additional feature sets after dropping columns
    nine_features = df.drop(columns=columns_to_drop, axis=1)
    nine_minus_two = nine_features.drop(['age', 'priors_count'], axis=1)
    five_features = df[["sex", "c_charge_desc", "c_charge_degree", "juv_misd_count",
                         "juv_fel_count"]]
    
    # One-hot encoding
    seven_features = pd.get_dummies(seven_features, columns=["sex", "c_charge_desc", "c_charge_degree"])
    nine_features = pd.get_dummies(nine_features, columns=["race", "sex", "c_charge_desc", "c_charge_degree"])
    nine_minus_two = pd.get_dummies(nine_minus_two, columns=["race", "sex", "c_charge_desc", "c_charge_degree"])
    five_features = pd.get_dummies(five_features, columns=["sex", "c_charge_desc", "c_charge_degree"])



    return compas_id, compas_target, two_features, seven_features, nine_features, nine_minus_two, five_features


def preprocess_credit_data(filepath="../data/South_German_Credit_Data.csv"):

    df = pd.read_csv(filepath)
    df['ID'] = range(len(df))
    id_credit = df['ID']
    target_credit = df['kredit']
    df = df.drop(['kredit', 'ID'], axis=1)
    
    df["hoehe"] = np.log2(df["hoehe"])
    df["alter"] = np.log2(df["alter"])
    df["laufzeit"] = np.log2(df["laufzeit"])


    df_9 = df.drop(['verw', 'buerge', 'wohnzeit', 'weitkred', 'wohn', 'telef', 'gastarb', 'pers', 
                    'sparkont', 'verm', 'pers', 'laufzeit'], axis=1)
    df_9 = pd.get_dummies(df_9, columns=["laufkont","moral", "beszeit", "famges", "beruf"])

 
    df_20 = pd.get_dummies(df, columns=["laufkont", "moral", "verw", "sparkont", 
                                        "beszeit", "famges", "buerge", "verm", 
                                        "weitkred", "wohn", "beruf", "telef", 
                                        "gastarb", "pers"])

    df_20_without_df_9 = [col for col in df_20.columns if col not in df_9.columns]
    df_11 = df_20[df_20_without_df_9]

    return id_credit, target_credit, df_9, df_20, df_11


def preprocess_unemployment_data(filepath="../data/1203_ALMP_Data_E_v1.0.0.csv"):
    df = pd.read_csv(filepath)
    
    df = df[(df['treatment3'] == 'no program') & (df['treatment6'] == 'no program')]

    df['ID'] = range(len(df))
    id_unemployment = df['ID']
    df = df.drop(['ID'], axis=1)

    
    # Define target variable as 'employed' based on employment status from month 1 to 36
    employed_columns = [f'employed{i}' for i in range(1, 37)]
    df['employed'] = df[employed_columns].apply(lambda x: x.any(), axis=1).astype(int)
    target_unemployment = df['employed']
    df = df.drop(employed_columns, axis=1)
    
    # Drop treatment group features
    treatment_group_features = [
        'computer3', 'computer6', 'vocational6', 'emp_program3', 'emp_program6', 
        'job_search3', 'job_search6', 'language3', 'language6', 'personality3', 
        'personality6', 'start_q2', 'treatment3', 'treatment6', 'vocational3', 'vocational6'
    ]
    if all(elem in df.columns for elem in treatment_group_features):
        df = df.drop(treatment_group_features, axis=1)


    # Scale selected features
    scaler = StandardScaler()
    for column in ['age', 'emp_share_last_2yrs', 'emp_spells_5yrs', 'gdp_pc', 'past_income', 'unemp_rate']:
        if column in df.columns:
            df[column] = scaler.fit_transform(df[[column]]) * 0.1

    
    # Define different feature sets
    columns_to_drop_30 = [
        'cw_id', 'employed', 'city', 'prev_job_sector', 'prev_job', 'cw_age',
        'cw_cooperative', 'cw_educ_above_voc', 'cw_educ_tertiary', 'cw_female',
        'cw_missing', 'cw_own_ue', 'cw_tenure', 'cw_voc_degree', 'cw_age_missing',
        'ue_cw_allocation1', 'ue_cw_allocation2', 'ue_cw_allocation3',
        'ue_cw_allocation4', 'ue_cw_allocation5', 'ue_cw_allocation6',
        'ue_spells_last_2yrs', 'prev_job_sec_cat', 'qual', 'prev_job_unskilled', 'employability',
    ]
    df_30 = df.drop([col for col in columns_to_drop_30 if col in df.columns], axis=1)


    columns_to_drop_22 = [
        'cw_id', 'employed', 'city', 'prev_job_sector', 'prev_job', 'cw_age',
        'cw_cooperative', 'cw_educ_above_voc', 'cw_educ_tertiary', 'cw_female',
        'cw_missing', 'cw_own_ue', 'cw_tenure', 'cw_voc_degree', 'cw_age_missing',
        'ue_cw_allocation1', 'ue_cw_allocation2', 'ue_cw_allocation3',
        'ue_cw_allocation4', 'ue_cw_allocation5', 'ue_cw_allocation6',
        'ue_spells_last_2yrs', 'unemp_rate', 'canton_french', 'canton_german',
        'canton_italian',  'canton_moth_tongue',  'gdp_pc',  'married',
        'past_income',  'unemp_rate', 'employability' , 'prev_job_sec_cat',
         'qual', 'prev_job_skilled'
    ]
    df_22 = df.drop([col for col in columns_to_drop_22 if col in df.columns], axis=1)


    columns_to_drop_9 = [
        'cw_id', 'employed', 'city', 'prev_job_sector', 'prev_job', 'cw_age',
        'cw_cooperative', 'cw_educ_above_voc', 'cw_educ_tertiary', 'cw_female',
        'cw_missing', 'cw_own_ue', 'cw_tenure', 'cw_voc_degree', 'cw_age_missing',
        'ue_cw_allocation1', 'ue_cw_allocation2', 'ue_cw_allocation3',
        'ue_cw_allocation4', 'ue_cw_allocation5', 'ue_cw_allocation6',
        'ue_spells_last_2yrs','canton_french', 'canton_german',
        'canton_italian','gdp_pc','unemp_rate', 'employability' , 'prev_job_sec_cat',
        'qual_degree', 'qual_semiskilled','qual_unskilled', 'qual_wo_degree',
        'other_mother_tongue', 'emp_spells_5yrs', 'foreigner_b', 'foreigner_c',
        'qual','swiss','prev_job_sec1', 'prev_job_sec2', 'prev_job_sec3', 'prev_job_sec_mis',
        'prev_job_manager', 'prev_job_self','married', 'prev_job_skilled',
    ]
    df_9 = df.drop([col for col in columns_to_drop_9 if col in df.columns], axis=1)

    # df_30 minus df_22
    df_8 = df_30.drop([col for col in columns_to_drop_22 if col in df_30.columns], axis=1)

    # df_30 minus df_9
    df_21 = df_30.drop([col for col in columns_to_drop_9 if col in df_30.columns], axis=1)

    return  id_unemployment, target_unemployment, df_30, df_22, df_9, df_8, df_21


# --------------------------------------
# Loading and Cleaning Predictions
# --------------------------------------

def load_and_clean_predictions(dataset, type, feature_numbers, model_types, data_path="."):
    """
    Loads and cleans a single prediction dataset based on feature numbers and model types, ignoring timestamps.

    Args:
        dataset (str): Name of the dataset (e.g., 'compas', 'credit').
        feature_numbers (list): List of feature counts, e.g., ['9', '20', '30'].
        model_types (list): List of model types, e.g., ['LP', 'RP', 'LB', 'RB'].
        data_path (str): Path to the directory where prediction files are stored. Default is current directory.

    Returns:
        dict: Nested dictionary with cleaned dataframes, accessed by [feature_number][model_type].
    """
    df = {model_number: {} for model_number in feature_numbers}
    
    for model_number in feature_numbers:
        for model_type in model_types:
            # file pattern with the specified data_path
            pattern = os.path.join(data_path, f"{dataset}_{type}_{model_number}{model_type}_*.csv")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                # uses the first matching file, if found
                file_name = matching_files[0]
                df_file = pd.read_csv(file_name)
                
                # data cleaning
                df_file = df_file.replace('-', np.nan).apply(pd.to_numeric, errors='coerce')
                df[model_number][model_type] = df_file
                
                print(f"Loaded file: {file_name} for model {model_number}{model_type}")
            else:
                # empty dataframe is assigned if no matching file is found
                print(f"No matching file found for model {model_number}{model_type}")
                df[model_number][model_type] = pd.DataFrame()
    
    return df


# --------------------------------------
# Data Combining Functions
# --------------------------------------


def combine_and_save_dataframes(df, feature_numbers, model_type):
    combined_dfs = []
    
    for feature_number in feature_numbers:
        # Check if key exists to avoid KeyErrors
        if model_type in df.get(feature_number, {}):
            current_df = df[feature_number][model_type]
            
            # Remove the ID column if it's not the first DataFrame
            if feature_number != feature_numbers[0]:
                current_df = remove_id_column(current_df)
            
            combined_dfs.append(current_df)
        else:
            print(f"Warning: Model type '{model_type}' not found for feature number '{feature_number}'.")
    
    # Combine and save if data is available
    if combined_dfs:
        combined_df = pd.concat(combined_dfs, axis=1)
        combined_filename = f"combined_{model_type}.csv"
        combined_df.to_csv(combined_filename, index=False)
        print(f"Saved combined DataFrame for model {model_type} to {combined_filename}")
        return combined_df  
    else:
        print(f"No data available to save for model type '{model_type}'.")
        return pd.DataFrame()  
