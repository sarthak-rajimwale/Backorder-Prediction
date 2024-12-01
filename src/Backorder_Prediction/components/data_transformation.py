import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from src.Backorder_Prediction.logger import logging
from src.Backorder_Prediction.exception import CustomException
import sys
import os
import numpy as np


def fill_missing_values(df):
    logging.info("Filling missing values in the dataset")
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if df[col].isnull().sum() > 0: 
                df[col].fillna(df[col].mode()[0], inplace=True)  
                logging.info(f"Filled missing values in column: {col}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:  
                df[col].fillna(df[col].mean(), inplace=True)  
                logging.info(f"Filled missing values in column: {col}")
        
        return df

    except Exception as e:
        logging.error(f"Error while filling missing values: {str(e)}")
        raise CustomException(e, sys)


def ensembling(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    try:
        logging.info("Ensembling categorical columns")
        for col in categorical_columns:
            df[col].replace(['No', 'Yes'], [0, 1], inplace=True)
        return df
    except Exception as ex:
        logging.error(f"Error while ensembling categorical columns: {ex}")
        raise CustomException(ex, sys)


def feature_selection(df: pd.DataFrame, target_column: str, corr_threshold: float = 0.004) -> list:
    try:
        logging.info("Performing feature selection based on correlation with the target column")
        corr_matrix = df.corr()
        corr_with_y = corr_matrix[target_column].drop(target_column)  
        relevant_features = corr_with_y[corr_with_y > corr_threshold]
        relevant_features_1 = corr_with_y[corr_with_y < -corr_threshold]
        all_relevant_features = pd.concat([relevant_features, relevant_features_1])
        
        logging.info(f"Relevant features: {all_relevant_features.index.tolist()}")
        return all_relevant_features.index.tolist()
    except Exception as ex:
        logging.error(f"Error while performing feature selection: {ex}")
        raise CustomException(ex, sys)


def balance_data(X: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.Series):
    try:
        logging.info("Balancing the dataset using RandomOverSampler")
        oversampler = RandomOverSampler()
        X_resampled, y_resampled = oversampler.fit_resample(X, y)
        
        logging.info(f"Original class distribution: {Counter(y)}")
        logging.info(f"Resampled class distribution: {Counter(y_resampled)}")
        
        return X_resampled, y_resampled
    except Exception as ex:
        logging.error(f"Error while balancing the data: {ex}")
        raise CustomException(ex, sys)


def transform_data(file_path: str, output_parquet_path: str, categorical_columns: list, target_column: str) -> (pd.DataFrame, pd.Series, list):
    try:
        logging.info(f"Loading training data from file: {file_path}")
        
        if not os.path.exists(file_path):
            logging.error(f"File not found at path: {file_path}")
            raise FileNotFoundError(f"File not found at path: {file_path}")
        
        df_train = pd.read_parquet(file_path)  
        
        if 'sku' in df_train.columns:
            df_train = df_train.drop(columns=['sku'])
            logging.info("Dropped 'sku' column successfully.")
        
        df_train = fill_missing_values(df_train)
        
        df_train = ensembling(df_train, categorical_columns)
        
        selected_features = feature_selection(df_train, target_column)
        
        X = df_train[selected_features]
        y = df_train[target_column]
        
        X_resampled, y_resampled = balance_data(X, y)
        
        logging.info(f"Saving transformed data to Parquet at: {output_parquet_path}")
        transformed_data = pd.concat(
            [pd.DataFrame(X_resampled, columns=selected_features), 
             pd.DataFrame(y_resampled, columns=[target_column])], axis=1)
        transformed_data.to_parquet(output_parquet_path, index=False)
        
        logging.info("Data transformation completed successfully.")
        return X_resampled, y_resampled, selected_features

    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
        raise CustomException(fnf_error, sys)
    except Exception as ex:
        logging.error(f"Error in transforming data: {ex}")
        raise CustomException(ex, sys)
