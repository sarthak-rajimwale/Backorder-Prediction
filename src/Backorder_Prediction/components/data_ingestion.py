import os
import sys
import pandas as pd
from src.Backorder_Prediction.exception import CustomException
from src.Backorder_Prediction.logger import logging
from src.Backorder_Prediction.utils import read_training_data, read_test_data  
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion, defining paths for saving the data."""
    train_data_path: str = os.path.join('artifacts', 'train.parquet')  
    test_data_path: str = os.path.join('artifacts', 'test.parquet')  

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df_train = read_training_data()
            df_test = read_test_data()

            logging.info("Successfully read both training and test data from MySQL")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df_train.to_parquet(self.ingestion_config.train_data_path, index=False)  

            df_test.to_parquet(self.ingestion_config.test_data_path, index=False)  
            logging.info("Data Ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise CustomException(e, sys)
