from src.Backorder_Prediction.logger import logging
from src.Backorder_Prediction.exception import CustomException
from src.Backorder_Prediction.components.data_ingestion import DataIngestion
from src.Backorder_Prediction.components.data_transformation import transform_data
from src.Backorder_Prediction.components.model_trainer import ModelTrainer
import pandas as pd
import sys


def main():
    logging.info("Execution started for the backorder prediction pipeline.")
    try:
        logging.info("Starting data ingestion...")
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train file: {train_path}, Test file: {test_path}")

        train_data = pd.read_parquet(train_path)

        categorical_columns = train_data.select_dtypes(include=['object']).columns.tolist()
        logging.info(f"Identified categorical columns: {categorical_columns}")

        target_column = "went_on_backorder"  

        logging.info("Starting data transformation...")
        output_parquet_path = "artifacts/transformed_data.parquet"

        X_resampled, y_resampled, selected_features = transform_data(
            file_path=train_path,
            output_parquet_path=output_parquet_path,
            categorical_columns=categorical_columns,
            target_column=target_column
        )
        logging.info(f"Data transformation completed. Transformed data saved at: {output_parquet_path}")
        logging.info(f"Selected features: {selected_features}")

        logging.info("Starting model training...")
        model_path = "artifacts/best_model.pkl"

        model_trainer = ModelTrainer(train_file=output_parquet_path, model_path=model_path)
        metrics = model_trainer.run()

        roc_auc = metrics["roc_auc"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]
        conf_matrix = metrics["conf_matrix"]

        logging.info(f"Model training completed. Best model saved at: {model_path}")
        logging.info(f"Model performance (ROC-AUC): {roc_auc:.4f}")
        logging.info(f"Model performance (Precision): {precision:.4f}")
        logging.info(f"Model performance (Recall): {recall:.4f}")
        logging.info(f"Model performance (F1-Score): {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")

        logging.info("Pipeline execution completed successfully.")

    except Exception as e:
        logging.error("An error occurred during pipeline execution.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
