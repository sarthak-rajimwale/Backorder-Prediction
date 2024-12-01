import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump
import logging
import numpy as np

class ModelTrainer:
    def __init__(self, train_file: str, model_path: str):
        self.train_file = train_file
        self.model_path = model_path

    def load_data(self, sample_fraction=1.0):
        logging.info("Loading data...")
        data = pd.read_parquet(self.train_file)
        if sample_fraction < 1.0:
            data = data.sample(frac=sample_fraction, random_state=42)
            logging.info(f"Sampled {len(data)} rows from the dataset.")
        return data

    def prepare_data(self, data, target_column="went_on_backorder"):
        logging.info("Preparing data for training...")
        X = data.drop(columns=[target_column])
        y = data[target_column].astype(int)  
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_and_tune_model(self, X_train, y_train):
        logging.info("Initializing the model...")
        model = HistGradientBoostingClassifier(random_state=42)

        param_grid = {
            'max_iter': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [None, 10, 20],
            'l2_regularization': [0.0, 1.0],
        }

        logging.info("Starting hyperparameter tuning...")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,  
            cv=3,
            scoring='roc_auc',
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        logging.info(f"Best Model Parameters: {random_search.best_params_}")
        return best_model

    def evaluate_model(self, model, X_test, y_test):
        logging.info("Evaluating the model...")
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, probabilities)

        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        conf_matrix = confusion_matrix(y_test, predictions)

        logging.info(f"Model ROC-AUC: {roc_auc:.4f}")
        logging.info(f"Model Precision: {precision:.4f}")
        logging.info(f"Model Recall: {recall:.4f}")
        logging.info(f"Model F1-Score: {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")

        return roc_auc, precision, recall, f1, conf_matrix

    def save_model(self, model):
        logging.info("Saving the model...")
        dump(model, self.model_path)
        logging.info(f"Model saved at {self.model_path}")

    def run(self):
        logging.info("Pipeline execution started.")
        data = self.load_data(sample_fraction=1.0)  
        X_train, X_test, y_train, y_test = self.prepare_data(data)

        best_model = self.train_and_tune_model(X_train, y_train)
        roc_auc, precision, recall, f1, conf_matrix = self.evaluate_model(best_model, X_test, y_test)

        self.save_model(best_model)
        logging.info(f"Pipeline execution completed. Final ROC-AUC: {roc_auc:.4f}")
        
        return {
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "conf_matrix": conf_matrix
        }
