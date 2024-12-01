import os
import sys
from src.Backorder_Prediction.exception import CustomException
from src.Backorder_Prediction.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

if not all([host, user, password, db]):
    logging.error("Environment variables not set properly. Check the .env file.")
    sys.exit(1)

def read_training_data():
    logging.info("Reading SQL database started")
    try:
        with pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db
        ) as mydb:
            logging.info("Connection Established")
            df_train = pd.read_sql_query('SELECT * FROM training_data', mydb)
            print(df_train.head())
            logging.info("Read Training Data Successfully")
            return df_train
    
    except Exception as ex:
        error_message = f"Error while reading training data: {ex}"
        error_details = str(sys.exc_info())  
        logging.error(f"{error_message} - Details: {error_details}")
        raise CustomException(error_message, error_details)  

def read_test_data():
    logging.info("Reading SQL database started")
    try:
        with pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db
        ) as mydb:
            logging.info("Connection Established")
            df_test = pd.read_sql_query('SELECT * FROM test_data', mydb)
            print(df_test.head())
            logging.info("Read Test Data Successfully")
            return df_test
        
    except Exception as ex:
        logging.error(f"Error while reading test data: {ex}")
        raise CustomException(ex)
