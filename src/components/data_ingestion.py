import pandas as pd
import numpy as np 
import os,sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransormationConfig,DataTransormation
from src.components.model_train import ModelTrainConfig,ModelTrain


@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self): 
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion Started")
            df=pd.read_csv('experiment/Craditcard.csv')

            logging.info(' Data read completed ')
           

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True
            )
            logging.info('data save stared')
            df.to_csv(self.data_ingestion_config.raw_data_path)
            logging.info('Data saved ')

            train_data,test_data=train_test_split(df,test_size=0.25,random_state=0)
            logging.info(f'Data split completed {df.head().to_string()}')
            logging.info(f'Data split completed {df.columns}')

            train_data.to_csv(self.data_ingestion_config.train_data_path,header=True,index=False)

            test_data.to_csv(self.data_ingestion_config.test_data_path,header=True,index=False)
            logging.info('Data ingestion completed')

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            ) 
        except Exception as e:
            logging.info('Error occured in data ingestion')
            raise CustomException(sys,e)


