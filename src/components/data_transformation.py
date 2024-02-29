import pandas as pd 
import numpy as np
import os,sys
import pickle
from dataclasses import dataclass

from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import  CustomException
from src.utils.utils import save_obj

@dataclass
class DataTransormationConfig:
    preprocess_path=os.path.join("preprocess","preprocess.pkl")

class DataTransormation:
    def __init__(self):
        self.data_transformation_config=DataTransormationConfig() 

    def data_transformation_obj(self):
        try:
            logging.info('feature Enggearning obj cration  has started ')

            Preprocess=Pipeline(
                steps=[
                    ('impute',KNNImputer(n_neighbors=7)),
                    # ('outlier',RobustScaler()),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info('feature Enggearning obj completed')

            return Preprocess



        except Exception as e:
            logging.info('Error occured in data transfomation')
            raise CustomException(e,sys)       



    def initiate_transformaation(self,train_path,test_path):
        try:
            logging.info('Data trnsformation has started')

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Data read completed')

            prprocess=self.data_transformation_obj()

            Target_column='default payment next month'
            drop_column='Unnamed: 0'

            # Spliting data into independent and depnednet features

            # x_train y_train
            input_feature_train_df=train_df.drop(columns=[Target_column,drop_column],axis=1)
            target_feature_train_df=train_df[Target_column]

            logging.info(f" input_feature_train_df: {input_feature_train_df.head(2).to_string()}")
            logging.info('kdkkkfkk')
            logging.info(f" target_feature_train_df: {target_feature_train_df[2]}")

            # x_test y_test
            input_feature_test_df=test_df.drop(columns=[Target_column,drop_column],axis=1)
            target_feature_test_df=test_df[Target_column]

            logging.info(f" input_feature_test_df: {input_feature_test_df.head(2).to_string()}")

            ## appling preprocess obj

            input_feature_train_arr=prprocess.fit_transform(input_feature_train_df)
            input_feature_test_arr=prprocess.transform(input_feature_test_df)


            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('train_arr test_arr')
            save_obj(
                file_path=self.data_transformation_config.preprocess_path,
                obj=prprocess
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_path
            )


        except Exception as e:
            logging.info('Error occured in data transfomation')
            raise CustomException(e,sys) from e   
