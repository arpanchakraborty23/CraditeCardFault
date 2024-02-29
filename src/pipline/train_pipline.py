import os,sys
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_train import ModelTrain
from src.logger import logging
from src.exception import CustomException

obj=DataIngestion()
train_data,test_data=obj.initiate_data_ingestion()


transformation_obj=DataTransformation()
train_arr,test_arr,_=transformation_obj.initiate_transformaation(train_data,test_data)
   

model_train_obj=ModelTrain()
print(model_train_obj.initate_model_train(train_arr,test_arr))