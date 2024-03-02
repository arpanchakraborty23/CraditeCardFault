import os,sys
import pickle
import pandas as pd 
import numpy as np 

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

def save_obj(file_path,obj):
    try:
        logging.info(f"Saving object in {file_path}")
        directory = os.path.dirname(file_path)
        os.makedirs(directory,exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(obj,f)
        
        logging.info(f"Saved object in {file_path}")
    
    except Exception as e:
            raise CustomException(e,sys) from e

def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)

        logging.info('pickl file loaded')
    except Exception as e:
            raise CustomException(e,sys) from e

def model_evaluate(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        logging.info('model evaluation started')
        
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f'accuracy score for {list(models.keys())[i]}: {accuracy}')
            report[list(models.keys())[i]] = accuracy

        return report  # Corrected indentation for the return statement

                
    except Exception as e:
        raise CustomException(e,sys) from e


def Hyperparameters_Tuning(x_train, y_train, x_test, y_test,classifiers):
    try:
        logging.info('Hyperparameter Tuning has started')
        report={}
        for clf_name, (classifier, param_grid) in classifiers.items():
            print(f"\nTuning hyperparameters for {clf_name}")

            # Use GridSearchCV to find the best hyperparameters
            grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='accuracy')
            grid_search.fit(x_train, y_train)

            # Get the best hyperparameters
            best_params = grid_search.best_params_
            print(f"Best Hyperparameters for {clf_name}: {best_params}")

            # Predict on the test set using the best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(x_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)*100
            print(f"Accuracy on Test Set for {clf_name}: {accuracy}")
            report[list(models.keys())[i]] = accuracy


        return  accuracy

    except Exception as e:
        logging.info('Error in Tuning')
        raise CustomException(sys,e)              

