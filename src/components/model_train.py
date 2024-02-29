import os,sys
import pickle
import pandas as pd
import numpy as np 
from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.logger import logging
from src.exception import CustomException
from src.utils.utils import save_obj,model_evaluate,Hyperparameters_Tuning

@dataclass
class ModelTrainConfig:
    model_path=os.path.join("model","model.pkl")

class ModelTrain:
    def __init__(self):
        self.model_train_config=ModelTrainConfig()

    def initate_model_train(self,train_array,test_array):
        try:
            logging.info('Model train started')

            # 1. Load the data

            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('model cration')
            # 5. Evaluate the model
            
            models = { 
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "LogisticRegression": LogisticRegression(),
                "VotingClassifier": VotingClassifier(
                    estimators=[
                        ("dt", DecisionTreeClassifier()),
                        ("rf", RandomForestClassifier()),
                        ("lr", LogisticRegression()),
                    ],
                    voting="soft",
                ),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "BaggingClassifier": BaggingClassifier(),
                "SVC": SVC(),
            }
            
            model_report:dict=model_evaluate(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)   
            print( model_report)

            
            logging.info(f'Model Report : {model_report}')

            print('\n====================================================================================\n')

            # 6. Find the best model
            best_model_score=max(sorted(model_report.values()))

            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Score : {best_model_score}')

    #         # 6. Save the model
            logging.info('Save the model')
            save_obj(
               
               file_path=self.model_train_config.model_path,
               obj=best_model
              )

            return (self.model_train_config.model_path,best_model,best_model_name)
  
        except Exception as e:
            logging.info('Error occure model train')
            raise CustomException(sys,e)  
    # def initate_Hyperparameters_Tuning(self,train_array,test_array):
    #     try:

    #         x_train, y_train, x_test, y_test = (
    #             train_array[:,:-1],
    #             train_array[:,-1],
    #             test_array[:,:-1],
    #             test_array[:,-1]
    #         )
    #         classifiers = {
    #                 'RandomForest': (RandomForestClassifier(), {
    #                     'n_estimators': [50, 100, 200],
    #                     'max_depth': [None, 10, 20, 30],
    #                     'min_samples_split': [2, 5, 10],
    #                     'min_samples_leaf': [1, 2, 4]
    #                 }),
    #                 'SVM': (SVC(), {
    #                     'C': [0.1, 1, 10],
    #                     'kernel': ['linear', 'rbf', 'poly'],
    #                     'gamma': ['scale', 'auto']
    #                 }),
    #                 'GradientBoosting': (GradientBoostingClassifier(), {
    #                     'learning_rate': [0.01, 0.1, 0.2],
    #                     'n_estimators': [50, 100, 200],
    #                     'max_depth': [3, 4, 5],
    #                     'subsample': [0.8, 1.0]
    #                 }),
    #                 'KNN':(KNeighborsClassifier(),{
    #                     'n_neighbors': [3, 5, 7],
    #                     'weights': ['uniform', 'distance'],
    #                     'p': [1, 2],
    #                     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #                 }),
    #                 'Logistic Regrrassion':(LogisticRegression(),{
    #                     'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    #                     'C': [0.1, 1, 10],
    #                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #                     'max_iter': [100, 200, 300],


    #                 })

    #             }


              
    #         model_report:dict= Hyperparameters_Tuning(x_train, y_train, x_test, y_test,classifiers)
    #         print( model_report)

            
    #         logging.info(f'Model Report : {model_report}')

    #         print('\n====================================================================================\n')
    #         return model_report

    #     except Exception as e:
    #         logging.info('eroor in Tuning 148')
    #         raise CustomException(sys,e)    

