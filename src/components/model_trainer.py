import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.utils import save_object
from src.utils import evaluate_model


@dataclass
class ModelTraningConfig:
    train_model_file_obj = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_traner_config = ModelTraningConfig()


    def initate_model_training(self,train_array,test_array):
        try:
            logging.info("Split Dependent And Independent Features")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                
                "Logastic":LogisticRegression()
            }
            params = {
                "Random Forest":{
                    "class_weight":["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                },
                
            
                "Logastic":{
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                }
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models,param=params)

                ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")

            save_object(file_path=self.model_traner_config.train_model_file_obj,
                obj = best_model
                )

        except Exception as e:
            logging.info("Error Occured in Model Traning")
            raise CustomException(e,sys)