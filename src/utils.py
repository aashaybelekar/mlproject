import os
import sys
import dill
from src.logger import logging

import numpy as np 
import pandas as pd 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    logging.info("Entered evaluation stage.")
    try:
        logging.info("Model Evaluation started")
        report = {}
        for i in range(len(models.values())):
            logging.info(f"loop {i+1}")
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_test, y_test)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict_proba(X_train)[:,1]
            y_test_pred = model.predict_proba(X_test)[:,1]
            test_model_score = roc_auc_score(y_test, y_test_pred)
            logging.info(f"{test_model_score}")

            report[list(models.keys())[i]] = test_model_score

            logging.info("completed the loop")
        return report
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
