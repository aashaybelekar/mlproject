import os
import sys
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Training")
        try:
            logging.info("Spliting Training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Support Vector Machine": SVC(probability=True),
                "Random Forest": RandomForestClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "XGBoost": XGBClassifier(),
            }

            param_grid = {
                "Logistic Regression": {
                    'penalty': [None, 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                "Support Vector Machine": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }



            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params = param_grid)

            best_model_score = max(list(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found.")
            
            logging.info(f"Best Found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, predicted)

            return auc
        except Exception as e:
            raise CustomException(e, sys)
