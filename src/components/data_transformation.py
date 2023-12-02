import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransormationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransormation:
    def __init__(self):
        self.data_transformation_config = DataTransormationConfig()
        self.numerical_features = [
                'Age',  
                'Cholesterol', 
                'MaxHR', 
                'Oldpeak',
            ]
        self.categorical_features = [
                'Sex', 
                'ChestPainType', 
                'FastingBS',
                'ExerciseAngina', 
                'ST_Slope',
            ]
        self.target_column_name = "HeartDisease"

        self.drop = [
            'RestingECG',
            'RestingBP'
        ]
    
    def get_data_transformer_object(self):
        '''
        This function will perform the following actions:
        Numerical Features:
            1.median imputation due to inclusion of outliers
            2.standard normal scaling 
        
        Categorical Features:
            1.Mode imputation
            2.standard normal scaling
        '''
        try:
            num_pipline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                ]
            )

            logging.info("Numerical columns scaling completed")

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipline", num_pipline, self.numerical_features),
                    ("cat_pipline", cat_pipeline, self.categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_date_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = train_df.drop(columns=self.drop, axis=1)
            test_df = test_df.drop(columns=self.drop, axis=1)
            
            logging.info("read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=self.target_column_name, axis=1)
            target_feature_train_df = train_df[self.target_column_name]

            input_feature_test_df = test_df.drop(columns=self.target_column_name, axis=1)
            target_feature_test_df = test_df[self.target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )
            
            logging.info("saved preprocessing object")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )


            
        except Exception as e:
            raise CustomException(e, sys)
