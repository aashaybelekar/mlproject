import sys
import os
import pandas as pd 
from src.exception import CustomException
from src.utils import load_object

class PredictPipline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            
            model_path = os.path.join(os.getcwd(), 'artifacts/model.pkl')
            preprocessor_path = os.path.join(os.getcwd(), 'artifacts/preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Age: int,
                 Sex: str,
                 ChestPainType: str,
                 Cholesterol: str,
                 FastingBS: int,
                 MaxHR: int,
                 ExerciseAngina: str,
                 Oldpeak: int,
                 ST_Slope: str):
        self.age = Age
        self.sex = Sex
        self.chest_pain_type = ChestPainType
        self.cholesterol = Cholesterol
        self.fasting_bs = FastingBS
        self.max_hr = MaxHR
        self.exercise_angina = ExerciseAngina
        self.oldpeak = Oldpeak
        self.st_slope = ST_Slope

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Age': [self.age],
                'Sex': [self.sex],
                'ChestPainType': [self.chest_pain_type],
                'Cholesterol': [self.cholesterol],
                'FastingBS': [self.fasting_bs],
                'MaxHR': [self.max_hr],
                'ExerciseAngina': [self.exercise_angina],
                'Oldpeak': [self.oldpeak],
                'ST_Slope': [self.st_slope]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
