from flask import Flask, request, render_template
import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipline

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('predict.html')
    else:
        data=CustomData(
            Age = int(request.form.get('age')),
            Sex = request.form.get('sex'),
            ChestPainType = request.form.get('chespaintype'),
            Cholesterol = request.form.get('cholesterol'),
            FastingBS = int(request.form.get('fastingbs')),
            MaxHR = int(request.form.get('maxhr')),
            ExerciseAngina = request.form.get('exerciseangina'),
            Oldpeak = float(request.form.get('oldpeak')),
            ST_Slope = request.form.get('st_segment_slope')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipline = PredictPipline()
        results = predict_pipline.predict(pred_df)
        if results >= 0.5:
            results = "The prediction is that Heart Disease is present"
        else:
            results = "The prediction is that Heart Disease is Absent"
        return render_template('predict.html', results=results)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)