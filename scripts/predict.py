import pandas as pd
import mlflow.sklearn
import os

class PredictModel:
    def __init__(self, model_name:str, version:str):
        mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
        self.model_uri = f'models:/{model_name}/{version}'
        self.model = mlflow.sklearn.load_model(self.model_uri)
        
    def predict(self,data:pd.DataFrame):
        result = self.model.predict()
        return result
    

