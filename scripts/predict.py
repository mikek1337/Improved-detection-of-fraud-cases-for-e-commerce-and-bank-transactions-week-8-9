import pandas as pd
import mlflow.sklearn
import os
import joblib
current_dir = os.path.dirname(os.path.abspath(__file__))
model_artifacts_dir = os.path.join(current_dir, '../full_feat_pipeline/full_feature_pipeline.joblib')
PIPELINE_PATH = model_artifacts_dir 
class PredictModel:
    def __init__(self, model_name:str, version:str):
        mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
        self.model_uri = f'models:/{model_name}/{version}'
        self.model = mlflow.sklearn.load_model(self.model_uri)
        
    def predict(self,data:pd.DataFrame):
        result = self.model.predict()
        return result
    

