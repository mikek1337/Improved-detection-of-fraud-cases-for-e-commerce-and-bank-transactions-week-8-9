import mlflow
import os
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
class TrainModel:
    def __init__(self, feat_data):
        mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
        # pdir = os.path.abspath(__file__).split('/')
        # pdir.pop()
        # preprocessed_data = load(os.path.dirname('/'.join(pdir))+'/data/processed/data.csv')
        # self.feat_data = feature_engineering_pipeline(preprocessed_data)
        # self.data = self.feat_data['processed_data']
        # print(self.data.columns)
        # x = self.data.drop('remainder__is_high_risk', axis=1)
        # y = self.data['remainder__is_high_risk']
        self.train_setup= feat_data

    def train_linear(self, params):
        lr = LogisticRegression(**params)
        lr.fit(self.train_setup['x_train'], self.train_setup['y_train'])
        return lr
    def train_tree(self, params):
        tree = DecisionTreeRegressor(**params)
        tree.fit(self.train_setup['x_train'], self.train_setup['y_train'])
        return tree
    def train_gbm(self, params):
        gbm = GradientBoostingClassifier(**params)
        gbm.fit(self.train_setup['x_train'], self.train_setup['y_train'])
        return gbm
    
    
    def metrics(self, model):
        y_predict = model.predict(self.train_setup['x_test'])
        accuracy = accuracy_score(self.train_setup['y_test'], y_predict)
        recall = recall_score(self.train_setup['y_test'], y_predict)
        f1 = f1_score(self.train_setup['y_test'], y_predict)
        percision = precision_score(self.train_setup['y_test'], y_predict)
        roc = roc_auc_score(self.train_setup['y_test'], y_predict)
        return {
            'accuracy':accuracy,
            'recall': recall,
            'f1': f1,
            'percision': percision,
            'roc':roc
        }

    
    def run(self, log_name:str):
        lr_params = {
            "solver": "lbfgs",
            "max_iter": 1000,
            "multi_class": "auto",
            "random_state": 8888,
        }
        gbm_params = {
            'n_estimators':100,             # default: 100
            'learning_rate':0.1,
            'loss':'log_loss',            # default: 0.1
            'subsample':1.0,                # default: 1.0
            'criterion':'friedman_mse',     # default: 'friedman_mse'
            'random_state':None,            # default: None
            'validation_fraction':0.1,      # default: 0.1
            'n_iter_no_change':None,        # default: None
            'tol':1e-4,                     # default: 1e-4
            'ccp_alpha':0.0
        }
        tree_params = {
            'criterion':'absolute_error',
            'max_depth':8,
            'min_samples_split':5,
            'min_samples_leaf':3,
            'max_features':0.7, # Use 70% of features
            'ccp_alpha':0.01,   # Apply some pruning
            'random_state':42
        }
        lr_model = self.train_linear(lr_params)
        lr_metrics = self.metrics(lr_model)
        self.logging(lr_model, lr_metrics,'Linear Regression', lr_params, log_name)
        tree_model = self.train_tree(tree_params)
        tree_metrics = self.metrics(tree_model)
        self.logging(tree_model, tree_metrics,'Decsion Tree', tree_params, log_name)
        gbm_model = self.train_gbm(gbm_params)
        gbm_metrics = self.metrics(gbm_model)
        self.logging(gbm_model, gbm_metrics,'GBM', gbm_params, log_name)

    def logging(self,model, metrics, experiment_name, params, model_name):
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            signature = infer_signature(self.train_setup['x_train'], model.predict(self.train_setup['x_test']))
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
                signature=signature,
                input_example=self.train_setup['x_train'],
                registered_model_name=f'tracking-{model_name}'
            )
            mlflow.set_logged_model_tags(
                model_info.model_id, {"Training Info":"Basic info"}
            )



if __name__ == '__main__':
    train = TrainModel()
    train.run()
