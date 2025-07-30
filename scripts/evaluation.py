import shap
import matplotlib.pyplot as plt

def lr_shap(model):
    return shap.LinearExplainer(model)
def gbm_shap(model):
    return shap.TreeExplainer(model)
def decision_shap(model):
    return shap.TreeExplainer(model)

def summary_plot(model, sharp_fn, x_train):
    shapvalue = sharp_fn(model).shap_value(x_train)
    shap.violin_plot(shapvalue)


