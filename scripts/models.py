from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
class Models:
    def __init__(self):
        pass
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