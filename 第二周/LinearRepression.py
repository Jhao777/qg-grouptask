import numpy as np
from sklearn.metrics import r2_score


class LinearRepression:
    def __init__(self):
        # 初始化 LinearRepression 模型
        self.interception_ = None  # 截距
        self.coef_ = None  # 系数
        self._theta = None

    def fit_normal(self, X_train, y_train):
        # 根据训练集 x_train, y_train 训练 linearRepression 模型
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        assert X_predict.shape[1] == len(self.coef_), \
            'the length of coef_ must be equal to the column of X_predict'
        assert self.coef_ is not None and self.interception_ is not None, \
            "coef_ and interception cant' be None"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_predict, y_test)

    def __repr__(self):
        return "LinearRepression()"


