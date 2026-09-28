







import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))





class RegressionLogistique:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.zeros(X_b.shape[1])
        for _ in range(self.n_iterations):
            z = X_b.dot(self.theta)
            y_pred = sigmoid(z)
            gradient = X_b.T.dot(y_pred - y) / y.size
            self.theta -= self.learning_rate * gradient











    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return (sigmoid(X_b.dot(self.theta)) >= 0.5).astype(int)



# Exemple
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 1])  # OR logique
model = RegressionLogistique()
model.fit(X, y)
print(model.predict(X))