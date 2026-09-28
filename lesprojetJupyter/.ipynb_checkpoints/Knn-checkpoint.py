import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.theta = None
        self.lr = lr
        self.epochs = epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Ajout du biais
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]

        # Initialisation
        self.theta = np.zeros((X_b.shape[1], 1))

        for _ in range(self.epochs):
            # Forward
            z = X_b @ self.theta
            p = self._sigmoid(z)

            # Gradient
            grad = (1/m) * X_b.T @ (p - y)

            # Update
            self.theta -= self.lr * grad

    def predict_proba(self, X):
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return self._sigmoid(X_b @ self.theta)

  