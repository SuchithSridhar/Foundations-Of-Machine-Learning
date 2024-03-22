import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.classes = None
        self.weights = None
        self.bias = None

    def softmax(self, Z):
        # Subtract from max so maintain numerical stability
        # This is because e^x is unstable when x is really large.
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def fit(self, X, y, verbose=False):
        # Classes
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        # Initialize weights and bias
        self.weights = np.random.rand(n_classes, n_features)
        self.bias = np.zeros((1, n_classes))

        # Convert y to one-hot encoding
        y_one_hot = np.zeros((n_samples, n_classes))
        for i, class_ in enumerate(self.classes):
            y_one_hot[:, i] = (y == class_)

        # Gradient descent
        for i in range(self.num_iterations):
            if (verbose):
                print(f"Epoch: {i}")
            # Model prediction
            model = np.dot(X, self.weights.T) + self.bias
            probabilities = self.softmax(model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot((probabilities - y_one_hot).T, X)
            db = (1 / n_samples) * np.sum(probabilities - y_one_hot, axis=0, keepdims=True)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_probabilities(self, X):
        model = np.dot(X, self.weights.T) + self.bias
        return self.softmax(model)

    def predict(self, X):
        return np.argmax(self.predict_probabilities(X), axis=1)
