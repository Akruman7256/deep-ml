import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing

    def forward(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.feature_probs = {}
        for c in self.classes:
            self.priors[c] = np.sum(y == c) / len(y)
            X_c = X[y == c]
            count_c = X_c.shape[0]
            self.feature_probs[c] = (np.sum(X_c, axis=0) + self.smoothing) / (count_c + 2*self.smoothing)

    def predict(self, X):
        predictions = []
        for sample in X:
            scores = {}
            for c in self.classes:
                log_prob = np.log(self.priors[c])
                for i, feature_val in enumerate(sample):
                    p = self.feature_probs[c][i]
                    if feature_val == 1:
                        log_prob += np.log(p)
                    else:
                        log_prob += np.log(1 - p)
                scores[c] = log_prob
            predicted_class = max(scores, key=scores.get)
            predictions.append(predicted_class)
        return np.array(predictions)