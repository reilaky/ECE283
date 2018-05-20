import numpy as np
import math
class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

class Adaboost():
    def __init__(self, iters = 10000, thres = 0.01):
        # weak learn for every iteration
        self.clfs = []
        self.min_error = float('inf')
        self.iters = iters
        self.thres = thres
        self.M = 0
        self.w = []

    def fit(self, X, y, thres = 0.01):
        n_samples, n_features = np.shape(X)
        # initial weights
        w = np.full(n_samples, (1.0 / n_samples))
        for i in range(self.iters):
            clf = DecisionStump()
            self.find_decision_stump(X, y, w, clf)
            if not self.inclfs(clf):
                # calculate alpha and pred to update weights
                # 1e-10 to prevent min_error = 0
                pred = np.ones(n_samples)
                pred[clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold] = -1
                # update weights
                w *= np.exp(-clf.alpha * y * pred)
                # normalize
                w /= np.sum(w)
                self.clfs.append(clf)
                self.M += 1
            else:
                i -= 1
            
            if self.min_error < self.thres:
                self.M = len(self.clfs)
                break

        return self

    def predict(self, X, clfs = None):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        # For each classifier => label the samples
        if clfs == None:
            for clf in self.clfs:
                # Set all predictions to '1' initially
                predictions = np.ones(np.shape(y_pred))
                # The indexes where the sample values are below threshold
                predictions[clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold] = -1
                # Add predictions weighted by the classifiers alpha
                # (alpha indicative of classifier's proficiency)
                y_pred += clf.alpha * predictions
                #print(y_pred.T)
            # Return sign of prediction sum
            y_pred = np.where(y_pred < 0, 0, 1).flatten()
        else:
            for clf in clfs:
                predictions = np.ones(np.shape(y_pred))
                predictions[clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold] = -1
                y_pred += clf.alpha * predictions
            y_pred = np.where(y_pred < 0, 0, 1).flatten()

        return y_pred           

    def inclfs(self, clf):
        for c in self.clfs:
            if c.feature_index == clf.feature_index and c.threshold == clf.threshold:
                return True
        return False

    def find_decision_stump(self, X, y, w, clf, num_step = 50):
        n_samples, n_features = np.shape(X)
        min_error = float('inf')
        for feature_i in range(n_features):
            min_val = X[:, feature_i].min()
            max_val = X[:, feature_i].max()
            step = (max_val - min_val) / num_step
            values = np.arange(min_val, max_val + step, step = step)
            # find the threshold, that has the minimum error
            for thres in values:
                p = 1
                pred = np.ones(n_samples)
                pred[X[:, feature_i] < thres] = 0
                error = sum(w[y != pred])

                # if error > 0.5, reverse it, X[:, feature_i > thres]
                if(error > 0.5):
                    error = 1 - error
                    p = -1

                if error < min_error:
                    clf.polarity = p 
                    clf.threshold = thres
                    clf.feature_index = feature_i
                    min_error = error
        self.min_error = min_error
        clf.alpha = 0.5 * math.log((1.0 - self.min_error) / (self.min_error + 1e-10))
    

