import numpy as np


class SVM:
    def __init__(self, alpha=0.001, lambda_param=0.01, iterations=1000):
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.iterations = iterations

        #wWights and bias
        self.w = None  # Should I initialize it to something else ?
        self.b = None

    # Gradient descent fit
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Adjust the labels to have distinct classes (1,-1)
        y_modified = [-1 if i<0 else 1 for i in y]

        # Add the bias term to our weights
        self.w = np.zeros(n_features)
        self.b = 0

        # Graident descent here:
        for _ in range(self.iterations):
            for x_i, y_i in zip(X, y_modified):
                # if the prediciton is correct ( have the same sign )
                if y_i * (np.dot(x_i, self.w) - self.b) >= 1:
                    self.w -= self.alpha * (2 * self.w * self.lambda_param)
                    # Gradient with respect to the bias is 0 so no need to update
                else:
                    self.w -= self.alpha * \
                        (2 * self.w * self.lambda_param - np.dot(x_i , y_i))
                    self.b -= self.alpha * y_i

    def predict(self, x_new):
        # We simply subtitue our x_new in our linear forumla
        prediciton = np.dot(self.w, x_new) - self.b
        return sign(prediciton)

    # ----- helping method:


def sign(num: float):
    """
    @arg: a float
    @return: -1 if num <0 , 0 otherwise
    """
    return -1 if num < 0 else 1


