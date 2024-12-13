import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau=0.5)
    clf.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    mse = np.mean((y_pred - y_eval) ** 2)
    print("Q5 (b): when tau=0.5, the MSE is", mse)

    # Plot validation predictions on top of training set

    plot(x_eval, y_eval, y_pred, "Validation Set")

    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


def plot(x_eval, y_eval, y_pred, title):
    plt.figure()
    plt.plot(x_eval[:, 1], y_eval, 'bx', label='y_eval')
    plt.plot(x_eval[:, 1], y_pred, 'ro', label='y_pred')
    plt.suptitle(title)

class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        res = np.zeros(m)
        for i in range(m):
            w = np.exp(- np.sum((self.x - x[i, :]) ** 2, axis=1) / (2 * (self.tau ** 2)))
            self.theta = np.dot(np.linalg.inv((self.x.T * w) @ self.x) @ (self.x.T * w), self.y)
            res[i] = np.dot(self.theta, x[i, :])
        return res

        # *** END CODE HERE ***
