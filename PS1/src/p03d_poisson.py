import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=lr, max_iter=10000)
    clf.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        def gradient(theta, x, y):
            """ Compute the gradient as a vector.

             Args:
                theta: Parameters of the model. Shape (n,).
                x: Training example inputs. Shape (m, n).
                y: Training example labels. Shape (m,).

            Returns:
                Outputs of shape (n,).
             """

            return np.dot(x.T, (y - np.exp(np.dot(x, theta))))

        if self.theta is None:
            theta = np.zeros(n)
        else:
            theta = self.theta

        i = 0
        while i < self.max_iter:
            new_theta = theta + self.step_size * gradient(theta, x, y) / m
            if np.linalg.norm(new_theta - theta, ord=1) < self.eps:
                theta = new_theta
                break
            theta = new_theta
            i += 1

        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***

        return np.exp(np.dot(x, self.theta))
        # *** END CODE HERE ***
