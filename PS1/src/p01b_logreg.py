import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    clf = LogisticRegression() # By default, eps=1e-5 and theta_0=0
    clf.fit(x_train, y_train)
    util.plot(x_train, y_train, clf.theta, save_path=pred_path[:-4])
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval) >= 0.5
    print("LogisticRegression: number of correct predictions for eval_set " + pred_path[-5] + " is" ,np.sum((y_eval == y_pred).astype(int)))
    np.savetxt(pred_path, y_pred, "%d")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        def h(theta, x):
            """ Compute the vectorized version of h (Each entry corresponds to the value of h at a single sample).

            Args:
                theta: Parameters of the model. Shape (n,).
                x: Training example inputs. Shape (m, n).

            Returns:
                Outputs of shape (m,).
            """
            return 1 / (1 + np.exp(- np.dot(x, theta)))
            # Here np.dot works like the matrix multiplication (m, n) * (n, 1).

        def gradient(theta, x, y):
            """ Compute the gradient as a vector.

            Args:
                theta: Parameters of the model. Shape (n,).
                x: Training example inputs. Shape (m, n).
                y: Training example labels. Shape (m,).

            Returns:
                Outputs of shape (n,).
            """

            return 1 / m * np.dot(x.T, h(theta, x) - y)
            # Here np.dot works like the matrix multiplication (n, m) * (m, 1).

        def Hessian(theta, x, y):
            """ Compute the Hessian as a 2-D array.

            Args:
                theta: Parameters of the model. Shape (n,).
                x: Training example inputs. Shape (m, n).
                y: Training example labels. Shape (m,).

            Returns:
                Outputs of shape(n, n)
            """

            return 1 / m * np.dot(x.T, (h(theta, x) * (1 - h(theta, x)) * x.T).T)
            # Here np.dot works like the matrix multiplication (n, m) * (m, n).

        if self.theta == None:
            theta = np.zeros(n)
        else:
            theta = self.theta

        i = 0
        while i < self.max_iter:
            new_theta = theta - np.dot(np.linalg.inv(Hessian(theta, x, y)), gradient(theta, x, y))
            if np.linalg.norm(new_theta - theta) < self.eps:
                theta = new_theta
                break
            theta = new_theta
            i += 1

        self.theta = theta
        # *** END CODE HERE ***


    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        return 1 / (1 + np.exp(- np.dot(x, self.theta)))
        # *** END CODE HERE ***
