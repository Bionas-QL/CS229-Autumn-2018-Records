import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()  # By default, eps=1e-5 and theta_0=0
    clf.fit(x_train, y_train)
    util.plot(x_train, y_train, clf.theta, save_path=pred_path[:-4])
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval) >= 0.5
    print("GDA: number of correct predictions for eval_set " + pred_path[-5] + " is",
          np.sum((y_eval == y_pred).astype(int)))
    np.savetxt(pred_path, y_pred >= 0.5, "%d")

    # Part h
    if pred_path[-5] == '1':
        # Here we apply sqrt to x2. One can also try other families in Box-Cox Transformations e.g. log.
        x_train_trans = np.stack((x_train[:, 0], np.sqrt(x_train[:, 1]))).T
        x_eval_trans = np.stack((x_eval[:, 0], x_eval[:, 1], np.sqrt(x_eval[:, 2]))).T
        clf_trans = GDA()
        clf_trans.fit(x_train_trans, y_train)
        y_pred_trans = clf_trans.predict(x_eval_trans) >= 0.5
        print("GDA_trans: number of correct predictions for eval_set " + pred_path[-5] + " is",
              np.sum((y_eval == y_pred_trans).astype(int)))

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n + 1)
        num_label_1 = np.sum(y)
        num_label_0 = m - num_label_1
        phi = num_label_1 / m
        mu_0 = np.sum((1 - y) * x.T, axis=1) / num_label_0
        mu_1 = np.sum(y * x.T, axis=1) / num_label_1
        deviation = x - (y.reshape(-1, 1) * mu_1 + (1 - y).reshape(-1, 1) * mu_0)
        sigma = 1 / m * (deviation.T @ deviation)
        sigma_inv = np.linalg.inv(sigma)
        theta_0 = -1 / 2 * (
            np.dot(mu_1, np.dot(sigma_inv, mu_1)) -
            np.dot(mu_0, np.dot(sigma_inv, mu_0))
        ) + np.log((1 - phi) / phi)
        theta_remaining = np.dot(sigma_inv, mu_1 - mu_0)
        self.theta[0] = theta_0
        self.theta[1:] = theta_remaining
        return theta_0, theta_remaining
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
        # *** END CODE HERE
