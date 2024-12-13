import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression, plot


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    best_tau = tau_values[0]
    clf = LocallyWeightedLinearRegression(best_tau)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_valid)
    best_mse = np.mean((y_pred - y_valid) ** 2)
    for tau in tau_values:
        clf.tau = tau
        y_pred = clf.predict(x_valid)
        title = "tau = " + str(tau)
        plot(x_valid, y_valid, y_pred, title=title)
        mse = np.mean((y_pred - y_valid) ** 2)
        if mse < best_mse:
            best_tau = tau
            best_mse = mse
    print("Q5 (c): the best tau is", best_tau)

    # Fit a LWR model with the best tau value
    clf.tau = best_tau
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = clf.predict(x_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print("Q5 (c): the MSE on the test case is", mse)
    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred)
    # Plot data
    plot(x_test, y_test, y_pred, title="Test Set")
    plt.show()
    # *** END CODE HERE ***
