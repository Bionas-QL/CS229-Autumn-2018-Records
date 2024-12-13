import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    clf1 = LogisticRegression()
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    clf1.fit(x_train, t_train)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    util.plot(x_test, t_test, clf1.theta, save_path=pred_path_c[:-4])
    t_pred = clf1.predict(x_test) >= 0.5
    print("Q2 (c): number of correct predictions for eval_set " + pred_path[-5] + " is", np.sum((t_test == t_pred).astype(int)))
    np.savetxt(pred_path_c, t_pred, "%d")

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    clf2 = LogisticRegression()
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    clf2.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    util.plot(x_test, t_test, clf2.theta, save_path=pred_path_d[:-4])
    y_pred = clf1.predict(x_test) >= 0.5
    np.savetxt(pred_path_d, y_pred, "%d")

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_valid, t_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    alpha = np.sum(clf2.predict(x_valid) * y_valid) / np.sum(y_valid)
    correction = 1 + np.log(2 / alpha - 1) / clf2.theta[0]
    util.plot(x_test, t_test, clf2.theta, save_path=pred_path_e[:-4], correction=correction)
    t_pred = clf2.predict(x_valid) / alpha >= 0.5
    np.savetxt(pred_path_e, t_pred, "%d")

    # *** END CODER HERE
