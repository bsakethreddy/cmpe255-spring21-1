import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDClassifier

detect_num = 5


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", IMAGE_DIR, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def random_digit():
    some_digit = X[36000]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

    save_fig("some_digit_plot")
    plt.show()


def load_and_sort():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml(name='mnist_784', version=1, cache=True)
        # fetch_openml() returns targets as strings
        mnist.target = mnist.target.astype(np.int8)
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
    return mnist


def sort_by_target(mnist_data):
    data_train = np.array(sorted(
        [(target, i) for i, target in enumerate(mnist_data.target[:60000])]))[:, 1]
    data_test = np.array(sorted(
        [(target, i) for i, target in enumerate(mnist_data.target[60000:])]))[:, 1]
    X_train = mnist_data.data.iloc[data_train]
    y_train = mnist_data.target.iloc[data_train]
    X_test = mnist_data.data.iloc[data_test + 60000]
    y_test = mnist_data.target.iloc[data_test + 60000]
    return X_train, y_train, X_test, y_test


def crossValidation_calculator(itr, data):
    total_rmse = 0
    for i in range(itr):
        X_train, Y_train, X_test, Y_test = sort_by_target(data)
        sgd_classifier = trainData_func(X_train, Y_train)
        pred_ytest = sgd_classifier.predict(X_test)
        actual_ytest = (Y_test == detect_num)
        ith_rmse = cal_rmse(pd.DataFrame(
            actual_ytest.to_numpy()), pd.DataFrame(pred_ytest))
        print(f"rmse {i + 1}: {ith_rmse}")
        total_rmse += ith_rmse
    return total_rmse / itr


def train_predict(some_digit, X_train, Y_train):
    sgd_classifier = train(X_train, Y_train)
    return sgd_classifier.predict(pd.DataFrame(some_digit).transpose())


def trainData_func(X_train, y_train):
    sh_index = np.random.permutation(60000)
    X_train, Y_train = X_train.iloc[sh_index], y_train.iloc[sh_index]
    y_trainClassifier = (Y_train == detect_num)
    sgd_classifier = SGDClassifier()
    sgd_classifier.fit(X_train, y_trainClassifier)
    return sgd_classifier


def cal_rmse(Y_actual, Y_hypothesis):
    return np.sqrt(mean_squared_error(Y_actual, Y_hypothesis))


if __name__ == "__main__":

    np.random.seed(42)
    # To plot pretty figures
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    # Where to save the figures
    PROJECT_ROOT_DIR = "."
    IMAGE_DIR = "FIXME"

    mnist_data = load_and_sort()
    crossValidation_score = crossValidation_calculator(5, mnist_data)
    print('Cross Validation Score is ', crossValidation_score)
