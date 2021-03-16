import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


class HousePrice:

    def __init__(self):
        COLUMNS = ["CRIM", "ZN", 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                   'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        self.df = pd.read_csv(
            'housing.csv', delim_whitespace=True, header=None, names=COLUMNS)
        # print(f'${self.df.shape} lines loaded')

    def findBestFit(self):
        # linear regression, take two variables to find best fit
        # Choosing 'RM' column
        plt.scatter(self.df.RM, self.df.MEDV)
        plt.xlabel("RM")
        plt.ylabel("MEDV")

        # plt.show(block=True)
        pass

    def splitData(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.df.RM, self.df.MEDV, random_state=0)
        return X_train, X_test, y_train, y_test

    def linear_regression(self, X_train, X_test, y_train, y_test):
        linear_regression = LinearRegression()
        x = X_train.values.reshape(-1, 1)
        y = y_train.values.reshape(-1, 1)

        x_test = X_test.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)
        linear_regression.fit(x, y)
        y_pred = linear_regression.predict(x_test)
        plt.scatter(X_test, y_test)
        plt.xlabel("RM")
        plt.ylabel("MEDV")
        plt.plot(x_test, y_pred, color="red")
        plt.savefig('linear_regression_bestfit_line')
        # plt.show(block=True)
        return y_pred
        # return x_test, y_test, linear_regression
        pass

    def polynomial_linear_regression(self, X_train, X_test, y_train, y_test, degree):
        poly = PolynomialFeatures(degree=degree)
        x = X_train.values.reshape(-1, 1)
        y = y_train.values.reshape(-1, 1)
        x_test = X_test.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)

        x_poly = poly.fit_transform(x)
        poly.fit(x, y)
        polyModel = LinearRegression()
        polyModel.fit(x_poly, y_train)
        y_poly_pred = polyModel.predict(poly.fit_transform(x_test))
        plt.scatter(X_test, y_test)
        plt.xlabel("RM")
        plt.ylabel("MEDV")
        plt.plot(x_test, y_poly_pred, color="red")
        plt.savefig('polynomial_regression_bestfit_line' + str(degree))
        # plt.show(block=True)
        return y_poly_pred
        pass

    def multiple_linear_regression(self):
        X = self.df.iloc[:, :-1]
        Y = self.df.iloc[:, -1]
        # Selecting highly correlated features
        plt.figure(figsize=(12, 10))
        cor = self.df.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.savefig('multiple_regression_heatMap')
        # plt.show()
        plt.clf()
        cor_target = abs(cor["MEDV"])
        relevant_features = cor_target[cor_target > 0.5]
        feature_set = self.df[['RM', 'PTRATIO', 'LSTAT']]
        # splitting the data
        XM_train, XM_test, ym_train, ym_test = train_test_split(
            feature_set, self.df.MEDV)
        multiple_linear_regression = LinearRegression()
        multiple_linear_regression.fit(XM_train, ym_train)
        y_multi_pred = multiple_linear_regression.predict(XM_test)

        # plotting the y_pred
        plt.scatter(XM_test['RM'], ym_test)
        plt.xlabel("RM")
        plt.ylabel("MEDV")
        plt.plot(XM_test['RM'], y_multi_pred, color="red")
        plt.savefig('multiple_regression_bestfit_line')
        # plt.show(block=True)
        return ym_test, y_multi_pred, XM_train

    def rmse(self, y_test, y_pred, funcName):
        # calculating the rmse score
        print("rmse value for " + funcName, mean_squared_error(
            y_test, y_pred, squared=False))
        return mean_squared_error(
            y_test, y_pred, squared=False)

    def rSquare(self, y_test, y_pred, funcName):
        print("r2_score for " +
              funcName, r2_score(y_test, y_pred))
        return r2_score(y_test, y_pred)


def funcLinearRegression():
    obj = HousePrice()
    obj.findBestFit()
    X_train, X_test, y_train, y_test = obj.splitData()
    y_pred = obj.linear_regression(X_train, X_test, y_train, y_test)
    obj.rmse(y_test, y_pred, "LinearRegression =")
    obj.rSquare(y_test, y_pred, "LinearRegression =")


def funcPolynomialLinearRegression():
    obj1 = HousePrice()

    X_train, X_test, y_train, y_test = obj1.splitData()
    y_pred = obj1.polynomial_linear_regression(
        X_train, X_test, y_train, y_test, 2)
    obj1.polynomial_linear_regression(
        X_train, X_test, y_train, y_test, 20)
    obj1.rmse(y_test, y_pred, "PolynomialLinearRegression =")
    obj1.rSquare(y_test, y_pred,  "PolynomialLinearRegression =")


def funcMultinomialLinearRegression():
    obj2 = HousePrice()
    y_test, y_pred, X_train = obj2.multiple_linear_regression()
    rmse = obj2.rmse(y_test, y_pred, "Multiple Linear Regression")
    rSquare = obj2.rSquare(y_test, y_pred, "Multiple Linear Regression")
    adjustedRSquare = 1 - (1-rSquare)*(len(y_test) - 1) / \
        (len(y_test) - X_train.shape[1] - 1)
    print("Adjusted Rsquare score = ", adjustedRSquare)


def main():
    print("LINEAR REGRESSION")
    funcLinearRegression()
    print("Polynomial Regression")
    funcPolynomialLinearRegression()
    print("Multiple Regression")
    funcMultinomialLinearRegression()


if __name__ == "__main__":
    main()
