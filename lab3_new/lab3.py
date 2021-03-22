import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_selection import RFE


class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin',
                     'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0,
                                names=col_names, usecols=col_names)
        # print(self.pima.head())
        # print(self.pima.isnull().sum())
        self.X_test = None
        self.y_test = None

    def set_glucose(self):
        newGlucose = pd.Series(
            ["normal", "prediabetes", "diabetic"], dtype="category")
        self.pima["newGlucose"] = newGlucose

        self.pima.loc[self.pima["glucose"] < 140, "newGlucose"] = newGlucose[0]
        self.pima.loc[((self.pima["glucose"] > 140) & (
            self.pima["glucose"] <= 200)), "newGlucose"] = newGlucose[1]
        self.pima.loc[self.pima["glucose"] > 200, "newGlucose"] = newGlucose[2]

    # Deriving new features from the current feature set
    def set_bmi(self):
        newBmi = pd.Series(
            ["underweight", "normal", "overweight", "obesity"], dtype="category")
        self.pima["newBmi"] = newBmi

        self.pima.loc[self.pima["bmi"] < 18.5, "newBmi"] = newBmi[0]
        self.pima.loc[((self.pima["bmi"] >= 18.5) & (
            self.pima["bmi"] <= 25)), "newBmi"] = newBmi[1]
        self.pima.loc[((self.pima["bmi"] >= 25) & (
            self.pima["bmi"] <= 30)), "newBmi"] = newBmi[2]
        self.pima.loc[self.pima["bmi"] > 30, "newBmi"] = newBmi[3]

    def feature_engineering(self):
        # Creating new logical feature variables from the existing features for better prediction values
        # we can create two new features new_age and new_bmi

        self.set_bmi()
        self.set_glucose()

        self.pima = pd.get_dummies(
            self.pima, columns=["newGlucose", "newBmi"])
        # print(self.pima.head())

    def define_feature(self):
        model = LogisticRegression(max_iter=10000)

        self.pima = self.pima[['newGlucose_normal', 'newGlucose_prediabetes',
                               'newBmi_obesity', 'newBmi_overweight', 'newBmi_underweight', 'pregnant', 'pedigree', 'insulin', 'skin', 'bp', 'age', 'label']]
        array = self.pima.values
        X = array[:, 0:11]

        y = array[:, 11]

        rfe = RFE(model, 7)
        fit = rfe.fit(X, y)

        reduced_dataset = self.pima.iloc[:, :-1].loc[:, fit.support_]
        return reduced_dataset, self.pima.label

    def train(self):

        # split X and y into training and testing sets
        X, y = self.define_feature()
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, random_state=12345)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(max_iter=10)
        logreg.fit(X_train, y_train)
        return logreg

    def predict(self):
        model = self.train()
        score = model.score(self.X_test, self.y_test)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class

    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)

    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()

    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)


if __name__ == "__main__":
    classifer = DiabetesClassifier()
    classifer.feature_engineering()
    # classifer.define_feature()
    result = classifer.predict()
    print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix=${con_matrix}")
