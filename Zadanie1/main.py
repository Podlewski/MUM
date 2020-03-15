from os import system, name
import pandas
import numpy
from bayes import Bayes


def clear():
    if name == 'nt':
        _ = system("cls")
    else:
        _ = system("clear")


def path(n):
    return {
        1: "./datasets/falldetection.csv",
        2: "./datasets/weatherAUS.csv",
        3: "./datasets/suicide-rates-overview-1985-to-2016.csv"
    }[n]


def load_dataset(n):
    f = open(path(n))
    return pandas.read_csv(f)


setup = {}

while True:
    clear()
    setup["dataset"] = int(input("Choose data set:\n"
                                 "[1] Fall detection data from China\n"
                                 "[2] Rain in Australia\n"
                                 "[3] Suicide rates overview 1985-2016\t\t"))
    if 1 <= setup["dataset"] <= 3:
        break

while True:
    clear()
    setup["method"] = int(input("Choose method:\n"
                                "[1] Decision trees algorithm\n"
                                "[2] Naive Bayes classifier\n"
                                "[3] Support-vector machine\n"
                                "[4] k-nearest neighbors algorithm\n"
                                "[5] Artificial neural network algorithm\t\t"))
    if 1 <= setup["method"] <= 5:
        break

clear()
data = load_dataset(setup["dataset"])
classifier = {1: Bayes(data),
              2: Bayes(data),
              3: Bayes(data),
              4: Bayes(data),
              5: Bayes(data)}[setup["dataset"]]
classifier.train()
classifier.test()
