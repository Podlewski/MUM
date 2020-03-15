from os import system, name
import pandas
from bayes import Bayes
from svm import SVM


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
    file = open(path(n))
    return pandas.read_csv(file)


def print_data():
    dataset_name = {1: "Fall detection data from China",
                    2: "Rain in Australia",
                    3: "Suicide rates overview 1985-2016"}[setup["dataset"]]
    print(f"Data set:\t{dataset_name}")
    method_name = {1: "Decision trees algorithm",
                   2: "Naive Bayes classifier",
                   3: "Support-vector machine",
                   4: "k-nearest neighbors algorithm",
                   5: "Artificial neural network algorithm"}[setup["method"]]
    print(f"Classification:\t{method_name}")
    print(f"Training data:\t{setup['training_precent']}%\n")


setup = {}

while True:
    clear()
    setup["dataset"] = int(input("Choose data set:\n"
                                 "[1] Fall detection data from China\n"
                                 "[2] Rain in Australia\n"
                                 "[3] Suicide rates overview 1985-2016\n\n"
                                 "Choice: "))
    if 1 <= setup["dataset"] <= 3:
        break



while True:
    clear()
    setup["method"] = int(input("Choose method:\n"
                                "[1] Decision trees algorithm\n"
                                "[2] Naive Bayes classifier\n"
                                "[3] Support-vector machine\n"
                                "[4] k-nearest neighbors algorithm\n"
                                "[5] Artificial neural network algorithm\n\n"
                                 "Choice: "))
    if 1 <= setup["method"] <= 5:
        break

while True:
    clear()
    setup["training_precent"] = float(input("Precent of dataset used for training: "))
    if 0 < setup["training_precent"] < 100:
        break
setup["training_fraction"] = setup["training_precent"] / 100

clear()
data = load_dataset(setup["dataset"])
classifier = {1: Bayes(data, setup["training_fraction"]),
              2: Bayes(data, setup["training_fraction"]),
              3: SVM(data, setup["training_fraction"]),
              4: Bayes(data, setup["training_fraction"]),
              5: Bayes(data, setup["training_fraction"])}[setup["method"]]
classifier.train()
classifier.test()
print_data()
print(classifier.get_metrics())
