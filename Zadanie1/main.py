"""
args:
- [d1 | d2 | d3]            -   for use of "built-in" datasets provided with the program
- file=<path to dataset>    -   for use of user-provided dataset,
                                needs c=[L | P] to work,
                                doesn't work when [d1 | d2 | d3] is provided as well
- c=[L | R]                 -   needed to specify side of column with classes - left or right,
                                works with file=<path to dataset> only
- [m1 | m2 | m3 | m4 | m5]  -   specify classifier
- tp=<percentage>           -   specify percentage of dataset used for training

Examples:
> main.py file=./datasets/weatherAUS.csv m2 tp=70 c=R
> main.py d3 m4 tp=60
"""

import sys
import pandas

from classifiers.bayes import Bayes
from classifiers.decisionTree import DecisionTree
from classifiers.kneighbors import KNeighbors
from classifiers.neuralNetwork import NeuralNetwork
from classifiers.svm import SVM
from utils import clear, load_dataset, factorize, dataset_sides


def print_stats(metrics=None):
    try:
        dataset_name = {1: "Fall detection data from China",
                        2: "Rain in Australia",
                        3: "Suicide rates overview 1985-2016"}[setup["dataset"]]
    except KeyError:
        dataset_name = "user-provided"
    print(f"Data set:\t{dataset_name}")
    method_name = {1: "Decision trees algorithm",
                   2: "Naive Bayes classifier",
                   3: "Support-vector machine",
                   4: "k-nearest neighbors algorithm",
                   5: "Artificial neural network algorithm"}[setup["method"]]
    print(f"Classification:\t{method_name}")
    print(f"Training data:\t{setup['training_percent']}%\n")
    print(metrics)


setup = {}
data = None
lr = None
args = pandas.Series(sys.argv[1:])
choices_data = {"d1", "d2", "d3"}
choices_method = {"m1", "m2", "m3", "m4", "m5"}

if args.isin(choices_data).any():
    intersection = set(args) & choices_data
    if len(intersection) > 1:
        raise Exception("Multiple arguments for data choice")
    setup["dataset"] = int(intersection.pop()[1:])
    lr = dataset_sides[setup["dataset"]]
    data = load_dataset(setup["dataset"])
elif any("file=" in s for s in args) and any(("c=L" in s) or ("c=R" in s) for s in args):
    matching_file = [s for s in args if "file=" in s]
    matching_classes_side = [s for s in args if ("c=L" in s) or ("c=R" in s)]
    if len(matching_file) > 1 or len(matching_classes_side) > 1:
        raise Exception("Multiple arguments for data set filepath")
    lr = matching_classes_side[0][-1:]
    data = load_dataset(matching_file[0][5:])
else:
    while True:
        clear()
        setup["dataset"] = int(input("Choose data set:\n"
                                     "[1] Fall detection data from China\n"
                                     "[2] Rain in Australia\n"
                                     "[3] Suicide rates overview 1985-2016\n\n"
                                     "Choice: "))
        if 1 <= setup["dataset"] <= 3:
            break
    lr = dataset_sides[setup["dataset"]]
    data = load_dataset(setup["dataset"])
labels = data[data.columns[0 if lr == 'L' else -1]].unique()
data = data.apply(factorize)

if args.isin(choices_method).any():
    intersection = set(args) & choices_method
    if len(intersection) > 1:
        raise Exception("Multiple arguments for method choice")
    setup["method"] = int(intersection.pop()[1:])
else:
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

if any("tp=" in s for s in args):
    matching_file = [s for s in args if "tp=" in s]
    if len(matching_file) > 1:
        raise Exception("Multiple arguments for training percentage")
    else:
        try:
            setup["training_percent"] = float(matching_file[0][3:])
        except ValueError:
            raise Exception("Wrong format for training percentage argument")
else:
    while True:
        clear()
        setup["training_percent"] = float(input("Percent of dataset used for training: "))
        if 0 < setup["training_percent"] < 100:
            break
setup["training_fraction"] = setup["training_percent"] / 100

clear()
classifier = {1: DecisionTree(data, lr, labels, setup["training_fraction"]),
              2: Bayes(data, lr, labels, setup["training_fraction"]),
              3: SVM(data, lr, labels, setup["training_fraction"]),
              4: KNeighbors(data, lr, labels, setup["training_fraction"]),
              5: NeuralNetwork(data, lr, labels, setup["training_fraction"])}[setup["method"]]
classifier.train()
classifier.test()
print_stats(
    classifier.get_metrics()
)
