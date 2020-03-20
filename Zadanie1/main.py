from os import system, name
import numpy
import pandas

from argParser import ArgumentParser

from bayes import Bayes
from svm import SVM
from neuralNetwork import NeuralNetwork
from kneighbors import KNeighbors
from decisionTree import DecisionTree


def clear():
    if name == 'nt':
        _ = system("cls")
    else:
        _ = system("clear")


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
    print(f"Training data:\t{setup['training_percent']}%\n")


def load_dataset(file_path):
    return pandas.read_csv(file_path)


def factorize(col):
    if col.dtype in [numpy.float64, numpy.float32, numpy.int32, numpy.int64]:
        return col
    else:
        return pandas.factorize(col)[0]

ArgParser = ArgumentParser()

clear()
fraction = ArgParser.get_training_fraction()
data = load_dataset(ArgParser.get_dataset_path()).dropna().apply(factorize)
classifier = {1: DecisionTree(data, fraction),
              2: Bayes(data, fraction),
              3: SVM(data, fraction),
              4: KNeighbors(data, fraction),
              5: NeuralNetwork(data, fraction)}[ArgParser.get_classifier()]
classifier.train()
classifier.test()
# print_data()
print(classifier.get_metrics())
