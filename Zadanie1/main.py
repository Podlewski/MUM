from argParser import ArgumentParser
from classifiers.bayes import Bayes
from classifiers.decisionTree import DecisionTree
from classifiers.kneighbors import KNeighbors
from classifiers.neuralNetwork import NeuralNetwork
from classifiers.svm import SVM
from utils import clear, load_dataset, factorize, dataset_sides


def print_stats(metrics=None):
    dataset_name = {1: "Fall detection data from China",
                    2: "Rain in Australia",
                    3: "Suicide rates overview 1985-2016"}[arg_parser.args.dataset]
    print(f"Data set:         {dataset_name}")
    method_name = {1: "Decision trees algorithm",
                   2: "Naive Bayes classifier",
                   3: "Support-vector machine",
                   4: "k-nearest neighbors algorithm",
                   5: "Artificial neural network algorithm"}[setup["classifier"]]
    print(f"Classification:   {method_name}")
    print(f"Training percent: {setup['training_fraction']*100}%\n")
    print(metrics)


arg_parser = ArgumentParser()

setup = {
    "dataset": arg_parser.get_dataset_path(),
    "classifier": arg_parser.get_classifier(),
    "training_fraction": arg_parser.get_training_fraction()
}
clear()

data = load_dataset(setup["dataset"])
lr = dataset_sides[arg_parser.args.dataset]
labels = data[data.columns[0 if lr == 'L' else -1]].unique()
data = data.apply(factorize)

fraction = setup["training_fraction"]
class_args = arg_parser.get_classifier_arguments()

classifier = {1: DecisionTree(data, lr, labels, fraction),
              2: Bayes(data, lr, labels, fraction),
              3: SVM(data, lr, labels, fraction, class_args),
              4: KNeighbors(data, lr, labels, fraction),
              5: NeuralNetwork(data, lr, labels, fraction)}[setup["classifier"]]
classifier.train()
classifier.test()

print_stats(
    classifier.get_metrics()
)
