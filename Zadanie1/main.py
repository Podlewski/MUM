from argParser import ArgumentParser
from classifiers.bayes import Bayes
from classifiers.decisionTree import DecisionTree
from classifiers.kneighbors import KNeighbors
from classifiers.neuralNetwork import NeuralNetwork
from classifiers.svm import SVM
from utils import clear, load_dataset, factorize, dataset_sides


arg_parser = ArgumentParser()

setup = {
    "dataset": arg_parser.get_dataset_path(),
    "dataset_name": arg_parser.get_dataset_name(),
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

classifier = {1: DecisionTree(data, lr, labels, fraction, class_args),
              2: Bayes(data, lr, labels, fraction, class_args),
              3: SVM(data, lr, labels, fraction, class_args),
              4: KNeighbors(data, lr, labels, fraction, class_args),
              5: NeuralNetwork(data, lr, labels, fraction, class_args)}[setup["classifier"]]
classifier.train()
classifier.test()

classifier.print_stats(setup["dataset_name"])
