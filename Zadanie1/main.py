from timeit import default_timer as timer

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

if setup["classifier"] is 1:
    classifier = DecisionTree(data, lr, labels, fraction, class_args)
elif setup["classifier"] is 2:
    classifier = Bayes(data, lr, labels, fraction, class_args)
elif setup["classifier"] is 3:
    classifier = SVM(data, lr, labels, fraction, class_args)
elif setup["classifier"] is 4:
    classifier = KNeighbors(data, lr, labels, fraction, class_args)
elif setup["classifier"] is 5:
    classifier = NeuralNetwork(data, lr, labels, fraction, class_args)

start = timer()

classifier.train()
classifier.test()

end = timer()

classifier.print_stats(setup["dataset_name"], arg_parser.is_just_accuracy())

if arg_parser.is_time_measured() is True:
    print(f"\nTime:\t{round(end - start, 2)}s\n")