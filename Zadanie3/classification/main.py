from numpy import flip
from timeit import default_timer as timer

from argument_parser import ArgumentParser
from classifiers.bayes import Bayes
from classifiers.decision_tree import DecisionTree
from classifiers.k_neighbors import KNeighbors
from classifiers.neural_network import NeuralNetwork
from classifiers.svm import SVM
from utils import factorize, load_dataset


args = ArgumentParser().get_arguments()

data = load_dataset(args.dataset)
labels = data[data.columns[-1]].unique()
data = data.apply(factorize)

fraction = args.training_fraction
class_args = args.class_args


if args.classifier is 0:
    pass
elif args.classifier is 1:
    classifier = DecisionTree(data, labels, fraction, class_args)
elif args.classifier is 2:
    classifier = Bayes(data, labels, fraction, class_args)
elif args.classifier is 3:
    classifier = SVM(data, labels, fraction, class_args)
elif args.classifier is 4:
    classifier = KNeighbors(data, labels, fraction, class_args)
elif args.classifier is 5:
    classifier = NeuralNetwork(data, labels, fraction, class_args)

start = timer()

classifier.train()
classifier.test()

end = timer()
time = end - start

classifier.print_stats(args.dataset_name, time)
