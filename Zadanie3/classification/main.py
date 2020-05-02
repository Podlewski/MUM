import matplotlib.pyplot as plt
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

if args.classifier is not 0:
    if args.classifier is 1:
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

    classifier.print_stats(args.dataset_name)

    if args.time is True:
        print(f'\nTime:  {round((end - start)*1000, 2)} ms')

else:
    classifiers = [DecisionTree(data, labels, fraction, class_args),
                   Bayes(data, labels, fraction, class_args),
                   SVM(data, labels, fraction, class_args),
                   KNeighbors(data, labels, fraction, class_args),
                   NeuralNetwork(data, labels, fraction, class_args)]
    
    for classifier in classifiers:

        classifier.train()
        classifier.test()

        fpr, tpr, _ = classifier.get_roc_curve()

        plt.plot(fpr, tpr, label=classifier.name)

    plt.plot([0, 1], [0, 1], color='black', ls='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig('ROC_Curve_' + args.short_dataset_name, bbox_inches='tight', dpi=300)