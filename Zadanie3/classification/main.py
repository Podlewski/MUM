import matplotlib.pyplot as plt
from numpy import flip
from timeit import default_timer as timer

from argument_parser import ArgumentParser
from classifiers.bayes import Bayes
from classifiers.decision_tree import DecisionTree
from classifiers.k_neighbors import KNeighbors
from classifiers.neural_network import NeuralNetwork
from classifiers.svm import SVM
from utils import factorize, load_dataset, learning_curve_plot, print_basic_stats

args = ArgumentParser().get_arguments()

data = load_dataset(args.dataset)
labels = data[data.columns[-1]].unique()
data = data.apply(factorize)

fraction = args.training_fraction
class_args = args.class_args
classifiers = [None, None, None, None, None]

if args.classifier is 0 or 1:
    classifiers[0] = DecisionTree(data, labels, fraction, class_args)
if args.classifier is 0 or 2:
    classifiers[1] = Bayes(data, labels, fraction, class_args)
if args.classifier is 0 or 3:
    classifiers[2] = SVM(data, labels, fraction, class_args)
if args.classifier is 0 or 4:
    classifiers[3] = KNeighbors(data, labels, fraction, class_args)
if args.classifier is 0 or 5:
    classifiers[4] = NeuralNetwork(data, labels, fraction, class_args)

classifiers = filter(None, classifiers)
fig, ax = plt.subplots()

print_basic_stats(args.dataset_name, args.training_percent)

for classifier in classifiers:
    start = timer()
    classifier.train()
    classifier.test()
    end = timer()

    if args.classifier is 0:
        print('\n~~~~~~~~~~~~~~~~~~~~ ' + classifier.name) 
        x, y, _ = classifier.get_roc_curve_plot()
        ax.plot(x, y, label=classifier.short_name)

    classifier.print_stats()

    if args.time is True:
        print(f'\nTime:  {round((end - start)*1000, 2)} ms')

    learning_curve_plot(classifier, 'learning_curve_' + args.short_dataset_name + '_' + classifier.short_name)

if args.classifier is 0: 
    ax.plot([0, 1], [0, 1], color='black', ls='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    ax.legend()
    fig.savefig('ROC_Curve_' + args.short_dataset_name, bbox_inches='tight', dpi=300)

    learning_curve_plot(classifiers, 'learning_curve_' + args.short_dataset_name)
