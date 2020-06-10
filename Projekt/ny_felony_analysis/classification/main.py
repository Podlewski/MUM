import matplotlib.pyplot as plt
from numpy import flip, unique
import pandas as pd
from timeit import default_timer as timer

from argument_parser import ArgumentParser
from classifiers.bayes import Bayes
from classifiers.decision_tree import DecisionTree
from classifiers.k_neighbors import KNeighbors
from classifiers.stacking import Stacking
from classifiers.bagging import Bagging
from classifiers.boosting import Boosting
from utils import factorize, learning_curve_plot, print_basic_stats, prepare_data, print_time

args = ArgumentParser().get_arguments()

data = pd.read_csv('../NYPD_Felony_Data_Fit.csv')
data = data.apply(factorize)
fraction = args.training_fraction

diminished_data, labels, label_name, drops_name = prepare_data(data, args.label, args.label_number, args.drops, args.drops_numbers, args.reduction)
unique_labels = unique(labels.values.ravel())

print_basic_stats(label_name, drops_name, args.training_percent)

classifiers = [DecisionTree(diminished_data, labels, unique, fraction),
               Bayes(diminished_data, labels, unique, fraction),
               KNeighbors(diminished_data, labels, unique, fraction)]

if args.stacking_classifier is True:
    classifiers.append(Stacking(diminished_data, labels, unique, fraction))

if args.bagging_classifier is True:
    classifiers.append(Bagging(diminished_data, labels, unique, fraction))

if args.boosting_classifier is True:
    classifiers.append(Boosting(diminished_data, labels, unique, fraction))

fig, ax = plt.subplots()

for classifier in classifiers:
    start = timer()
    classifier.train()
    classifier.test()
    end = timer()

    classifier.print_stats()
    if args.time is True:
        print_time(start, end)

    if args.roc_curve is True: 
        x, y, _ = classifier.get_roc_curve_plot()
        ax.plot(x, y, label=classifier.short_name)

if args.roc_curve is True: 
    ax.plot([0, 1], [0, 1], color='black', ls='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    ax.legend()
    fig.savefig('ROC_Curve', bbox_inches='tight', dpi=300)

if args.learning_curve is True:
    learning_curve_plot(classifiers, 'learning_curve')
