import matplotlib.pyplot as plt
from numpy import flip, unique
import pandas as pd
from timeit import default_timer as timer

from argument_parser import ArgumentParser
from classifiers.bayes import Bayes
from classifiers.decision_tree import DecisionTree
from classifiers.k_neighbors import KNeighbors
from classifiers.stacking import Stacking
from utils import factorize, learning_curve_plot, print_basic_stats, diminish_data 

args = ArgumentParser().get_arguments()

data = pd.read_csv('../NYPD_Felony_Data_Fit.csv')
data = data.apply(factorize)
fraction = args.training_fraction

diminished_data, labels, label_name, drops_name = diminish_data(data, args.label, args.label_number, args.drops, args.drops_numbers, args.reduction)
unique_labels = unique(labels.values.ravel())

print_basic_stats(label_name, drops_name, args.training_percent)

classifiers = [DecisionTree(diminished_data, labels, unique, fraction),
               Bayes(diminished_data, labels, unique, fraction),
               KNeighbors(diminished_data, labels, unique, fraction),
               Stacking(diminished_data, labels, unique, fraction)]

fig, ax = plt.subplots()

for classifier in classifiers:
    start = timer()

    classifier.train()
    classifier.test()

    end = timer()
    time = end - start
    if args.time is False:
        time = None

    classifier.print_stats(time=time)

    # x, y, _ = classifier.get_roc_curve_plot()
    # ax.plot(x, y, label=classifier.short_name)

    # learning_curve_plot(classifier, feature_name + '_learning_curve_' +
    #                     classifier.short_name)

# if args.classifier is 0: 
    # ax.plot([0, 1], [0, 1], color='black', ls='--')
    # plt.xlim([-0.01, 1.01])
    # plt.ylim([-0.01, 1.01])
    # ax.legend()
    # fig.savefig(feature_name + '_ROC_Curve', bbox_inches='tight', dpi=300)

#     learning_curve_plot(classifiers, feature_name + '_learning_curve')
