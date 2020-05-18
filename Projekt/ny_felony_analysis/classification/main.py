import matplotlib.pyplot as plt
from numpy import flip
import pandas as pd
from timeit import default_timer as timer

from argument_parser import ArgumentParser
from classifiers.bayes import Bayes
from classifiers.decision_tree import DecisionTree
from classifiers.k_neighbors import KNeighbors
from classifiers.neural_network import NeuralNetwork
from classifiers.svm import SVM
from utils import factorize, learning_curve_plot, print_basic_stats

args = ArgumentParser().get_arguments()

data = pd.read_csv('../NYPD_Felony_Data.csv')

data.drop([data.columns[2],  data.columns[3],  data.columns[13],
           data.columns[14]], axis='columns', inplace=True)
data.dropna(inplace=True)

labels_list = [['KY_CD', data['KY_CD'].unique()],
               ['SUSP_SEX', data['SUSP_SEX'].unique()],
               ['VIC_SEX', data['VIC_SEX'].unique()]]

data = data.apply(factorize)

fraction = args.training_fraction
class_args = args.class_args

print_basic_stats(args.training_percent)

for feature_name, labels in labels_list: 
    print('\n~~~~~~~~~~~~~~~~~~~~ ' + feature_name) 

    if args.classifier is 0:
        classifiers = [DecisionTree(data, labels, fraction),
                       Bayes(data, labels, fraction),
                       # SVM(data, labels, fraction),
                       KNeighbors(data, labels, fraction),
                       # NeuralNetwork(data, labels, fraction)
                       ]
    elif args.classifier is 1:
        classifiers = [DecisionTree(data, labels, fraction)]
    elif args.classifier is 2:
        classifiers = [Bayes(data, labels, fraction)]
    elif args.classifier is 3:
        classifiers = [SVM(data, labels, fraction)]
    elif args.classifier is 4:
        classifiers = [KNeighbors(data, labels, fraction)]
    elif args.classifier is 5:
        classifiers = [NeuralNetwork(data, labels, fraction)]

    fig, ax = plt.subplots()

    for classifier in classifiers:
        start = timer()
        classifier.train()
        classifier.test()
        end = timer()

        print('\n~~~~~~~~~~ ' + classifier.name) 

        if args.classifier is 0:
            x, y, _ = classifier.get_roc_curve_plot()
            ax.plot(x, y, label=classifier.short_name)

        classifier.print_stats()

        if args.time is True:
            print(f'\nTime:  {round((end - start), 2)} s')

        learning_curve_plot(classifier, feature_name + '_learning_curve_' +
                            classifier.short_name)

    if args.classifier is 0: 
        ax.plot([0, 1], [0, 1], color='black', ls='--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        ax.legend()
        fig.savefig(feature_name + '_ROC_Curve', bbox_inches='tight', dpi=300)

        learning_curve_plot(classifiers, feature_name + '_learning_curve')
