import collections
from os import system, name

import numpy
import pandas
from matplotlib import pyplot
from sklearn.model_selection import learning_curve


def factorize(column):
    if column.dtype in [numpy.float64, numpy.float32, numpy.int32, numpy.int64]:
        return column
    else:
        return pandas.factorize(column)[0]


def print_basic_stats(training_percent):
    print(f'Training percent:  {training_percent}%')


def learning_curve_add_subplot(classifier, ax):
    train_sizes, train_scores, test_scores = learning_curve(
        classifier.model,
        numpy.concatenate((classifier.training_data, classifier.test_data)),
        numpy.concatenate((classifier.training_target_values, classifier.test_target_values)),
        train_sizes=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    )
    ax[0].plot(
        train_sizes,
        train_scores.mean(1),
        label=classifier.short_name
    )
    ax[1].plot(
        train_sizes,
        test_scores.mean(1),
        label=classifier.short_name
    )


def learning_curve_plot(classifier, title='pic'):
    fig, ax = pyplot.subplots(1, 2, sharey='all', figsize=(9, 6))
    if isinstance(classifier, collections.Sequence):
        for c in classifier:
            learning_curve_add_subplot(c, ax)
    else:
        learning_curve_add_subplot(classifier, ax)
    ax[0].set_title('Training', fontsize='medium')
    ax[1].set_title('Validation', fontsize='medium')
    ax[0].set_xlabel('Train size')
    ax[1].set_xlabel('Train size')
    ax[0].set_ylabel('Score')
    ax[1].tick_params(length=0)
    # ax[0].legend(prop={'size': 7})
    ax[1].legend()
    ax[0].grid(alpha=0.2)
    ax[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(title, bbox_inches='tight', dpi=300)
