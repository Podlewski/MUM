import collections
from os import system, name

import numpy
import pandas
from matplotlib import pyplot
from sklearn.decomposition import FastICA, PCA
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler


def factorize(column):
    if column.dtype in [numpy.float64, numpy.float32, numpy.int32, numpy.int64]:
        return column
    else:
        return pandas.factorize(column)[0]

def normalize(dataset):
    ds = dataset.values
    ds = StandardScaler().fit_transform(ds)
    return ds

def ica_reduction(dataset):
    ica = FastICA(n_components=2)
    ds = ica.fit_transform(dataset)
    return pandas.DataFrame(ds)

def pca_reduction(dataset):
    pca = PCA(n_components=2)
    ds = pca.fit_transform(dataset)
    return pandas.DataFrame(ds)

def prepare_data(data, label_name, label_number, drops_names, drops_numbers, reduction):
    if (label_name is None and label_number is None):
        raise Exception('Label is not provided')
    elif label_name is None:
        label_name = data.columns[label_number]

    labels = data[label_name]
    data = data.drop(columns=[label_name])
    
    if reduction == "ica":
        data = ica_reduction(normalize(data))
        drops_names = "ICA reduction"
    elif reduction == "pca":
        data = pca_reduction(normalize(data))
        drops_names = "PCA reduction"
    else:
        if drops_numbers is not None:
            if drops_names is not None:
                drops_names.extend(data.columns[drops_numbers].tolist()) 
            else:
                drops_names = data.columns[drops_numbers]

        if drops_names is not None:
            data = data.drop(columns=drops_names)

        if label_name == 'KY_CD':
            data = data.drop(columns=['PD_CD'], errors='ignore')
        elif label_name == 'PD_CD':
            data = data.drop(columns=['KY_CD'], errors='ignore')

    return data, labels, label_name, drops_names

def print_basic_stats(label, dropped_features, training_percent):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Label feauture:     {label}')
    if type(dropped_features) is list:
        print( 'Dropped feautures:  ', end = '')
        print(', '.join(dropped_features))
    else:
        print(f'Dropped feautures:  {dropped_features}')
    print(f'Training percent:   {training_percent}%')

def print_time(start_time, end_time):
    print(f'\nTime:  {round(end_time - start_time, 2)}s')

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
    ax[1].legend()
    ax[0].grid(alpha=0.2)
    ax[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(title, bbox_inches='tight', dpi=300)

