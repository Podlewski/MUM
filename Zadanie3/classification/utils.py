from os import system, name

import numpy
import pandas

datasets = { 
            2: './datasets/weatherAUS.csv',
            3: './datasets/diabetes.csv'}
datasets_names = {
                  2: 'Rain in Australia',
                  3: 'Pima Indians Diabetes Database'}
short_datasets_names = {
                        2: 'Rain in Australia',
                        3: 'Pima Indians Diabetes Database'}


def factorize(column):
    if column.dtype in [numpy.float64, numpy.float32, numpy.int32, numpy.int64]:
        return column
    else:
        return pandas.factorize(column)[0]


def load_dataset(n):
    try:
        file = open(datasets[n])
    except KeyError:
        file = open(n)
    return pandas.read_csv(file).dropna()


def print_datasets_names():
    result = ''
    for key in datasets_names:
        result += '  [' + str(key) + '] - ' + datasets_names[key] + '\n'
    return result

