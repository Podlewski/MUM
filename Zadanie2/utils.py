from os import system, name

import numpy
import pandas

datasets = {1: "./datasets/Mall_Customers.csv",
            2: "./datasets/winequality-red.csv"}
datasets_names = {1: "Mall Customer segmentation",
                  2: "Red wine quality"}
datasets_clusters = {1: 5,
                     2: 8} 


def print_datasets_names(extra_spacing = ''):
    result = ""
    for key in datasets_names:
        result += extra_spacing + "[" + str(key) + "] - " + datasets_names[key] + '\n'
    return result


def clear():
    if name == 'nt':
        _ = system("cls")
    else:
        _ = system("clear")


def load_dataset(n):
    try:
        file = open(datasets[n])
    except KeyError:
        file = open(n)
    return pandas.read_csv(file).dropna()


def factorize(column):
    if column.dtype in [numpy.float64, numpy.float32, numpy.int32, numpy.int64]:
        return column
    else:
        return pandas.factorize(column)[0]
