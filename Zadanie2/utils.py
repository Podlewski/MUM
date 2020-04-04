from os import system, name

import numpy
import pandas

datasets = {1: "./datasets/Mall_Customers.csv",
            2: "./datasets/College.csv"}
dataset_names = {1: "Mall Customer segmentation",
                 2: "Statistics for US Colleges"}


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
