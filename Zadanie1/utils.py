from os import system, name
import numpy
import pandas

datasets = {1: "./datasets/falldetection.csv",
            2: "./datasets/weatherAUS.csv",
            3: "./datasets/suicide-rates-overview-1985-to-2016.csv"}
dataset_sides = {1: 'L',
                 2: 'R',
                 3: 'R'}


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


def factorize(col):
    if col.dtype in [numpy.float64, numpy.float32, numpy.int32, numpy.int64]:
        return col
    else:
        return pandas.factorize(col)[0]
