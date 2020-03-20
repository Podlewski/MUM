import argparse
from os import system, name
import sys
import pandas


class ArgumentParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog='MUM - Task 1',
            description='Lodz University of Technology (TUL)'
                        '\nMachine learning methods (Metody uczenia maszynowego)'
                        '\n\nTask 1 - Problem set 1'
                        '\nAuthors:\nPaweł Galewicz'
                        '\nBartosz Jurczewski\nZbigniew Nowacki'
                        '\nKarol Podlewski\nPiotr Wardęcki')

        # data sets
        self.parser.add_argument('-d1', dest='dataset', action='store_const', const=1,
                                 help='Use data set "Fall detection data from China"')
        self.parser.add_argument('-d2', dest='dataset', action='store_const', const=2,
                                help='Use data set "Rain in Australia"')
        self.parser.add_argument('-d3', dest='dataset', action='store_const', const=3,
                                help='Use data set "Sucide rates overview"')
        # self.parser.add_argument('-d', metavar='STR', dest='dataset_path', type=string,
        #                          default=False, help='Insert path to data set file')                         

        # classifiers
        self.parser.add_argument('-c1', dest='classifier', action='store_const', const=1,
                                help='Use Decision trees algorithm to classify')
        self.parser.add_argument('-c2', dest='classifier', action='store_const', const=2,
                                help='Use Naive Bayes classifier to classify')
        self.parser.add_argument('-c3', dest='classifier', action='store_const', const=3,
                                help='Use Support-vector machine to classify')
        self.parser.add_argument('-c4', dest='classifier', action='store_const', const=4,
                                help='Use k-nearest neighbors algorith to classify')
        self.parser.add_argument('-c5', dest='classifier', action='store_const', const=5,
                                help='Use Artificial neural network algorithm to classify')

        # training percent
        self.parser.add_argument('-t', metavar='N', dest='training_percent', type=int,
                                default=0, help='Set percent of training set')

        # parse args
        self.args = self.parser.parse_args()


    def clear(self):
        if name == 'nt':
            _ = system('cls')
        else:
            _ = system('clear')


    def path(self, n):
        return {1: './datasets/falldetection.csv',
                2: './datasets/weatherAUS.csv',
                3: './datasets/suicide-rates-overview-1985-to-2016.csv'}[n]


    def get_dataset_path(self):
        if self.args.dataset is None:
            self.args.dataset = 0
            while 1 > self.args.dataset or self.args.dataset > 3:
                self.clear()
                self.args.dataset = int(input('Choose data set:\n'
                                '[1] Fall detection data from China\n'
                                '[2] Rain in Australia\n'
                                '[3] Suicide rates overview 1985-2016\n\n'
                                'Choice: '))
        return self.path(self.args.dataset)


    def get_classifier(self):
        if self.args.classifier is None:
            self.args.classifier = 0
            while 1 > self.args.classifier or self.args.classifier > 5:
                self.clear()
                self.args.classifier = int(input('Choose method:\n'
                                '[1] Decision trees algorithm\n'
                                '[2] Naive Bayes classifier\n'
                                '[3] Support-vector machine\n'
                                '[4] k-nearest neighbors algorithm\n'
                                '[5] Artificial neural network algorithm\n\n'
                                'Choice: '))
        return self.args.classifier


    def get_training_fraction(self):
        while 0 >= self.args.training_percent or self.args.training_percent >= 100:
            self.clear()
            self.args.training_percent = int(input('Percent of dataset used for training: '))
        return self.args.training_percent / 100

    