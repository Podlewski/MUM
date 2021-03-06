import argparse

from utils import datasets_names, print_datasets_names, short_datasets_names


class ArgumentParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog='MUM - Task 3 - Classification',
            formatter_class=argparse.RawTextHelpFormatter,
            description='Lodz University of Technology (TUL)'
                        '\nMachine learning methods (Metody uczenia maszynowego)'
                        '\n\nTask 3 - Problem set 3 - Classification'
                        '\n\nAuthors:'
                        '\n  Paweł Galewicz\t234053'
                        '\n  Bartosz Jurczewski\t234067'
                        '\n  Zbigniew Nowacki\t234102'
                        '\n  Karol Podlewski\t234106'
                        '\n  Piotr Wardęcki\t234128')

        self.parser.add_argument(metavar='DATASET', dest='dataset', type=int,
                                 default=0, choices=range(1, 4),
                                 help='Select data set:\n'+print_datasets_names())

        self.parser.add_argument('-c', metavar='CLASSIFIER', dest='classifier',
                                 type=int, default=0, choices=range(1, 6),
                                 help='Use only one classifier:'
                                      '\n  [1] - Decision trees algorithm'
                                      '\n  [2] - Naive Bayes'
                                      '\n  [3] - Support-vector machine'
                                      '\n  [4] - k-nearest neighbors'
                                      '\n  [5] - Artificial neural network')

        self.parser.add_argument('-t', metavar='PERCENT', dest='training_percent',
                                 type=int, default=75, choices=range(1, 100),
                                 help='Set percent of training set')

        self.parser.add_argument('-a', metavar='N', dest='class_args',
                                 type=str, nargs='+', default=None,
                                 help='Arguments of chosen classifier')

        self.parser.add_argument('--time', dest='time', action='store_const',
                                 const=True, default=False,
                                 help='Measure time of classification')

        self.args = self.parser.parse_args()


    def get_arguments(self):
        self.args.dataset_name = datasets_names[self.args.dataset]
        self.args.short_dataset_name = short_datasets_names[self.args.dataset]
        self.args.training_fraction = self.args.training_percent / 100
        return self.args
