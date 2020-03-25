import argparse

from utils import clear, datasets, dataset_names


class ArgumentParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog='MUM - Task 1', formatter_class=argparse.RawTextHelpFormatter,
            description="Lodz University of Technology (TUL)"
                        "\nMachine learning methods (Metody uczenia maszynowego)"
                        "\n\nTask 1 - Problem set 1"
                        "\n\nAuthors:\n  Paweł Galewicz\t234053"
                        "\n  Bartosz Jurczewski\t234067\n  Zbigniew Nowacki\t234102"
                        "\n  Karol Podlewski\t234106\n  Piotr Wardęcki\t234128")

        # data sets
        self.parser.add_argument('-d', metavar='N', dest='dataset', type=int,
                                 default=0, help='Select data set:'
                                                 '\n  [1] - Fall detection data from China'
                                                 '\n  [2] - Rain in Australia'
                                                 '\n  [3] - Pima Indians Diabetes Database')

        # classifiers
        self.parser.add_argument('-c', metavar='N', dest='classifier', type=int,
                                 default=0, help='Select algorithm:'
                                                 '\n  [1] - Decision trees algorithm'
                                                 '\n  [2] - Naive Bayes'
                                                 '\n  [3] - Support-vector machine'
                                                 '\n  [4] - k-nearest neighbors'
                                                 '\n  [5] - Artificial neural network')

        # training percent
        self.parser.add_argument('-t', metavar='N', dest='training_percent', type=int,
                                 default=0, help='Set percent of training set')

        # classifiers argument
        self.parser.add_argument('-a', metavar='N', dest='class_args', type=float,
                                 nargs='+', default=[-1, -1, -1],
                                 help='Arguments of chosen classifier')

        # extra options
        self.parser.add_argument('--time', dest='time', action='store_const',
                                 const=True, default=False,
                                 help='Measure time of classification')

        # extra options
        self.parser.add_argument('--accuracy', dest='just_accuracy',
                                 action='store_const', const=True, default=False,
                                 help='Print only accuracy (and time with --time)')

        # parse
        self.args = self.parser.parse_args()

    def get_dataset_path(self):
        while 1 > self.args.dataset or self.args.dataset > 3:
            clear()
            self.args.dataset = int(input('Select data set:\n'
                                          '[1] Fall detection data from China\n'
                                          '[2] Rain in Australia\n'
                                          '[3] Pima Indians Diabetes Database\n\n'
                                          'Choice: '))
        return datasets[self.args.dataset]

    def get_dataset_name(self):
        return dataset_names[self.args.dataset]

    def get_classifier(self):
        while 1 > self.args.classifier or self.args.classifier > 5:
            clear()
            self.args.classifier = int(input('Select method:\n'
                                             '[1] Decision trees algorithm\n'
                                             '[2] Naive Bayes classifier\n'
                                             '[3] Support-vector machine\n'
                                             '[4] k-nearest neighbors algorithm\n'
                                             '[5] Artificial neural network algorithm\n\n'
                                             'Choice: '))
        return self.args.classifier

    def get_training_fraction(self):
        while 0 >= self.args.training_percent or self.args.training_percent >= 100:
            clear()
            self.args.training_percent = int(input('Percent of dataset used for training: '))
        return self.args.training_percent / 100

    def get_classifier_arguments(self):
        return self.args.class_args

    def is_just_accuracy(self):
        return self.args.just_accuracy

    def is_time_measured(self):
        return self.args.time
