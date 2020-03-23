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
        self.parser.add_argument('-d1', dest='dataset', action='store_const', const=1,
                                 help='Use data set "Fall detection data from China"')
        self.parser.add_argument('-d2', dest='dataset', action='store_const', const=2,
                                 help='Use data set "Rain in Australia"')
        self.parser.add_argument('-d3', dest='dataset', action='store_const', const=3,
                                 help='Use data set "Sucide rates overview"')                      

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

        # classifiers argument
        self.parser.add_argument('-a', metavar='N', dest='class_args', type=float,
                                 nargs='+', default=[-1,-1,-1],
                                 help = 'Argument of chosen classifier')

        self.parser.add_argument('--time', dest='time', action='store_const',
                                 const=True, default=False,
                                 help='Measure time of classification')

        # parse
        self.args = self.parser.parse_args()

    def get_dataset_path(self):
        if self.args.dataset is None:
            self.args.dataset = 0
            while 1 > self.args.dataset or self.args.dataset > 3:
                clear()
                self.args.dataset = int(input('Choose data set:\n'
                                              '[1] Fall detection data from China\n'
                                              '[2] Rain in Australia\n'
                                              '[3] Prima Indians Diabetes Database\n\n'
                                              'Choice: '))
        return datasets[self.args.dataset]

    def get_dataset_name(self):
        return dataset_names[self.args.dataset]

    def get_classifier(self):
        if self.args.classifier is None:
            self.args.classifier = 0
            while 1 > self.args.classifier or self.args.classifier > 5:
                clear()
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
            clear()
            self.args.training_percent = int(input('Percent of dataset used for training: '))
        return self.args.training_percent / 100

    def get_classifier_arguments(self):
        return self.args.class_args

    def is_time_measured(self):
        return self.args.time
