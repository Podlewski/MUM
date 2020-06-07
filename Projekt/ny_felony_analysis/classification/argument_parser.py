import argparse


class ArgumentParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog='MUM - Project - Classification',
            formatter_class=argparse.RawTextHelpFormatter,
            description='Lodz University of Technology (TUL)'
                        '\nMachine learning methods (Metody uczenia maszynowego)'
                        '\n\nProject - Classification'
                        '\n\nAuthors:'
                        '\n  Paweł Galewicz\t234053'
                        '\n  Bartosz Jurczewski\t234067'
                        '\n  Zbigniew Nowacki\t234102'
                        '\n  Karol Podlewski\t234106'
                        '\n  Piotr Wardęcki\t234128')

        self.parser.add_argument('-l', metavar='LABEL', dest='label',
                                 type=str, default=None,
                                 help='Specify feature label by name '
                                      '(overwrites -ln)')

        self.parser.add_argument('-ln', metavar='LABEL_NUMBER', dest='label_number',
                                 type=int, default=None,
                                 help='Specify feature label by number '
                                      '(overwrited by -l)')

        self.parser.add_argument('-d', metavar='D', dest='drops',
                                 type=str, default=None, nargs='+',
                                 help='Specify features to drop by name '
                                      '(merges features from -dn)')

        self.parser.add_argument('-dn', metavar='D', dest='drops_numbers',
                                 default=None, type=int, nargs='+',
                                 help='Specify features to drop by number '
                                      '(merges features from -d)')

        self.parser.add_argument('-t', metavar='PERCENT', dest='training_percent',
                                 type=int, default=75, choices=range(1, 100),
                                 help='Set percent of training set')

        self.parser.add_argument('-i', dest='reduction', 
                                 action='store_const', const="ica", default=None,
                                 help='Reduce features with ICA algorithm')

        self.parser.add_argument('-p', dest='reduction', 
                                 action='store_const', const="pca",
                                 help='Reduce features with PCA algorithm') 

        self.parser.add_argument('-s', '--stack', dest='stacking_classifier', 
                                 action='store_const', const=True, default=False,
                                 help='Run program with stacking classifier')

        self.parser.add_argument('-bg', '--bagging', dest='bagging_classifier',
                                 action='store_const', const=True, default=False,
                                 help='Run program with decision tree bagging classifier')

        self.parser.add_argument('-bt', '--boosting', dest='boosting_classifier',
                                 action='store_const', const=True, default=False,
                                 help='Run program with gradient boosting classifier')

        self.parser.add_argument('--time', dest='time', 
                                 action='store_const', const=True, default=False,
                                 help='Measure time of classification')

        self.parser.add_argument('-rp', '--roc', dest='roc_curve', 
                                 action='store_const', const=True, default=False,
                                 help='Create ROC curve plot')

        self.parser.add_argument('-lp', '--learn', dest='learning_curve', 
                                 action='store_const', const=True, default=False,
                                 help='Create learning curve plot')

        self.args = self.parser.parse_args()


    def get_arguments(self):
        self.args.training_fraction = self.args.training_percent / 100
        return self.args
