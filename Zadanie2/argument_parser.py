import argparse

from utils import clear, datasets, datasets_names, simple_datasets_names, datasets_clusters, print_datasets_names


class ArgumentParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog='MUM - Task 2', formatter_class=argparse.RawTextHelpFormatter,
            description="Lodz University of Technology (TUL)"
                        "\nMachine learning methods (Metody uczenia maszynowego)"
                        "\n\nTask 2 - Problem set 2"
                        "\n\nAuthors:\n  Paweł Galewicz\t234053"
                        "\n  Bartosz Jurczewski\t234067\n  Zbigniew Nowacki\t234102"
                        "\n  Karol Podlewski\t234106\n  Piotr Wardęcki\t234128")

        # data sets
        self.parser.add_argument('-d', metavar='N', dest='dataset', type=int,
                                 default=0, help='Select data set:\n'+print_datasets_names('  '))

        # algorithm
        self.parser.add_argument('-a', metavar='N', dest='algorithm', type=int,
                                 default=0, help='Select algorithm:'
                                                 '\n  [1] - Expectation–Maximization'
                                                 '\n  [2] - k-means'
                                                 '\n  [3] - Agglomerative hierarchical clustering'
                                                 '\n  [4] - Density-Based Spatial Clustering of Applications with Noise'
                                                 '\n  [5] - Optics')

        # clusters
        self.parser.add_argument('-c', metavar='N', dest='clusters', type=int,
                                 default=5, help='Set the number of clusters')

        # algorithm parameters
        self.parser.add_argument('-p', metavar='N', dest='class_args',
                                 type=str, nargs='+', default=[-1, -1, -1],
                                 help='Arguments of chosen algorithm')

        # fixed clusters
        self.parser.add_argument('--fixed', dest='fixed_n_clusters',
                                 action='store_const', const=True, default=False,
                                 help='Fixed number of clusters od dataset '
                                      '(overwrites -c)')

        # ploting options
        self.parser.add_argument('--show', dest='show_plot',
                                 action='store_const', const=True, default=False,
                                 help='Displays plot')

        self.parser.add_argument('-x', metavar='N', dest='x', type=int, default=None,
                                 help='Set column attached to the X axis '
                                      '(by default it is Index')

        self.parser.add_argument('-y', metavar='N', dest='y', type=int, default=None,
                                 help='Set column attached to the Y axis ' 
                                      '(by default it is last column of dataset')

        # another
        self.parser.add_argument('--elbow', dest='elbow', action='store_const',
                                 const=True, default=False,
                                 help='Runs Elbow Method on chosen dataset')

        # parse
        self.args = self.parser.parse_args()

    def get_dataset_path(self):
        while 1 > self.args.dataset or self.args.dataset > 4:
            clear()
            self.args.dataset = int(input('Select data set:\n' +
                                          print_datasets_names() + 
                                          '\nChoice: '))
        return datasets[self.args.dataset]

    def get_dataset_name(self):
        return datasets_names[self.args.dataset]

    def get_simple_dataset_name(self):
        return simple_datasets_names[self.args.dataset]

    def get_algorithm(self):
        while 1 > self.args.algorithm or self.args.algorithm > 5:
            clear()
            self.args.algorithm = int(input('Select algorithm:\n'
                                            '[1] Expectation–Maximization\n'
                                            '[2] k-means\n'
                                            '[3] Agglomerative hierarchical clustering\n'
                                            '[4] Density-Based Spatial Clustering of Applications with Noise\n'
                                            '[5] Optics\n\n'
                                            'Choice: '))
        return self.args.algorithm

    # def get_training_fraction(self):
    #     while 0 >= self.args.training_percent or self.args.training_percent >= 100:
    #         clear()
    #         self.args.training_percent = int(input('Percent of dataset used for training: '))
    #     return self.args.training_percent / 100

    def get_number_of_clusters(self):
        return self.args.clusters

    def get_classifier_arguments(self):
        return self.args.class_args

    def is_n_clusters_fixed(self):
        return self.args.fixed_n_clusters

    def get_fixed_n_clusters(self):
        return datasets_clusters[self.args.dataset]

    def is_plot_shown(self):
        return self.args.show_plot

    def get_plot_x_axis(self):
        return self.args.x

    def get_plot_y_axis(self):
        return self.args.y

    def is_elbow_method_run(self):
        return self.args.elbow
