import warnings
from sklearn import metrics

import utils


class Classifier:
    model = None
    tt_ratio = 0
    lr = ''
    data = None
    training_data = None
    training_target_values = None
    test_data = None
    test_target_values = None
    prediction = None
    labels = None
    name = None
    arguments = None

    def __init__(self, data, lr, labels, training_fraction, arguments):
        self.tt_ratio = training_fraction
        self.data = data
        self.lr = lr
        self.labels = labels
        self.arguments = arguments

    def __lr__(self):
        return {'L': [self.data.columns[0]],
                'R': [self.data.columns[-1]]}[self.lr]

    def shuffle_and_assign_data(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        data_size = len(self.data.index)
        self.training_data \
            = self.data.loc[
              :int(self.tt_ratio * data_size),
              self.data.columns[~self.data.columns.isin(self.__lr__())]].values
        self.training_target_values \
            = self.data.loc[
              :int(self.tt_ratio * data_size),
              self.data.columns[self.data.columns.isin(self.__lr__())]].values.ravel()
        self.test_data \
            = self.data.loc[
              int(self.tt_ratio * data_size) + 1:,
              self.data.columns[~self.data.columns.isin(self.__lr__())]].values
        self.test_target_values \
            = self.data.loc[
              int(self.tt_ratio * data_size) + 1:,
              self.data.columns[self.data.columns.isin(self.__lr__())]].values.ravel()

    def train(self):
        self.shuffle_and_assign_data()
        self.model.fit(self.training_data,
                       self.training_target_values)

    def test(self):
        self.prediction = self.model.predict(self.test_data)

    def get_metrics(self):
        # print(metrics.confusion_matrix(self.test_target_values, self.prediction))
        warnings.filterwarnings('ignore')
        return metrics.classification_report(
            self.test_target_values,
            self.prediction,
            target_names=list(map(str, self.labels)),
            digits=3
        )

    def print_basic_stats(self, dataset_name):
        utils.clear()
        if dataset_name is not None:
            print(f"Data set:\t\t{dataset_name}")
        print(f"Classificator:\t\t{self.name}")
        print(f"Training percent:\t{self.tt_ratio * 100}%\n")

    def print_stats(self, dataset_name, basic=True):
        utils.clear()
        if basic is True:
            self.print_basic_stats(dataset_name)
        print(self.get_metrics())
