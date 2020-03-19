import warnings
from sklearn import metrics


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

    def __init__(self, data, lr, labels, training_fraction):
        self.tt_ratio = training_fraction
        self.data = data
        self.lr = lr
        self.labels = labels

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
