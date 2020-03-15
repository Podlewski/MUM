import warnings
from sklearn import metrics

result_column_labels = ["ACTIVITY", "RainTomorrow", "generation"]


class Classifier:
    model = None
    tt_ratio = 0
    data = None
    training_data = None
    training_target_values = None
    test_data = None
    test_target_values = None
    prediction = None

    def __init__(self, data, training_fraction):
        self.tt_ratio = training_fraction
        self.data = data

    def shuffle_and_assign_data(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        data_size = len(self.data.index)
        self.training_data \
            = self.data.loc[
              :int(self.tt_ratio * data_size),
              self.data.columns[~self.data.columns.isin(result_column_labels)]].values
        self.training_target_values \
            = self.data.loc[
              :int(self.tt_ratio * data_size),
              self.data.columns[self.data.columns.isin(result_column_labels)]].values.ravel()
        self.test_data \
            = self.data.loc[
              int(self.tt_ratio * data_size) + 1:,
              self.data.columns[~self.data.columns.isin(result_column_labels)]].values
        self.test_target_values \
            = self.data.loc[
              int(self.tt_ratio * data_size) + 1:,
              self.data.columns[self.data.columns.isin(result_column_labels)]].values.ravel()

    def train(self):
        self.shuffle_and_assign_data()
        self.model.fit(self.training_data,
                       self.training_target_values)

    def test(self):
        self.prediction = self.model.predict(self.test_data)

    def get_metrics(self):
        # print(metrics.confusion_matrix(self.test_target_values, self.prediction))
        warnings.filterwarnings('ignore')
        return metrics.classification_report(self.test_target_values, self.prediction, digits=3)
