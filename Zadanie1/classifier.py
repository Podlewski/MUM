result_column_labels = ["ACTIVITY", "RainTomorrow", "generation"]


class Classifier:
    tt_ratio = 0
    data = None
    data_size = 0
    training_data = None
    training_target_values = None
    test_data = None
    test_target_values = None

    def __init__(self, data, training_fraction):
        self.tt_ratio = training_fraction
        self.data = data.sample(frac=1).reset_index(drop=True)
        self.data_size = len(self.data.index)
        self.training_data = self.data.loc[
                             :int(self.tt_ratio * self.data_size),
                             self.data.columns[~self.data.columns.isin(result_column_labels)]].values
        self.training_target_values = self.data.loc[
                                      :int(self.tt_ratio * self.data_size),
                                      self.data.columns[self.data.columns.isin(result_column_labels)]].values.ravel()
        self.test_data = self.data.loc[
                         int(self.tt_ratio * self.data_size) + 1:,
                         self.data.columns[~self.data.columns.isin(result_column_labels)]].values
        self.test_target_values = self.data.loc[
                                  int(self.tt_ratio * self.data_size) + 1:,
                                  self.data.columns[self.data.columns.isin(result_column_labels)]].values.ravel()

    def count_accuracy(self, number_of_correct_predictions):
        return number_of_correct_predictions / len(self.data.index)
