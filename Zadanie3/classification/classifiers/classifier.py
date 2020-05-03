import warnings
from sklearn import metrics


class Classifier:
    model = None
    tt_ratio = 0
    data = None
    training_data = None
    training_target_values = None
    test_data = None
    test_target_values = None
    prediction = None
    labels = None
    name = None
    short_name = None

    def __init__(self, data, labels, training_fraction):
        self.tt_ratio = training_fraction
        self.data = data
        self.label = 'R'
        self.labels = labels

    def __label__(self):
        return {'L': [self.data.columns[0]],
                'R': [self.data.columns[-1]]}[self.label]

    def shuffle_and_assign_data(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        data_size = len(self.data.index)
        self.training_data \
            = self.data.loc[
              :int(self.tt_ratio * data_size),
              self.data.columns[~self.data.columns.isin(self.__label__())]].values
        self.training_target_values \
            = self.data.loc[
              :int(self.tt_ratio * data_size),
              self.data.columns[self.data.columns.isin(self.__label__())]].values.ravel()
        self.test_data \
            = self.data.loc[
              int(self.tt_ratio * data_size) + 1:,
              self.data.columns[~self.data.columns.isin(self.__label__())]].values
        self.test_target_values \
            = self.data.loc[
              int(self.tt_ratio * data_size) + 1:,
              self.data.columns[self.data.columns.isin(self.__label__())]].values.ravel()

    def train(self):
        self.shuffle_and_assign_data()
        self.model.fit(self.training_data,
                       self.training_target_values)

    def test(self):
        self.prediction = self.model.predict(self.test_data)

    def get_accuracy(self, digits):
        return round(metrics.accuracy_score(
            self.test_target_values,
            self.prediction), digits)

    def get_confusion_matrix(self):
        return metrics.confusion_matrix(
            self.test_target_values,
            self.prediction)    

    def get_metrics(self, digits):
        warnings.filterwarnings('ignore')
        return metrics.classification_report(
            self.test_target_values,
            self.prediction,
            target_names=list(map(str, self.labels)),
            digits=digits
        )

    def get_precision(self, digits):
        return round(metrics.precision_score(
            self.test_target_values,
            self.prediction), digits)

    def get_recall(self, digits):
        return round(metrics.recall_score(
            self.test_target_values,
            self.prediction), digits)

    def get_roc_curve_plot(self):
        roc_cure_plot = metrics.plot_roc_curve(
            self.model,
            self.test_data,
            self.test_target_values)
        
        fpr = roc_cure_plot.fpr
        tpr = roc_cure_plot.tpr
        roc_auc = roc_cure_plot.roc_auc

        return fpr, tpr, roc_auc

    def print_confusion_matrix(self):
        tn, fp, fn, tp = self.get_confusion_matrix().ravel()

        print('\nConfusion matrix:')
        print('  ' + str(tp) + '\t' + str(fn))
        print('  ' + str(fp) + '\t' + str(tn))

    def print_metrics(self, digits):
        tn, fp, fn, tp = self.get_confusion_matrix().ravel()

        metrics_names = ['Precision', 'Accuracy', 'Recall', 'Specifity']
        metrics_scores = [self.get_precision(digits),
                          self.get_accuracy(digits),
                          self.get_recall(digits),
                          round((tn/(tn+fp)), digits)]

        metrics_first_line = '  '
        metrics_second_line = '  '

        for name, score in zip(metrics_names, metrics_scores):
            length = max(len(name), len(str(score))) + 3
            metrics_first_line += name.ljust(length)
            metrics_second_line += str(score).ljust(length)

        print('\nMetrics:')
        print(metrics_first_line)
        print(metrics_second_line)  

    def print_stats(self, dataset_name, digits=3):
        print(f'Dataset:           {dataset_name}')
        print(f'Classificator:     {self.name}')
        print(f'Training percent:  {self.tt_ratio * 100}%')

        print('\nLabels: ' + str(self.labels))

        self.print_confusion_matrix()
        self.print_metrics(digits)
