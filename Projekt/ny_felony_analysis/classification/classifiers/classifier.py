import warnings
from numpy import unique
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
    labels_column = None
    labels = None
    name = None
    short_name = None

    def __init__(self, data, labels, unique, training_fraction):
        self.tt_ratio = training_fraction
        self.data = data
        self.labels = labels
        self.unique_labels = unique

    def assign_data(self):
        partition_index = int(len(self.data.index) * self.tt_ratio)
        self.training_data = self.data.iloc[:partition_index].values
        self.training_target_values = self.labels.iloc[:partition_index].values.ravel()
        self.test_data = self.data.iloc[partition_index:].values
        self.test_target_values = self.labels.iloc[partition_index:].values.ravel()

    def train(self):
        self.assign_data()
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
        return metrics.classification_report(
            self.test_target_values,
            self.prediction,
            target_names=list(map(str, self.unique_labels)),
            digits=digits
        )

    def get_precision(self, digits):
        return round(metrics.precision_score(
            self.test_target_values,
            self.prediction, average='macro'), digits)

    def get_recall(self, digits):
        return round(metrics.recall_score(
            self.test_target_values,
            self.prediction, average='macro'), digits)

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
        # tn, fp, fn, tp = self.get_confusion_matrix().ravel()

        print('\nConfusion matrix:')
        print(self.get_confusion_matrix())
        # print('  ' + str(tp) + '\t' + str(fn))
        # print('  ' + str(fp) + '\t' + str(tn))

    def print_metrics(self, digits):
        warnings.filterwarnings('ignore')
        # tn, fp, fn, tp = self.get_confusion_matrix().ravel()
        # metrics_names = ['Precision', 'Accuracy', 'Recall', 'Specifity']
        metrics_names = ['Precision', 'Accuracy', 'Recall']
        metrics_scores = [self.get_precision(digits),
                          self.get_accuracy(digits),
                          self.get_recall(digits)]

        metrics_first_line = '  '
        metrics_second_line = '  '

        for name, score in zip(metrics_names, metrics_scores):
            length = max(len(name), len(str(score))) + 3
            metrics_first_line += name.ljust(length)
            metrics_second_line += str(score).ljust(length)

        print('\nMetrics:')
        print(metrics_first_line)
        print(metrics_second_line)  

    def print_stats(self, time=None, digits=3):
        print('\n~~~~~~~~~~ ' + self.name) 
        # self.print_confusion_matrix()
        self.print_metrics(digits)
        if time is not None:
            print(f'\nTime:  {round(time, 2)} s')
