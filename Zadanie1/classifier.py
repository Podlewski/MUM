class Classifier:
    data = None

    def __init__(self, data):
        self.data = data

    def count_accuracy(self, number_of_correct_predictions):
        return number_of_correct_predictions / self.data.size
