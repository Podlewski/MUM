class Classifier:
    data = None

    def __init__(self, data):
        self.data = data.sample(frac=1).reset_index(drop=True)

    def count_accuracy(self, number_of_correct_predictions):
        return number_of_correct_predictions / self.data.size
