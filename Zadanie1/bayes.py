from sklearn import naive_bayes as nb
import pandas

from classifier import Classifier


class Bayes(Classifier):
    model = nb.GaussianNB()

    def __init__(self, data, training_fraction=0.7):
        super().__init__(data, training_fraction)

    def train(self):
        self.model.fit(self.training_data,
                       self.training_target_values)

    def test(self):
        prediction = self.model.predict(self.test_data)
        print(prediction - self.test_target_values)
