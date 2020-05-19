from sklearn import neural_network as nn

from classifiers.classifier import Classifier


class NeuralNetwork(Classifier):
    name = "Artificial neural network algorithm"
    short_name = "NN"

    def __init__(self, data, labels, unique, training_fraction):
        super().__init__(data, labels, unique, training_fraction)
        self.model = nn.MLPClassifier(alpha=0.0001,
                                      max_iter=5000)
