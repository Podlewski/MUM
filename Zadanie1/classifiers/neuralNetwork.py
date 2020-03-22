from sklearn import neural_network as nn

from classifiers.classifier import Classifier


class NeuralNetwork(Classifier):

    name = "Artificial neural network algorithm"

    def __init__(self, data, lr, labels, training_fraction, arguments):
        super().__init__(data, lr, labels, training_fraction, arguments)
        self.model = nn.MLPClassifier(max_iter=5000)
