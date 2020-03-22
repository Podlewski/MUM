from sklearn import neural_network as nn

from classifiers.classifier import Classifier


class NeuralNetwork(Classifier):

    def __init__(self, data, lr, labels, training_fraction, args):
        super().__init__(data, lr, labels, training_fraction)
        self.model = nn.MLPClassifier(max_iter=5000)
