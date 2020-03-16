from sklearn import neural_network as nn

from classifier import Classifier


class NeuralNetwork(Classifier):

    def __init__(self, data, training_fraction=0.3):
        super().__init__(data, training_fraction)
        self.model = nn.MLPClassifier(max_iter=5000)
