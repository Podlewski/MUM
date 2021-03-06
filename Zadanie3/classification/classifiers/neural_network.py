from sklearn import neural_network as nn

from classifiers.classifier import Classifier


class NeuralNetwork(Classifier):
    name = "Artificial neural network algorithm"
    short_name = "NN"

    def __init__(self, data, labels, training_fraction, arguments):
        super().__init__(data, labels, training_fraction)
        # number_of_hidden_layer = int(input("Choose number of hidden_layers:"))
        # self.model = nn.MLPClassifier(alpha=0.0001,
        #                               hidden_layer_sizes=(number_of_hidden_layer, 10),
        #                               max_iter=5000)
        self.model = nn.MLPClassifier(alpha=0.0001,
                                      max_iter=5000)
