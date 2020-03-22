from sklearn import svm

from classifiers.classifier import Classifier

class SVM(Classifier):

    def kernel(self, kernel_number):
        return {1: "rbf",
                2: "linear",
                3: "rbf",
                4: "sigmoid"}[kernel_number]

    def __init__(self, data, lr, labels, training_fraction, args):
        super().__init__(data, lr, labels, training_fraction)

        while(0 >= args[0] or args[0] >= 5):
            # clear()
            args[0] = int(input("Choose kernel:\n"
                                "[1] RBF\n"
                                "[2] Linear\n"
                                "[3] Polynomial\n"
                                "[3] Sigmoid\n\n"
                                "Choice: "))
        
        self.model = svm.SVC(kernel=self.kernel(args[0]))
