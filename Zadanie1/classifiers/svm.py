from sklearn import svm

from classifiers.classifier import Classifier

class SVM(Classifier):

    name = "Support-vector machine"

    kernel = {1: "rbf", 2: "linear", 3: "poly", 4: "sigmoid"}
    kernel_names = {1: "RBF (Radial Basis Function)", 2: "Linear",
                    3: "Polynomial", 4: "Sigmoid"}

    def __init__(self, data, lr, labels, training_fraction, arguments):
        super().__init__(data, lr, labels, training_fraction, arguments)

        while(0 >= arguments[0] or arguments[0] >= 5):
            # clear()
            arguments[0] = int(input("Choose kernel:\n"
                                "[1] RBF (Radial Basis Function)\n"
                                "[2] Linear\n"
                                "[3] Polynomial\n"
                                "[3] Sigmoid\n\n"
                                "Choice: "))
        
        self.model = svm.SVC(kernel=self.kernel[arguments[0]])

    def print_stats(self, dataset_name, basic=True):
        if basic is True:
            super().print_basic_stats(dataset_name)
        print(f"Kernel:\t\t\t{self.kernel_names[self.arguments[0]]}")

        print()

        print(self.get_metrics())
