from sklearn import svm

from utils import clear
from classifiers.classifier import Classifier


class SVM(Classifier):

    name = "Support-vector machine"

    kernel = {1: "linear", 2: "rbf", 3: "poly", 4: "sigmoid"}
    kernel_names = {1: "Linear", 2: "RBF (Radial Basis Function)",
                    3: "Polynomial", 4: "Sigmoid"}
    gamma = {1: "scale", 2: "auto"}
    gamma_names = {1: "Scale", 2: "Auto"}

    def __init__(self, data, lr, labels, training_fraction, arguments):
        super().__init__(data, lr, labels, training_fraction, arguments)

        while(0.01 > arguments[0] or arguments[0] > 10):
            clear()
            arguments[0] = int(input("Regularization (from 0.01 to 10): "))

        arguments[1] = int(arguments[1])
        while(0 >= arguments[1] or arguments[1] >= 5):
            clear()
            arguments[1] = int(input("Choose kernel:\n"
                                "[1] Linear\n"
                                "[2] RBF (Radial Basis Function)\n"
                                "[3] Polynomial\n"
                                "[4] Sigmoid\n\n"
                                "Choice: "))

        if arguments[1] is 1:
            self.model = svm.SVC(C=arguments[0],
                                 kernel=self.kernel[arguments[1]])
        
        else:
            while(0 >= arguments[2] or arguments[2] >= 4):
                clear()
                arguments[2] = int(input("Choose gamma:\n"
                                    "[1] Scale\n"
                                    "[2] Auto\n"
                                    "[3] Float\n\n"
                                    "Choice: "))

            if arguments[2] is not 3:
                arguments[2] = int(arguments[2])
                self.model = svm.SVC(C=arguments[0],
                                     kernel=self.kernel[arguments[1]],
                                     gamma=self.gamma[arguments[2]])
            else:
                while(0.1 > arguments[3] or arguments[3] >= 10):
                    clear()
                    arguments[0] = int(input("Gamma value (from 0.1 to 10): "))

                self.model = svm.SVC(C=arguments[0],
                                    kernel=self.kernel[arguments[1]],
                                    gamma=arguments[3])


    def print_stats(self, dataset_name, only_accuracy=False):
        if only_accuracy is True:
            super().print_stats(dataset_name, only_accuracy)

        else:
            super().print_basic_stats(dataset_name)
            print(f"Regularization:\t\t{self.arguments[0]}")
            print(f"Kernel:\t\t\t{self.kernel_names[self.arguments[1]]}")

            if self.arguments[1] is not 1:
                if self.arguments[2] is not 3: 
                    print(f"Gamma:\t\t\t{self.gamma_names[self.arguments[2]]}")
                else:
                    print(f"Gamma:\t\t\t{self.arguments[3]}")
            print()

            print(self.get_metrics())
