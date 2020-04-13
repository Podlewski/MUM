from matplotlib import pyplot

from argument_parser import ArgumentParser
from clusterers.agglomerative import Agglomerative
from clusterers.density_based import DensityBased
from clusterers.expectation_maximization import ExpectationMaximization
from clusterers.k_means import Kmeans
from utils import clear, load_dataset, factorize

argument_parser = ArgumentParser()

setup = {
    "dataset": argument_parser.get_dataset_path(),
    "dataset_name": argument_parser.get_dataset_name(),
    "algorithm": argument_parser.get_algorithm()
}
clear()

data = load_dataset(setup['dataset'])
data = data.apply(factorize)

class_args = argument_parser.get_classifier_arguments()

algorithm = {
    1: ExpectationMaximization(data, class_args),
    2: Kmeans(data),
    3: Agglomerative(data,
                     n_clusters=8),
    4: DensityBased(data, class_args)
}[setup['algorithm']]

algorithm.fit_predict()

# figure = pyplot.figure()
# ax = figure.add_subplot(211, projection='3d')
# ax.scatter(
#     data.index,
#     data.loc[:, correlation[0]],
#     data.loc[:, correlation[1]],
#     c=algorithm.model.labels_,
#     cmap='rainbow'
# )

pyplot.scatter(
    x=data.index,
    y=data.iloc[:, 4],
    c=algorithm.get_labels(),
    cmap='rainbow'
)

pyplot.xlabel("index")
pyplot.ylabel(data.columns[4])
pyplot.grid(True, alpha=0.3)
pyplot.savefig("plot", dpi=200, bbox_inches='tight')
pyplot.show()
