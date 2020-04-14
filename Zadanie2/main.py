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
    "algorithm": argument_parser.get_algorithm(),
    "clusters": argument_parser.get_number_of_clusters(),
    "class_args": argument_parser.get_classifier_arguments()
}
clear()

if argument_parser.if_n_clusters_fixed() is True:
    setup['clusters'] = argument_parser.get_fixed_n_clusters()

data = load_dataset(setup['dataset'])
data = data.apply(factorize)

algorithm = {
    1: ExpectationMaximization(data, setup['clusters'], setup['class_args']),
    2: Kmeans(data, setup['clusters'], setup['class_args']),
    3: Agglomerative(data, setup['clusters']),
    4: DensityBased(data, setup['class_args'])
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

x = data.index
y = data.iloc[:, len(data.columns)-1]

xlabel = "Index"
ylabel = data.columns[len(data.columns)-1]

if argument_parser.get_plot_x_axis() is not None:
    x = data.iloc[:, argument_parser.get_plot_x_axis()]
    xlabel = data.columns[argument_parser.get_plot_x_axis()]

if argument_parser.get_plot_y_axis() is not None:
    y = data.iloc[:, argument_parser.get_plot_y_axis()]
    ylabel = data.columns[argument_parser.get_plot_y_axis()]

pyplot.scatter(
    x=x,
    y=y,
    c=algorithm.get_labels(),
    cmap='rainbow'
)

pyplot.xlabel(xlabel)
pyplot.ylabel(ylabel)
pyplot.grid(True, alpha=0.3)
pyplot.savefig("plot", dpi=200, bbox_inches='tight')

if argument_parser.is_plot_shown() is True:
    pyplot.show()
