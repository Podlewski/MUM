from matplotlib import pyplot

from argument_parser import ArgumentParser
from clusterers.agglomerative import Agglomerative
from clusterers.density_based import DensityBased
from clusterers.expectation_maximization import ExpectationMaximization
from clusterers.k_means import Kmeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
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

if argument_parser.is_n_clusters_fixed() is True:
    setup['clusters'] = argument_parser.get_fixed_n_clusters()

data = load_dataset(setup['dataset'])
data = data.apply(factorize)

if argument_parser.is_elbow_method_run() is False:
    algorithm = {
        1: ExpectationMaximization(data, setup['clusters'], setup['class_args']),
        2: Kmeans(data, setup['clusters'], setup['class_args']),
        3: Agglomerative(data, setup['clusters']),
        4: DensityBased(data, setup['class_args'])
    }[setup['algorithm']]

    labels = algorithm.fit().labels_
    data_labels = algorithm.fit_predict()

    print(f'Silikon:\t %0.4f' % silhouette_score(data, labels))
    print(f'Chrabąszcz:\t %0.4f' % calinski_harabasz_score(data, labels))
    print(f'David Bowie:\t %0.4f' % davies_bouldin_score(data, labels))

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
        c=data_labels,
        cmap='rainbow'
    )

    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.grid(True, alpha=0.3)

    filename = (algorithm.__class__.__name__ + "_x" + xlabel + "_y" + ylabel).replace(".", "")
    print(filename)
    pyplot.savefig(filename, dpi=200, bbox_inches='tight')

    if argument_parser.is_plot_shown() is True:
        pyplot.show()

else:
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = Kmeans(data, k, setup['class_args'])
        distortions.append(kmeans.get_inertia_for_elbow())

    pyplot.clf()
    pyplot.close()
    pyplot.plot(K, distortions, 'bx-')
    pyplot.xlabel('k')
    pyplot.ylabel('Distortion')
    pyplot.show()
