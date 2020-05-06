from matplotlib import pyplot
from argument_parser import ArgumentParser
from clusterers.agglomerative import Agglomerative
from clusterers.density_based import DensityBased
from clusterers.expectation_maximization import ExpectationMaximization
from clusterers.k_means import Kmeans
from clusterers.Optic_Clustering import optics
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from utils import clear, load_dataset, factorize
from gap import optimalK
from sklearn.cluster import KMeans


argument_parser = ArgumentParser()

setup = {
    "dataset": argument_parser.get_dataset_path(),
    "dataset_name": argument_parser.get_dataset_name(),
    "simple_name": argument_parser.get_simple_dataset_name(),
    "algorithm": argument_parser.get_algorithm(),
    "clusters": argument_parser.get_number_of_clusters(),
    "class_args": argument_parser.get_classifier_arguments()
}

if argument_parser.is_n_clusters_fixed() is True:
    setup['clusters'] = argument_parser.get_fixed_n_clusters()

data = load_dataset(setup['dataset'])
data = data.apply(factorize)

if argument_parser.is_elbow_method_run() is False:
    algorithm = None
    if setup['algorithm'] == 1:
        algorithm = ExpectationMaximization(data, setup['clusters'], setup['class_args'])
    elif setup['algorithm'] == 2:
        algorithm = Kmeans(data, setup['clusters'], setup['class_args'])
    elif setup['algorithm'] == 3:
        algorithm = Agglomerative(data, setup['clusters'], setup['class_args'])
    elif setup['algorithm'] == 4:
        algorithm = DensityBased(data, setup['class_args'])
    elif setup['algorithm'] == 5:
        algorithm = optics(data, setup['class_args'])

    data_labels = algorithm.fit_predict()

    print(f'Silhouette:\t\t%0.4f' % silhouette_score(data, data_labels))
    print(f'Calinski-Harabasz:\t%0.1f' % calinski_harabasz_score(data, data_labels))
    print(f'Davies-Bouldin:\t\t%0.4f' % davies_bouldin_score(data, data_labels))

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

    filename = (algorithm.__class__.__name__ + "_" + setup['simple_name'] +
                "_x" + xlabel + "_y" + ylabel).replace(".", "")
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
    
    filename = ("elbow_" + setup['simple_name']).replace(".", "")
    pyplot.savefig(filename, dpi=200, bbox_inches='tight')
    dawidekArray = []
    hrabaszczArray = []
    silchoutArray = []
    clusters = range(2, 11)
    for n_cluster in clusters:
        x = data
        kmeans = KMeans(n_clusters=n_cluster).fit(x)
        label = kmeans.labels_
        dawidek = davies_bouldin_score(x, label)
        hrabaszcz = calinski_harabasz_score(x, label)
        silchout = silhouette_score(x, label)
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, silchout))
        silchoutArray.append(silchout)
        print("For n_clusters={}, hrabaszcz {}".format(n_cluster, hrabaszcz))
        hrabaszczArray.append(hrabaszcz)
        print("For n_clusters={}, dawidek {}".format(n_cluster, dawidek))
        dawidekArray.append(dawidek)
    pyplot.clf()
    pyplot.close()
    pyplot.plot(clusters, silchoutArray, 'bx-')
    pyplot.xlabel('clusters')
    pyplot.ylabel('silhouette')
    filename = ("silhouette_" + setup['simple_name']).replace(".", "")
    pyplot.savefig(filename, dpi=200, bbox_inches='tight')

    pyplot.clf()
    pyplot.close()
    pyplot.plot(clusters, hrabaszczArray, 'bx-')
    pyplot.xlabel('clusters')
    pyplot.ylabel('calinski_harabasz')
    filename = ("calinski_harabasz_" + setup['simple_name']).replace(".", "")
    pyplot.savefig(filename, dpi=200, bbox_inches='tight')

    pyplot.clf()
    pyplot.close()
    pyplot.plot(clusters, dawidekArray, 'bx-')
    pyplot.xlabel('clusters')
    pyplot.ylabel('davies_bouldin')
    filename = ("davies_bouldin_" + setup['simple_name']).replace(".", "")
    pyplot.savefig(filename, dpi=200, bbox_inches='tight')

    if argument_parser.is_GAP_method_run() is True:
        x = data
        k, gapdf = optimalK(x, nrefs=5, maxClusters=15)
        print(f'Optymalny cluster:\t\t%0.4f' %k)
        pyplot.clf()
        pyplot.close()
        pyplot.plot(gapdf.clusterCount, gapdf.gap,'bx-', linewidth=3,)
        pyplot.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
        pyplot.grid(True)
        pyplot.xlabel('Cluster Count')
        pyplot.ylabel('Gap Value')
        pyplot.title('Gap Values by Cluster Count')
        pyplot.show()
        filename = ("GAP_" + setup['simple_name']).replace(".", "")
        pyplot.savefig(filename, dpi=200, bbox_inches='tight')


    if argument_parser.is_plot_shown() is True:
        pyplot.show()
