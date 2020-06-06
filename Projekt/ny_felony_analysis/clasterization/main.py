import pandas
import seaborn
from matplotlib import pyplot

from ny_felony_analysis.clasterization.clusterers.agglomerative import Agglomerative
from ny_felony_analysis.clasterization.clusterers.density_based import DensityBased
from ny_felony_analysis.clasterization.clusterers.expectation_maximization import ExpectationMaximization
from ny_felony_analysis.clasterization.clusterers.k_means import Kmeans
from ny_felony_analysis.clasterization.clusterers.optic import Optic
from utils.common import factorize, pca, normalize

data = pandas.read_csv('../NYPD_Felony_Data.csv')
data = data.dropna(thresh=data.shape[1] - 0)
# data = data.impute_hotdeck()
data = data.apply(factorize)
data = normalize(data)
# data = pca(data.astype('float32'))

# checkout fastcluster
seaborn.clustermap(
    data[:1_000]
)
pyplot.show()
exit(1)

clusterer = Kmeans(
    data,
    5,
    [-1, -1, -1]
)

data_labels = clusterer.fit_predict()

pyplot.scatter(
    x=data.iloc[:, 0],
    y=data.iloc[:, 1],
    c=data_labels,
    cmap='viridis'
)
pyplot.show()
