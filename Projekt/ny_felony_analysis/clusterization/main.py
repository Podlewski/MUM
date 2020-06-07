import pandas
import seaborn
from matplotlib import pyplot

from ny_felony_analysis.clusterization.clusterers.agglomerative import Agglomerative
from ny_felony_analysis.clusterization.clusterers.density_based import DensityBased
from ny_felony_analysis.clusterization.clusterers.expectation_maximization import ExpectationMaximization
from ny_felony_analysis.clusterization.clusterers.k_means import Kmeans
from ny_felony_analysis.clusterization.clusterers.optic import Optic
from utils.common import factorize, pca, normalize, correlate_sort

data = pandas.read_csv('../NYPD_Felony_Data.csv')
data = data.drop(
    columns=['CMPLNT_TO_DT', 'CMPLNT_TO_TM'],
    axis=1
)
data = data.dropna()
data = data.apply(factorize)
data = normalize(data)
# data = pca(data, n_components=5)

seaborn.clustermap(
    data.drop(columns=['Longitude', 'Latitude'])
        .sample(n=13_000, random_state=666),
    # metric='correlation',  # correlation,>= euclidean (default)
    method='ward',  # ward > centroid > average (default)
    # standard_scale=1,
    # z_score=0,
    cbar_pos=None,
)
pyplot.savefig(
    '../dendrogram',
    dpi=300,
    bbox_inches='tight'
)

correlation = correlate_sort(data)
print(correlation.to_string())
exit(1)


clusterer = Kmeans(
    data,
    3,
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