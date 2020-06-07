import numpy
import pandas
from matplotlib import pyplot

from ny_felony_analysis.clusterization._utils import plot_clustermap, plot_regression
from ny_felony_analysis.clusterization.clusterers.agglomerative import Agglomerative
from ny_felony_analysis.clusterization.clusterers.density_based import DensityBased
from ny_felony_analysis.clusterization.clusterers.expectation_maximization import ExpectationMaximization
from ny_felony_analysis.clusterization.clusterers.k_means import Kmeans
from ny_felony_analysis.clusterization.clusterers.optic import Optic
from utils.common import factorize, pca, normalize, correlate_sort, get_linear_regression_values, LABEL_UNIQUES, \
    drop_infrequent

data = pandas.read_csv('../NYPD_Felony_Data.csv')
data = data.drop(
    columns=['CMPLNT_TO_DT', 'CMPLNT_TO_TM'],
    axis=1
)
data = data.dropna()
data = drop_infrequent(data)
data = data.apply(factorize)
# data = normalize(data)
# data = pca(data, n_components=5)

plot_clustermap(
    data.drop(columns=['Longitude', 'Latitude'])
        .sample(n=13_000, random_state=666),
    method='ward',
    # standard_scale=1,
    cbar_pos=None
)
pyplot.savefig(
    '../dendrogram',
    dpi=300,
    bbox_inches='tight'
)
pyplot.clf()
pyplot.cla()
pyplot.close()

print(correlate_sort(data).to_string())

plot_regression(
    data.sample(n=13_000, random_state=666),
    x='SUSP_RACE', y='SUSP_AGE_GROUP',
    hue='SUSP_SEX',
    x_estimator=numpy.mean,
    # lowess=True,
    xticklabels=LABEL_UNIQUES['SUSP_RACE'],
    yticklabels=LABEL_UNIQUES['SUSP_AGE_GROUP'],
    hticklabels=LABEL_UNIQUES['SUSP_SEX'],
    bottom=0.4, left=0.15
)
pyplot.savefig(
    '../regression',
    dpi=300,
    bbox_inches='tight'
)
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
