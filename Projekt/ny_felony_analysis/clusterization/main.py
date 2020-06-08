import numpy
import pandas
import seaborn
from matplotlib import pyplot

from ny_felony_analysis.clusterization._utils import plot_clustermap, plot_regression
from ny_felony_analysis.clusterization.clusterers.agglomerative import Agglomerative
from ny_felony_analysis.clusterization.clusterers.density_based import DensityBased
from ny_felony_analysis.clusterization.clusterers.expectation_maximization import ExpectationMaximization
from ny_felony_analysis.clusterization.clusterers.k_means import Kmeans
from ny_felony_analysis.clusterization.clusterers.optic import Optic
from utils.common import factorize, pca, normalize, correlate_sort, get_linear_regression_values, LABEL_UNIQUES, \
    drop_infrequent

seaborn.set(color_codes=True)

data = pandas.read_csv('../NYPD_Felony_Data.csv')
data = data.drop(
    columns=['CMPLNT_TO_DT', 'CMPLNT_TO_TM'],
    axis=1
)
data = data.dropna()
data = drop_infrequent(data)
data = data.apply(factorize)
data_norm = normalize(data)
data_pca = pca(data_norm, n_components=3)

plot_clustermap(
    data_norm.drop(columns=['Longitude', 'Latitude'])
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

print(correlate_sort(data, 0.2).to_string())

plot_regression(
    data.sample(n=13_000, random_state=666),
    x='SUSP_RACE', y='SUSP_AGE_GROUP',
    hue='SUSP_SEX',
    x_estimator=numpy.mean,
    # lowess=True,
    bottom=0.4, left=0.15
)
pyplot.savefig(
    '../regression',
    dpi=300,
    bbox_inches='tight'
)
pyplot.close()

clusterer = Kmeans(
    data_pca,
    3,
    [-1, -1, -1]
)

data_labels = clusterer.fit_predict()

pyplot.scatter(
    x=data_pca.iloc[:, 0],
    y=data_pca.iloc[:, 1],
    c=data_labels,
    cmap='viridis'
)
pyplot.savefig(
    '../clusterization',
    dpi=300,
    bbox_inches='tight'
)
