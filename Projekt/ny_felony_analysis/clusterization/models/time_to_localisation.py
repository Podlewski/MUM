import pandas
import seaborn
from matplotlib import pyplot

from ny_felony_analysis.clusterization.clusterers.expectation_maximization import ExpectationMaximization
from utils.common import factorize, pca, normalize, drop_infrequent

seaborn.set(color_codes=True)

data = pandas.read_csv('../../NYPD_Felony_Data.csv')
data = data.drop(
    columns=['CMPLNT_TO_DT', 'CMPLNT_TO_TM'],
    axis=1
)
data = data.dropna()
data = drop_infrequent(data)
data_factor = data.apply(factorize)
data_norm = normalize(data_factor)

location_info = pca(data_norm[['BORO_NM', 'ADDR_PCT_CD']], n_components=1)

data_chosen_pca = pandas.DataFrame()
data_chosen_pca['TIME'] = data_norm['CMPLNT_FR_TM']
data_chosen_pca['LOCATION'] = location_info[0]

clusterer = ExpectationMaximization(data_chosen_pca, 3, [-1, -1])
data_labels = clusterer.fit_predict()

data['cluster'] = data_labels

clustered_data = data.groupby(['cluster'])
for name, cluster in clustered_data:
    print("CLUSTER_" + str(name))
    print("* most common crimes:")
    print(cluster['PD_CD'].value_counts(normalize=True).head(3).to_string())


pyplot.scatter(
    x=data_chosen_pca['LOCATION'],
    y=data_chosen_pca['TIME'],
    c=data_labels,
    cmap='viridis'
)
pyplot.savefig(
    'time_to_location_clusters',
    dpi=300,
    bbox_inches='tight'
)