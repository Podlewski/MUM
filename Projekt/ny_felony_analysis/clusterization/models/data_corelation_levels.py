import numpy
import pandas
import seaborn
from matplotlib import pyplot

from ny_felony_analysis.clusterization._utils import plot_regression
from utils.common import factorize, correlate_sort, drop_infrequent, normalize

seaborn.set(color_codes=True)

data = pandas.read_csv('../../NYPD_Felony_Data.csv')
data = data.drop(
    columns=['CMPLNT_TO_DT', 'CMPLNT_TO_TM'],
    axis=1
)
data = data.dropna()
data = drop_infrequent(data)
data_factorized = data.apply(factorize)
data_norm = normalize(data_factorized)

print("Mode and mean of violation code grouped by suspect race:")
for name, cluster in data_norm.groupby(['SUSP_RACE']):
    print("CLUSTER_" + str(name))
    print("mean: " + str(cluster['PD_CD'].mean()))
    print("mode: " + str(cluster['PD_CD'].mode()))
    print("\n")

print("Mode and mean of violation code grouped by victim race:")
for name, cluster in data_norm.groupby(['VIC_RACE']):
    print("CLUSTER_" + str(name))
    print("mean: " + str(cluster['PD_CD'].mean()))
    print("mode: " + str(cluster['PD_CD'].mode()))
    print("\n")

print("Most common races of victims grouped by suspect races:")
for name, cluster in data.groupby(['SUSP_RACE']):
    print(name)
    print(cluster['VIC_RACE'].value_counts(normalize=True).head(3))
    print("\n")

print("Most common races of suspects grouped by victim races:")
for name, cluster in data.groupby(['SUSP_RACE']):
    print(name)
    print(cluster['VIC_RACE'].value_counts(normalize=True).head(3))
    print("\n")

plot_regression(
    data_factorized,
    x='SUSP_RACE',
    y='VIC_RACE',
    x_estimator=numpy.mean,
    fit_reg=False,
    # lowess=True,
    bottom=0.4, left=0.15
)
pyplot.savefig(
    'suspect_on_victim_race',
    dpi=300,
    bbox_inches='tight'
)
pyplot.close()

plot_regression(
    data_factorized,
    x='VIC_RACE',
    y='SUSP_RACE',
    x_estimator=numpy.mean,
    fit_reg=False,
    # lowess=True,
    bottom=0.4, left=0.15
)
pyplot.savefig(
    'victim_on_suspect_race',
    dpi=300,
    bbox_inches='tight'
)
pyplot.close()