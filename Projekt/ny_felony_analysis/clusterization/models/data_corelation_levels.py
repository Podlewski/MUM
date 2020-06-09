import numpy
import pandas
import seaborn
from matplotlib import pyplot

from ny_felony_analysis.clusterization._utils import plot_regression
from utils.common import factorize, correlate_sort, drop_infrequent

seaborn.set(color_codes=True)

data = pandas.read_csv('../../NYPD_Felony_Data.csv')
data = data.drop(
    columns=['CMPLNT_TO_DT', 'CMPLNT_TO_TM'],
    axis=1
)
data = data.dropna()
data = drop_infrequent(data)
data = data.apply(factorize)
print(correlate_sort(data, 0.2).to_string())

plot_regression(
    data.sample(n=13_000, random_state=666),
    x='SUSP_RACE', y='VIC_RACE',
    x_estimator=numpy.mean,
    # lowess=True,
    bottom=0.4, left=0.15
)
pyplot.savefig(
    'regression_susp_race_to_vic_race',
    dpi=300,
    bbox_inches='tight'
)
pyplot.close()

plot_regression(
    data.sample(n=13_000, random_state=666),
    x='PD_CD', y='SUSP_AGE_GROUP',
    x_estimator=numpy.mean,
    # lowess=True,
    bottom=0.4, left=0.15
)
pyplot.savefig(
    'regression_susp_pd_to_susp_age_race',
    dpi=300,
    bbox_inches='tight'
)
pyplot.close()