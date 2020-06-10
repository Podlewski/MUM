import pandas
import seaborn
from datetime import datetime
from io import StringIO

from utils.common import drop_infrequent


def parse_date_to_daytime(time):
    hour = datetime.strptime(time, '%H:%M:%S').time().hour
    if 4 <= hour <= 12:
        return "MORNING"
    elif 12 < hour <= 18:
        return "AFTERNOON"
    elif 18 < hour <= 23:
        return "EVENING"
    else:
        return "NIGHT"


seaborn.set(color_codes=True)

data = pandas.read_csv('../../NYPD_Felony_Data.csv')
data = data.drop(
    columns=['CMPLNT_TO_DT', 'CMPLNT_TO_TM'],
    axis=1
)
data = data.dropna()
data = drop_infrequent(data)
data['DAY_TIME'] = data['CMPLNT_FR_TM']
data['DAY_TIME'] = data['DAY_TIME'].map(parse_date_to_daytime)

print("BOROUGH, AFTERNOON, EVENING, MORNING, NIGHT")
for neigh_name, neigh in data.groupby(['BORO_NM']):
    print(neigh_name, end=", ")
    for daytime_name, daytime in neigh.groupby(['DAY_TIME']):
        print(round(float(len(daytime.index) / len(neigh.index)), 3), end=", ")
    print("\n")


