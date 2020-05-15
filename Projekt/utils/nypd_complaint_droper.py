import pandas as pd 

df = pd.read_csv('NYPD_Complaint_Data_Historic.csv', low_memory=False)

df.drop([df.columns[0],  df.columns[6],  df.columns[8],  df.columns[10],
         df.columns[16], df.columns[17], df.columns[18], df.columns[19],
         df.columns[20], df.columns[21], df.columns[22], df.columns[26],
         df.columns[29], df.columns[30], df.columns[31]],
         axis='columns', inplace=True)

df.drop(df[df['CRM_ATPT_CPTD_CD'] != 'COMPLETED'].index, inplace=True)
df.drop(df[df['LAW_CAT_CD'] != 'FELONY'].index, inplace=True)

df.drop(['CRM_ATPT_CPTD_CD', 'LAW_CAT_CD'], axis='columns', inplace=True)

df.drop(df[df['VIC_SEX'] == 'D'].index, inplace=True)
df.drop(df[df['VIC_SEX'] == 'E'].index, inplace=True)

df.to_csv('NYPD_Felony_Data.csv', encoding='utf-8', index=False)