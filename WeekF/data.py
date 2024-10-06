import pandas as pd

df = pd.read_parquet('/vols/cms/hw423/Data/DF_all.parquet')
columns = df.columns
columns_df = pd.DataFrame(columns)
columns_df.to_csv('/vols/cms/hw423/Data/columns.csv', index=False, header=False)
sp = df.sample(n=500000)
sp.to_parquet('/vols/cms/hw423/Data/sampled.parquet')

dfm = df['diphotonMass']
dfw = df['weight']
dfy = df['proc']
dfmass = pd.concat([dfm,dfw,dfy],axis=1)
dfmass.to_parquet('/vols/cms/hw423/Data/mass.parquet')
