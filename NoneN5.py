import pandas as pd
import numpy as np
df = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet')
dfx = df.iloc[:,:-9]
print(dfx.columns)

mk = np.array(dfx.mean()<-0.1)
print(mk)
dfxCol = dfx.columns
dfxCol = dfxCol[mk]
print(dfxCol)
print(dfx[dfxCol])

