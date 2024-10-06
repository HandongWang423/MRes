import pandas as pd

data = pd.read_parquet(f'/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet')

print(data)

parquetcode = 'InfA_RD_DPrmvd_Bld_x10'
df = pd.read_parquet(f'/vols/cms/hw423/Data/Week14/df_{parquetcode}.parquet')
df = df.dropna().reset_index(drop=True)
mask = (df['diphotonMass']<127) & (df['diphotonMass']>123)
df = df[mask]
dfx = df.iloc[:,:-9]

print(dfx)
Col = dfx.columns

data = data[Col]


# col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']
# lb = pd.DataFrame(df.iloc[:,-7:].values,columns=col).idxmax(axis=1)
# print(lb.value_counts())
# label = pd.DataFrame(df.iloc[:,-7:].values).idxmax(axis=1)
# label.to_pickle('/vols/cms/hw423/Data/Week14/Label_DPrmvd.pickle')
# print(pd.concat([df['weight'],lb],axis=1).groupby([0])['weight'].sum())
 

# df = df = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet').reset_index(drop=True) 
# print(df)
# col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']
# lb = pd.DataFrame(df.iloc[:,-7:].values,columns=col).idxmax(axis=1)
# print(lb.value_counts())
# wtd = pd.concat([df['weight'],lb],axis=1).groupby([0])['weight'].sum()
# print(wtd)
# DF_proc = df['proc']
# print(DF_proc)

# samplenum = 14000 #appx = ZH

# br = [5,5,5,2,4,4,4]
# df_ppidx = DF_proc[DF_proc == 0].sample(n=br[0]*samplenum).index
# print(df_ppidx)