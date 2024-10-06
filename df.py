import pandas as pd
df_sampled = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_x30x20.parquet').reset_index(drop=True)
dfx = df_sampled.iloc[:,:-2]
dfx  = dfx.drop(columns=['diphotonMass'])
# dfw = df_sampled['weight']
# dfm = df_sampled['diphotonMass']
print('Data input finished')

dfx.dropna()

dfx_new = pd.DataFrame(columns = dfx.columns)
for col in dfx.columns:
    x = dfx[col]
    x9 = x[x!=-999]
    mean = x9.mean()
    sigma = x9.std()
    if sigma !=0:
        x9z = (x9-mean)/sigma
    else:
        x9z = x9-mean
    x_new = x
    x_new[x!=-999] = x9z.values.astype('float32')
    x_new[x==-999] = -10
    dfx_new[col]=x_new
dfx_new = dfx_new.dropna()
dfy =  pd.get_dummies(df_sampled['proc'][dfx_new.index])
dfw = df_sampled['weight'][dfx_new.index]
dfm = df_sampled['diphotonMass'][dfx_new.index]

df = pd.concat([dfx_new,dfw,dfm,dfy],axis = 1).reset_index(drop=True)
print(df)
df.to_parquet('/vols/cms/hw423/Data/Week14/df_InfA_x30x20.parquet')
