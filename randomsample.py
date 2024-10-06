'''
This py-script is for sampling a small dataset to test the new model-trainning based on new loss
Run in batch system
Adapted from /Week7/Sampling.py
'''

#Import packages
import pandas as pd
import numpy as np

filecode = 'InfA_DPrmvd'
#InfA: The inference-aware purpose datasets
#RD_DPrmvd: Random sampled with removed diphoton mass<123 and >127
#DPrmvd: Random sampled with removed diphoton mass<120 and >130
df = pd.read_parquet(f'/vols/cms/hw423/Data/DF.parquet').dropna().reset_index(drop=True) 
DF_proc = df['proc']
diphoton_mass_mask = (df['diphotonMass']<130) & (df['diphotonMass']>120) 
df_sampled = df[diphoton_mass_mask].reset_index(drop=True)

DF_proc_sampled = df_sampled['proc']
df_sampled.to_parquet(f'/vols/cms/hw423/Data/Week7/df_old_{filecode}.parquet')


dfx = df_sampled.iloc[:,:-2]
dfw = df_sampled['weight']
dfm = df_sampled['diphotonMass']

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
    x_new[x==-999] = -5
    dfx_new[col]=x_new
dfx_new = dfx_new.dropna().drop(['diphotonMass'],axis=1)
dfy =  pd.get_dummies(df_sampled['proc'][dfx_new.index])
df = pd.concat([dfx_new,dfw,dfm,dfy],axis = 1).reset_index(drop=True)

label = dfy.iloc[:,-7:].idxmax(axis=1)

# dfx_new.to_parquet('/vols/cms/hw423/Data/Week6/dfx_new.parquet')
print('Data normalisation finished')
print(df)
df.to_parquet(f'/vols/cms/hw423/Data/Week14/df_{filecode}.parquet')
label.to_pickle(f'/vols/cms/hw423/Data/Week14/label_{filecode}.pkl')
dfw.to_pickle(f'/vols/cms/hw423/Data/Week14/weight_{filecode}.pkl')
