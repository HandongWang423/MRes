'''
This py-script is for sampling a small dataset to test the new model-trainning based on new loss
Run in batch system
Adapted from /Week7/Sampling.py
'''

#Import packages
import pandas as pd
# import numpy as np

filecode = 'InfA_DPrmvd_Bld_x20x30'
#InfA: The inference-aware purpose datasets
#_x2min: Balanced sampling in the double number of least channel - ZH, means the number of WH.

DF = pd.read_parquet(f'/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet').dropna().reset_index(drop=True) 
dfy = pd.DataFrame(DF.iloc[:,-7:].idxmax(axis=1),columns=['proc'])
dfb = DF.iloc[:,:-7]
df = pd.concat([dfb,dfy],axis=1)
DF_proc = df['proc'].astype('int')

samplenum = 14000 #appx = ZH

br = [2,20,30,30,30,3,30]
df_ppidx = DF_proc[DF_proc == 0].index #15 ZH
df_ggHidx = DF_proc[DF_proc == 1].sample(n=br[1]*samplenum).index #50 ZH
df_qqHidx = DF_proc[DF_proc == 2].sample(n=br[2]*samplenum).index # 50 ZH
df_WHidx = DF_proc[DF_proc == 3].index #2 ZH
df_ZHidx = DF_proc[DF_proc == 4].index #1 ZH
df_ttHidx = DF_proc[DF_proc == 5].index #10 ZH
df_tHidx = DF_proc[DF_proc == 6].sample(n=br[6]*samplenum).index #70 ZH


df_pp_sampled = pd.concat([df.iloc[df_ppidx]]*br[0])
print(df_pp_sampled)
df_ggH_sampled = df.iloc[df_ggHidx]
df_qqH_sampled = df.iloc[df_qqHidx]
df_WH_sampled = pd.concat([df.iloc[df_WHidx]]*br[3])
df_ZH_sampled = pd.concat([df.iloc[df_ZHidx]]*br[4])
df_ttH_sampled = pd.concat([df.iloc[df_ttHidx]]*br[5])
df_tH_sampled =df.iloc[df_tHidx]

del df

df_sampled = pd.concat([df_pp_sampled,df_ggH_sampled,df_qqH_sampled,df_ZH_sampled,df_WH_sampled,df_tH_sampled,df_ttH_sampled]).reset_index(drop=True)
DF_proc_sampled = df_sampled['proc']
print(DF_proc_sampled.value_counts())
df_sampled.to_parquet(f'/vols/cms/hw423/Data/Week7/df_old_{filecode}.parquet')


dfx = df_sampled.iloc[:,:-3]
dfw = df_sampled['weight']
dfm = df_sampled['diphotonMass']
print('Data input finished')

dfx.dropna()

dfx_new = pd.DataFrame(columns = dfx.columns)
# for col in dfx.columns:
#     x = dfx[col]
#     x9 = x[x!=-999]
#     mean = x9.mean()
#     sigma = x9.std()
#     if sigma !=0:
#         x9z = (x9-mean)/sigma
#     else:
#         x9z = x9-mean
#     x_new = x
#     x_new[x!=-999] = x9z.values.astype('float32')
#     x_new[x==-999] = 0
#     dfx_new[col]=x_new
dfx_new = dfx
dfx_new = dfx_new.dropna()
dfy =  pd.get_dummies(df_sampled['proc'][dfx_new.index])
df = pd.concat([dfx_new,dfw,dfm,dfy],axis = 1).reset_index(drop=True)


# dfx_new.to_parquet('/vols/cms/hw423/Data/Week6/dfx_new.parquet')
print('Data normalisation finished')
print(df)
df.to_parquet(f'/vols/cms/hw423/Data/Week14/df_{filecode}.parquet')
dfy.to_parquet(f'/vols/cms/hw423/Data/Week14/dfy_{filecode}.parquet')
