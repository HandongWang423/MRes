import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

parquetcode = 'InfA_DPrmvd'
# parquetcode = 'InfA_DPrmvd_Bld_x40x60'
# filecode = 'InfAwar_En_DPrmvd'
savecode = 'sftmx_md10_wtd'

df = pd.read_parquet(f'/vols/cms/hw423/Data/Week14/df_{parquetcode}.parquet')
dfx = df.iloc[:,:140]
dfw = df['weight']
mask =(dfw<0)
dfw[mask]=0

# dfy =  pd.read_pickle(f'/vols/cms/hw423/Data/Week14/label_{parquetcode}.pkl')
dfy = df.iloc[:,-7:].idxmax(axis=1).astype('int')
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softprob', 
    max_depth=10,
    num_class=7)
print('Training Start')
xgb_classifier.fit(dfx,dfy,sample_weight=dfw)
xgb_classifier.save_model(f'/vols/cms/hw423/Data/Week14/xgb_model_{savecode}.json')
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.load_model(f'/vols/cms/hw423/Data/Week14/xgb_model_{savecode}.json')
data = pd.read_parquet(f'/vols/cms/hw423/Data/Week14/df_InfA_DPrmvd.parquet')
dfx = data.iloc[:,:140]
y_pred = xgb_classifier.predict_proba(dfx)
np.save(f'/vols/cms/hw423/Data/Week14/octest_xgb_Wd_{savecode}.npy',y_pred)
# np.save(f'/vols/cms/hw423/Data/Week14/octest_xgb_Wd_{savecode}.npy',y_pred)
