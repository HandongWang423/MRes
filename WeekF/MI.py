import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
import warnings
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf', 'svg')
set_matplotlib_formats('retina')
plt.rcParams['figure.dpi'] = 60
warnings.filterwarnings("ignore", category=UserWarning)

print('Packaged Imported')
df = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_DPrmvd.parquet').reset_index(drop=True)
dfx = df.iloc[:,:140]
MI = np.zeros((140,7))
MIcol = np.zeros(141)
col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']


dfy = df.iloc[:,-7:]
dfl = dfy.idxmax(axis=1)
for i in range(141):
    df1=  dfx.iloc[:,i].reset_index(drop=True)
    x1 = np.array(df1).flatten()
    MIcol[i]= mutual_info_score(x1,dfl)
    print(i)
    
dfMIcol = pd.DataFrame(MIcol,index=dfx.columns,columns = ['MI'])
MIC = MI.sum(axis=0)
MIN = []
for i in range(7):
    MIN.append(MI[:,i]/MIC[i])

dfMI = pd.DataFrame(np.array(MIN).T,index=dfx.columns,columns=col).drop(['diphotonMass'])
MIlist = pd.DataFrame(dfMI.T.sum(axis=0),index = dfMI.index,columns=['MI'])

MIlist.sort_values(by='MI',ascending=False)
dfMIcol.sort_values(by=['MI'],ascending=False)

import seaborn as sns
fig,ax = plt.subplots(2,1,figsize=(35,20))
sns.heatmap(dfMI.iloc[:80,:].T,cmap='Greens',cbar=False,ax=ax[0])
sns.heatmap(dfMI.iloc[80:,:].T,cmap='Greens',cbar_kws={"pad": 0.01},ax=ax[1])
hep.cms.label(ax=ax[0],rlabel="(Continuous)")
ax[1].set_xlabel('Features')
ax[0].set_ylabel('Labels')
ax[1].set_ylabel('Labels')
plt.subplots_adjust(hspace=4)
cbar = ax[1].collections[0].colorbar
cbar.set_label('MI')
fig.tight_layout()
fig.savefig('/vols/cms/hw423/WeekF/plots/MI.pdf',bbox_inches='tight')


        
