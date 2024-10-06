
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
hep.style.use("CMS")
# from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', 150)
import os


filecode = 'x30x20_DPrmvd_100'

folder_name = f"/vols/cms/hw423/Week14/plots/{filecode}"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"folder'{folder_name}' has made")
else:
    print(f"folder '{folder_name}' already exists")

col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']

#Old
# y = np.load(f'/vols/cms/hw423/Data/Week7/y_{filecode}')
# y_pred = torch.load(f'/vols/cms/hw423/Data/Week7/y_pred_{filecode}')
# oc = torch.load(f'/vols/cms/hw423/Data/Week7/oc_{filecode}')
# octest = torch.load(f'/vols/cms/hw423/Data/Week7/octest_{filecode}')

#New
df = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet')
oc = np.load(f'/vols/cms/hw423/Data/Week14/octest_{filecode}.npy')
label = df.iloc[:,-7:].idxmax(axis=1).astype(int)
label = pd.DataFrame(label.values,columns=['true'])
pred = pd.DataFrame(oc.argmax(axis=1),columns=['pred'])
dfw = df['weight']
# octest = torch.load(f'/vols/cms/hw423/Data/Week7/octest_{filecode}')

result=pd.concat([label,pred,dfw],axis=1)
print(result)

cm_w = result.groupby(['pred','true'])['weight'].sum().unstack(fill_value=0).reindex(fill_value=0).iloc[1:,1:].T
print(cm_w)
CM = []
for i in range(6):
    cl = cm_w.iloc[:,i]/cm_w.iloc[:,i].sum()
    CM.append(cl)
cm = pd.DataFrame(CM).T

# Adjusted plotting function that includes saving the figures
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues, filename='/vols/cms/hw423/Week14/plots/{filecode}/confusion_matrix_c.pdf'):
    fig, ax = plt.subplots(figsize=(10,8))
    ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap)  # Using '.2f' for floating point format
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    ax.set_xticklabels(col[1:])
    ax.set_yticklabels(col[1:])
    fig.savefig(filename, bbox_inches='tight')  
    # ax.show()  

# The unnormalized confusion matrix
#plot_confusion_matrix(cm, title='Confusion Matrix (Unnormalized)', filename='Project/New/Week5/plots/Confusion_matrix_unnorm.pdf')

# The row-normalized confusion matrix
# plot_confusion_matrix(cm_row_norm, title='Confusion Matrix \n(Row-normalized, Wtd)', cmap=plt.cm.Greens, filename=f'{folder_name}/Confusion_matrix_row_{filecode}.pdf')

# The column-normalized confusion matrix
plot_confusion_matrix(cm, title='Confusion Matrix \n(Column-normalized, Wtd)', cmap=plt.cm.Oranges, filename=f'{folder_name}/Confusion_matrix_col_{filecode}.pdf')

# # The column-normalized confusion matrix
# plot_confusion_matrix(cm_col_norm_wgg, title='Confusion Matrix \n(Column-normalized, Weighted without yy)', cmap=plt.cm.Oranges, filename=f'{folder_name}/Confusion_matrix_col_wgg_{filecode}.pdf')

# # The column-normalized confusion matrix
# plot_confusion_matrix(cm_col_norm_wpgg_2, title='Confusion Matrix \n(Column-normalized, Weighted without p_yy>0.2)', cmap=plt.cm.Oranges, filename=f'{folder_name}/Confusion_matrix_col_wpgg_2_{filecode}.pdf')
