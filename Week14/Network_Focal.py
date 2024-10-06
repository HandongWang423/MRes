import pandas as pd
import numpy as np
# import mplhep as hep
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', 150)

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
# pd.set_option('display.max_columns', 150)

######################
#### Parameters ######
######################
parquetcode = 'InfA_DPrmvd'
filecode = 'x60x40_x100_fc'
# filecode = parquetcode

train_hp = {
    "lr":3e-04,
    "batch_size":1280,
    "N_epochs":20,
    "seed":0
}
MInum = 140
nodes = [100,100]
######################
######################

print(filecode)
print(train_hp)

# Set seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(train_hp['seed'])


#Define the Net
class Net(nn.Module):
    def __init__(self, n_features=40, nodes=[100,100], output_nodes=5):
        super(Net, self).__init__()
        # Build network
        n_nodes = [n_features] + nodes + [output_nodes]
        self.layers = nn.ModuleList()
        for i in range(len(n_nodes)-1):
            self.layers.append(nn.Linear(n_nodes[i], n_nodes[i+1]))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        # Apply softmax
        return torch.softmax(out, dim=1)

def focal_loss(output, y_tensor, alpha=1, gamma=2):
    ce_loss = nn.CrossEntropyLoss()
    celoss = ce_loss(output, y_tensor)
    pt = torch.exp(-celoss)
    focal_loss = alpha * (1 - pt) ** gamma * celoss
    return focal_loss.mean()



#Define the trainning function
from NNfunctions import get_batches, get_total_loss
def train_network_cross_entropy(model, X_train,X_test,y_train,y_test, train_hp={}):
    optimiser = torch.optim.Adam(model.parameters(), lr=train_hp["lr"])

    X_train =X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    train_loss, test_loss = [], []

    print(">> Training...")
    with tqdm(range(train_hp["N_epochs"])) as t:
        for i_epoch in t:
            model.train()

            # "get_batches": function defined in statml_tools.py to separate the training data into batches
            batch_gen = get_batches([X_train, y_train], batch_size=train_hp['batch_size'],
                                    randomise=True, include_remainder=False
                                )
            
            for X_tensor, y_tensor in batch_gen:
                optimiser.zero_grad()
                output = model(X_tensor)
                loss = focal_loss(output, y_tensor)
                loss.backward()
                optimiser.step()

            model.eval()
            # "get_total_loss": function defined in statml_tools.py to evaluate the network in batches (useful for large datasets)
            train_loss.append(get_total_loss(model, focal_loss, X_train, y_train))
            test_loss.append(get_total_loss(model, focal_loss, X_test, y_test))
            t.set_postfix(train_loss=train_loss[-1], test_loss=test_loss[-1])

    print(">> Training finished")
    model.eval()

    return model, train_loss, test_loss





######## 
########

##Initialise
mi_series = pd.read_csv('/vols/cms/hw423/Week6/MI_balanced.csv')
MIcol = mi_series.head(MInum)['Features']
model_bce = Net(n_features=140, nodes=nodes, output_nodes=7)
# model_bce = Net(n_features=MInum, nodes=nodes, output_nodes=7)
# dfx_new = pd.DataFrame(columns = MIcol)


#Data Preprocesing - Old
# df = pd.read_parquet(f'/vols/cms/hw423/Data/Week7/df_{parquetcode}.parquet')
# df = df.dropna().reset_index(drop=True)
# dfx = df.iloc[:,:-1]
# dfy =  pd.get_dummies(df['proc'])
# print(dfy)
# print('Data input finished')

#Data Preprocesing - New
df = pd.read_parquet(f'/vols/cms/hw423/Data/Week14/df_{parquetcode}.parquet')
df = df.dropna().reset_index(drop=True)
dfx = df.iloc[:,:-9]
print(dfx.columns)

dfy =  df.iloc[:,-7:]

dfxCol = dfx.columns
Cols= dfxCol


print('Data input finished')

# dfx.dropna()


# #Normalisation
# dfx_new = pd.DataFrame(columns = dfx.columns)
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
#     x_new[x==-999] = -10
#     dfx_new[col]=x_new
# dfx_new = dfx_new.dropna()
# # dfx_new.to_parquet('/vols/cms/hw423/Data/Week6/dfx_new.parquet')
# print('Data normalisation finished')

# del dfx



MIcol = Cols
# print(dfx_new)
dfx_new=dfx[MIcol]

print('MI selection finished')

X_train, X_test, y_train, y_test = train_test_split(dfx_new, dfy.loc[dfx_new.index], test_size=0.2, random_state=42)

del dfx_new
###### Train ######
model_bce, train_loss_bce, test_loss_bce = train_network_cross_entropy(model_bce, X_train, X_test, y_train, y_test, train_hp=train_hp)

##### Store the model ####
torch.save(model_bce.state_dict(), f'/vols/cms/hw423/Data/Week14/model_b_u_{filecode}.pth')
np.save(f'/vols/cms/hw423/Data/Week14/train_loss_bce_{filecode}.npy', np.array(train_loss_bce))
np.save(f'/vols/cms/hw423/Data/Week14/test_loss_bce_{filecode}.npy', np.array(test_loss_bce))
data = pd.read_parquet(f'/vols/cms/hw423/Data/Week14/df_InfA_DPrmvd.parquet')
data = data[MIcol]
print(data)
col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']
oc= model_bce(torch.tensor(data.to_numpy(),dtype=torch.float32))
octest =pd.DataFrame(oc.detach().numpy(), columns = col, index = data.index)
np.save(f'/vols/cms/hw423/Data/Week14/octest_Wd_{filecode}.npy', np.array(octest))
print('ALL FINISHED')