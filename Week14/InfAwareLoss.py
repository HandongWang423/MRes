import pandas as pd
import numpy as np
# import mplhep as hep
from tqdm import tqdm
import  matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', 150)

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from iminuit import Minuit

parquetcode = 'x30x20'
filecode = '_InfAwar_test'

train_hp = {
    "lr":0.001,
    "batch_size":300000,
    "N_epochs":200,
    "seed":0,
    'eplim':80
}
MInum = 140
nodes = [100,100]
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(train_hp['seed'])

def hess_to_tensor(H):
    hess_elements = []
    for i in range(len(H)):
        for j in range(len(H)):
            hess_elements.append(H[i][j].reshape(1))
    return torch.cat(hess_elements).reshape(len(H),len(H))

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

class InfAwareLoss(nn.Module):
    def __init__(self,weight,i,epoch):
        super(InfAwareLoss, self).__init__()
        self.weight = weight
        self.i = i
        self.epoch =epoch
    
    def forward(self,input,target):
        print(self.i, self.epoch)
        while self.i > self.epoch:
            # Input = torch.tensor(input)
            # Target = torch.tensor(target,dtype=torch.int8)
            label = torch.argmax(target,dim=1)
            pred = torch.argmax(input,dim=1)
            # plt.hist(pred)
            plt.show()
            weight = torch.tensor(self.weight.values)
            cm = torch.zeros(7,7)
            for t, p, w in zip(label.view(-1), pred.view(-1), weight.view(-1)):
                cm[p,t] += 1
            cm =cm[1:, :]
            O = cm.sum(dim=1)
            # print(cm)
            # print(O)
            def NLL(mu):
                mu0 =torch.tensor([1.0])
                theta = torch.cat((mu0,mu))
                return -(O@(torch.log(cm@theta+1e-3))-(cm@theta).sum())
            mu = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0],requires_grad=True)
            hess = torch.func.hessian(NLL)(mu)
            # print(hess)
            # print(torch.det(hess))
            I = torch.inverse(hess_to_tensor(hess))
            # print(hess)
            loss = torch.trace(I)**0.5 
            print(loss)
            return loss.clone().detach().requires_grad_(True)
        else:
            return 0


#Define the trainning function
from NNfunctions import get_batches, get_total_loss
def train_network_cross_entropy(model, X_train,X_test,y_train,y_test,weight, train_hp={}):
    optimiser = torch.optim.Adam(model.parameters(), lr=train_hp["lr"])

    X_train =X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    
    ce_loss = nn.CrossEntropyLoss()
    train_loss, test_loss = [], []
    i = 0
    eplim = train_hp['eplim']
    epchs = train_hp['N_epochs']

    print(">> Training...")
    with tqdm(range(train_hp["N_epochs"])) as t:
        for i_epoch in t:
            model.train()
            # print(i)
            # "get_batches": function defined in statml_tools.py to separate the training data into batches
            batch_gen = get_batches([X_train, y_train], batch_size=train_hp['batch_size'],
                                    randomise=True, include_remainder=False
                                )
            ia_loss = InfAwareLoss(weight,i,eplim)
            for X_tensor, y_tensor in batch_gen:
                optimiser.zero_grad()
                output = model(X_tensor)
                # print(output)
                ia = ia_loss(output, y_tensor)
                ce = ce_loss(output,y_tensor)
                loss = i/epchs*ia+ (1-i/epchs)*ce
                loss.backward()
                print(ia,ce)
                optimiser.step()

            model.eval()
            # if i > eplim:
            #     # print('Yes')
            #     Loss = ia_loss
            # else:
            #     Loss = ce_loss
            Loss = ce_loss
            # "get_total_loss": function defined in statml_tools.py to evaluate the network in batches (useful for large datasets)
            train_loss.append(get_total_loss(model, Loss, X_train, y_train))
            test_loss.append(get_total_loss(model, Loss, X_test, y_test))
            t.set_postfix(train_loss=train_loss[-1], test_loss=test_loss[-1])
            i+=1

    print(">> Training finished")
    model.eval()

    return model, train_loss, test_loss

df = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet')
mi_series = pd.read_csv('/vols/cms/hw423/Week6/MI_balanced.csv')
MIcol = mi_series.head(MInum)['Features']
model_ia = Net(n_features=MInum, nodes=nodes, output_nodes=7)
dfx = df[MIcol]
dfy =  df.iloc[:,-7:]
dfw = df['weight']
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy.loc[dfx.index], test_size=0.2, random_state=42)
print('Begin to train')
model_bce, train_loss_bce, test_loss_bce = train_network_cross_entropy(model_ia, X_train, X_test, y_train, y_test, train_hp=train_hp,weight=dfw)

data = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet')
data = data[MIcol]
col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']
oc= model_bce(torch.tensor(data.to_numpy(),dtype=torch.float32))
octest =pd.DataFrame(oc.detach(), columns = col, index = data.index)
np.save(f'/vols/cms/hw423/Data/Week14/octest_{filecode}.npy', np.array(octest))