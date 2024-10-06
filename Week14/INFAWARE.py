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
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
torch.set_default_dtype(torch.float64)

parquetcode = 'InfA_RD_DPrmvd_Bld_x40x20'
# parquetcode = 'InfA_RD_DPrmvd_En'
filecode = 'InfAwar_En_DPrmvd'
savecode = 'x40x20_'

train_hp = {
    "lr":0.001,
    "batch_size":20000,
    "N_epochs":100,
    "seed":0,
    'eplim':400
}
MInum = 140
nodes = [200,200]
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
        return F.softmax(out)



# class InfAwareLoss(nn.Module):
#     def __init__(self,i,epoch):
#         super(InfAwareLoss, self).__init__()
#         self.i = i
#         self.epoch =epoch
    
#     def forward(self,input,target,weight):
#         while self.i > self.epoch:
#             # Input = torch.tensor(input)
#             # Target = torch.tensor(target,dtype=torch.int8)
            
#             label = torch.argmax(target,dim=1)
#             pred = torch.argmax(input,dim=1)
#             cm = torch.zeros(7,7)
#             up = pred.unique()
#             ul = label.unique()
#             for p in up:
#                 for l in ul:
#                     cm[p,l] = weight[pred==p][label[pred==p]==l].sum()
#             print(cm)
#             cm =cm[1:, :]
#             O = cm.sum(dim=1)
#             def NLL(mu):
#                 mu0 =torch.tensor([1.0])
#                 theta = torch.cat((mu0,mu))
#                 return -(O@(torch.log(cm@theta+1e-4))-(cm@theta).sum())
#             mu = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0])
#             hess = torch.func.hessian(NLL)(mu)
#             I = torch.inverse(hess_to_tensor(hess))
#             loss = torch.trace(I)**0.5 
#             return loss.clone().detach().requires_grad_(True)
#         else:
#             return torch.tensor([0.0],requires_grad=True)

def smooth_argmax(tensor, dim=-1, temperature=1.0):
    softmax_tensor =F.softmax(tensor / temperature, dim=dim)
    return softmax_tensor
def confusion_matrix(OC,label,weight,model):
    label_w = weight.unsqueeze(1)*label
    pred_matrix = smooth_argmax(model(OC),temperature=.01,dim=1)
    confusion_matrix = torch.matmul(pred_matrix.t(),label_w)[1:,:]
    return confusion_matrix
def nll(theta1,OC,label,weight,model):
    cm = confusion_matrix(OC,label,weight,model)
    O = torch.sum(cm,dim=1)
    theta0 = torch.ones(1)
    theta = torch.cat([theta0,theta1])
    return -(O@(torch.log(cm@theta))-torch.sum((cm@theta)))
def InfAwareLoss(input,label,weight, model,i,eplim,theta_init = torch.ones(6)):
    while i > eplim:
        theta = torch.tensor(theta_init)
        hess = torch.func.hessian(nll,0)(theta,input,label,weight,model)
        H = hess_to_tensor(hess)
        I = torch.inverse(H)
        return torch.trace(I)**0.5
    else:
        return torch.zeros(1)



#Define the trainning function
from NNfunctions import get_batches, get_total_loss,get_total_lossM
def train_network_cross_entropy(model, X_train,X_test,y_train,y_test,w_train,w_test, train_hp={}):
    optimiser = torch.optim.Adam(model.parameters(), lr=train_hp["lr"])
    X_train =X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    w_train = w_train.to_numpy()
    w_test = w_test.to_numpy()
    
    
    train_loss, test_loss = [], []
    i = 0
    eplim = train_hp['eplim']
    epchs = train_hp['N_epochs']
    mu_ini = torch.ones(6)
    ia_loss = lambda x,y,w,m,i: InfAwareLoss(x,y,w,m,i,eplim,mu_ini)
    ce_loss = nn.CrossEntropyLoss()

    print(">> Training...")
    with tqdm(range(train_hp["N_epochs"])) as t:
        for i_epoch in t:
            model.train()
            # print(i)
            # "get_batches": function defined in statml_tools.py to separate the training data into batches
            batch_gen = get_batches([X_train, y_train, w_train], batch_size=train_hp['batch_size'],
                                    randomise=True, include_remainder=False)
            for X_tensor, y_tensor, w_tensor in batch_gen:
                optimiser.zero_grad()
                if i <= eplim:
                    output = model(X_tensor)
                    loss = ce_loss(output, y_tensor)
                else:
                    loss = ia_loss(X_tensor, y_tensor, w_tensor,model,i)
                loss.backward()
                optimiser.step()
                

            model.eval()
            if i>eplim:
                Loss = ia_loss
                train_loss.append(get_total_lossM(model, Loss, X_train, y_train,w_train,i))
                test_loss.append(get_total_lossM(model, Loss, X_test, y_test,w_test,i))
            else:
                Loss = ce_loss
                train_loss.append(get_total_loss(model, Loss, X_train, y_train))
                test_loss.append(get_total_loss(model, Loss, X_test, y_test))
            # "get_total_loss": function defined in statml_tools.py to evaluate the network in batches (useful for large datasets)
            
            t.set_postfix(train_loss=train_loss[-1], test_loss=test_loss[-1])
            i+=1

    print(">> Training finished")
    model.eval()

    return model, train_loss, test_loss






df = pd.read_parquet(f'/vols/cms/hw423/Data/Week14/df_{parquetcode}.parquet')
dfx = df.iloc[:,:138]
model_ia = Net(n_features=138, nodes=nodes, output_nodes=7)

dfy =  df.iloc[:,-7:]
# dfy = pd.get_dummies(df.iloc[:,-1])
dfw = df['weight']

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(dfx,dfy,dfw, test_size=0.2, random_state=42)

model_bce, train_loss_ia, test_loss_ia = train_network_cross_entropy(model_ia, X_train, X_test, y_train, y_test,w_train,w_test, train_hp=train_hp)


# data = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd_En.parquet')

data = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet')
data = data.iloc[:,:138]
# col14 = ['$\gamma\gamma$-0','$\gamma\gamma$-1','ggH-0','ggH-1','qqH-0','qqH-1','WH-0','WH-1','ZH-0','ZH-1','ttH-0','ttH-1','tH-0','tH-1']
col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']
oc= model_bce(torch.tensor(data.to_numpy(),dtype = torch.float64))
# octest =pd.DataFrame(oc.detach(), columns = col14, index = data.index)
octest =pd.DataFrame(oc.detach(), columns = col, index = data.index)
np.save(f'/vols/cms/hw423/Data/Week14/Combined_octest_{filecode}_{savecode}.npy', np.array(octest))
np.save(f'/vols/cms/hw423/Data/Week14/Post_test_loss_{filecode}_{savecode}.npy',np.array(test_loss_ia))
np.save(f'/vols/cms/hw423/Data/Week14/Post_train_loss_{filecode}_{savecode}.npy',np.array(train_loss_ia))