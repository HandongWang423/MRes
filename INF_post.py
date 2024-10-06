import pandas as pd
import numpy as np
# import mplhep as hep
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', 150)

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
# pd.set_option('display.max_columns', 150)



filecode = 'InfA_RD_DPrmvd'
# filecode = parquetcode

train_hp = {
    "lr":0.001,
    "batch_size":1000000,
    "N_epochs":5,
    "seed":0,
}
nodes = [7,7,7,7,7]


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
    def __init__(self, n_features=7, nodes=[7,7], output_nodes=7,temp=0.0001):
        super(Net, self).__init__()
        self.temperature = temp
        # Build network
        n_nodes = [n_features] + nodes + [output_nodes]
        self.layers = nn.ModuleList()
        for i in range(len(n_nodes)-1):
            linear_layer = nn.Linear(n_nodes[i], n_nodes[i+1])

            with torch.no_grad():
                linear_layer.weight.copy_(torch.eye(n_nodes[i+1], n_nodes[i]))

            with torch.no_grad():
                linear_layer.bias.zero_()
            self.layers.append(linear_layer)
            self.layers.append(nn.ReLU())
        
        
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.copy_(torch.eye(7))
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     print('INITIALED')


    def forward(self, x):
        out = torch.tensor(self.layers[0](x),dtype=x.dtype)
        for layer in self.layers[1:]:
            out = F.relu((layer(out)))
        # return out
            out = out / self.temperature
            return torch.softmax(out, dim=1)
        
    def set_temperature(self, temp):
        self.temperature = temp


# def InfAwareLoss(input,target,weight,model):
    
#     def NLL(mu,input,label,model,weight):
#         input = model(input)
#         input = torch.tensor(input)
#         label = torch.argmax(target,dim=1)
#         pred = torch.argmax(input,dim=1)
#         cm = torch.zeros(7,7)
#         up = pred.unique()
#         ul = label.unique()
#         for p in up:
#             for l in ul:
#                 cm[p,l] = torch.sum(weight[pred==p][label[pred==p]==l])
#         cm =cm[1:, :]
#         # print(cm)
#         O = torch.sum(cm,dim=1)
#         mu = mu.view(-1)
#         mu0 =torch.tensor([1.0]).view(-1)
#         # mu1 = mu[0].view(-1)
#         # mu2 = mu[1].view(-1)
#         # mu3 = mu[2].view(-1)
#         # mu4 = mu[3].view(-1)
#         # mu5 = mu[4].view(-1)
#         # mu6 = mu[5].view(-1)
#         # theta = torch.cat((mu0,mu1,mu2,mu3,mu4,mu5,mu6))
#         theta = torch.cat((mu0,mu))
#         return -(O@(torch.log(cm@theta))-torch.sum((cm@theta)))
#     mu = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0])
#     hess = torch.func.hessian(NLL,(0))(mu,input,label,model,weight)
#     I = torch.inverse(hess_to_tensor(hess))
#     loss = torch.trace(I)**0.5/1000
#     return loss
def NLL(mu1,mu2,mu3,mu4,mu5,mu6,input,target,model,weight):
    outcome = model(input)
    # input = torch.tensor(input).view(-1)
    # label =label.view(-1)
    label = torch.argmax(target,dim=1)
    pred = torch.argmax(outcome,dim=1)
    cm = torch.zeros(7,7)
    up = pred.unique()
    ul = label.unique()
    for p in up:
        for l in ul:
            cm[p,l] = torch.sum(weight[pred==p][label[pred==p]==l])
    cm =cm[1:, :]
    O = torch.sum(cm,dim=1)
    mu1 = mu1.view(-1)
    mu2 = mu2.view(-1)
    mu3 = mu3.view(-1)
    mu4 = mu4.view(-1)
    mu5 = mu5.view(-1)
    mu6 = mu6.view(-1)
    mu0 =torch.tensor([1.0],requires_grad=True).view(-1)
    theta = torch.cat([mu0,mu1,mu2,mu3,mu4,mu5,mu6])
    return -(O@(torch.log(cm@theta))-torch.sum((cm@theta)))

def InfAwareLoss(input,target,weight,model,mu_init):

    # mu1 = mu_init['mu1']
    # mu2 = mu_init['mu2']
    # mu3 = mu_init['mu3']
    # mu4 = mu_init['mu4']
    # mu5 = mu_init['mu5']
    # mu6 = mu_init['mu6']

    mu1 = torch.tensor(mu_init['mu1'], requires_grad=True)
    mu2 = torch.tensor(mu_init['mu2'], requires_grad=True)
    mu3 = torch.tensor(mu_init['mu3'], requires_grad=True)
    mu4 = torch.tensor(mu_init['mu4'], requires_grad=True)
    mu5 = torch.tensor(mu_init['mu5'], requires_grad=True)
    mu6 = torch.tensor(mu_init['mu6'], requires_grad=True)
    

    hess = torch.func.hessian(NLL,(0,1,2,3,4,5))(mu1,mu2,mu3,mu4,mu5,mu6,input,target,model,weight)
    
    I = torch.inverse(hess_to_tensor(hess))
    loss = torch.trace(I**0.5)
    for name, param in model.named_parameters():
                    print(f"梯度 {name}: {param.grad}")
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")
    print(loss)
    return loss

#Define the trainning function
from NNfunctions import get_batches, get_total_loss,get_total_lossM
def train_network(model, X_train,X_test,y_train,y_test,w_train,w_test, train_hp={}):
    optimiser = torch.optim.Adam(model.parameters(), lr=train_hp["lr"])
    # optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    X_train =X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    w_train = w_train.to_numpy()
    w_test = w_test.to_numpy()
    
    mu_ini_ex = {'mu1': 1.0, 'mu2': 1.0, 'mu3': 1.0, 'mu4': 1.0, 'mu5': 1.0, 'mu6': 1.0}
    mu_ini = {key: torch.tensor(value) for key, value in mu_ini_ex.items()}
    
    
    train_loss, test_loss = [], []
    ia_loss = lambda x,y,z,m: InfAwareLoss(x,y,z,m,mu_ini)

    print(">> Training...")
    with tqdm(range(train_hp["N_epochs"])) as t:
        for i_epoch in t:
            model.train()
            
            # print(i)
            # "get_batches": function defined in statml_tools.py to separate the training data into batches
            batch_gen = get_batches([X_train, y_train, w_train], batch_size=train_hp['batch_size'],
                                    randomise=True, include_remainder=False
                                )
            
            for X_tensor, y_tensor, w_tensor in batch_gen:
                optimiser.zero_grad()
                loss = ia_loss(X_tensor, y_tensor, w_tensor,model)
                if torch.isnan(loss):
                    raise ValueError("Loss is NaN, terminating training")
                loss.backward()
                optimiser.step()
                

                for name, param in model.named_parameters():
                    print(f"梯度 {name}: {param.grad}")
                

            model.eval()
            for layer in model_ia.layers:
                print(i_epoch,layer.weight.data)
            
            Loss = ia_loss
            train_loss.append(get_total_lossM(model, Loss, X_train, y_train,w_train))
            test_loss.append(get_total_lossM(model, Loss, X_test, y_test,w_test))
            
            # "get_total_loss": function defined in statml_tools.py to evaluate the network in batches (useful for large datasets)
            
            t.set_postfix(train_loss=train_loss[-1], test_loss=test_loss[-1])


    print(">> Training finished")
    model.eval()

    return model, train_loss, test_loss
mi_series = pd.read_csv('/vols/cms/hw423/Week6/MI_balanced.csv')
MIcol = mi_series.head(140)['Features']


oc = np.load(f'/vols/cms/hw423/Data/Week14/octest_{filecode}.npy')
df = pd.DataFrame(oc)
dfx = df
# mi_series = pd.read_csv('/vols/cms/hw423/Week6/MI_balanced.csv')
# df = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet')

# dfx=df[MIcol]
label = pd.read_pickle('/vols/cms/hw423/Data/Week14/Label.pkl')
dfy = pd.get_dummies(label)
dfw = pd.read_pickle('/vols/cms/hw423/Data/Week14/weight.pkl')

# model_ia = Net(n_features=140, nodes=nodes, output_nodes=7)
# model_ia.load_state_dict(torch.load(f'/vols/cms/hw423/Data/Week14/model_b_u_x30x20_DPrmvd_100.pth'))

model_ia = Net(n_features=7, nodes=nodes, output_nodes=7)
model_ia.apply(model_ia._init_weights)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(dfx, dfy,dfw, test_size=0.2, random_state=42)

model, train_loss_bce, test_loss_bce = train_network(model_ia, X_train, X_test, y_train, y_test,w_train,w_test, train_hp=train_hp)


# data = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet')
data = df
oc= model(torch.tensor(data.to_numpy(),dtype=torch.float32))
octest =pd.DataFrame(oc.detach(), index = data.index)
np.save(f'/vols/cms/hw423/Data/Week14/octest_P_{filecode}.npy', np.array(octest))