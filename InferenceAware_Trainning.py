import pandas as pd
import numpy as np

from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', 150)

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
torch.set_default_dtype(torch.float64)



# filecode = 'InfAwar_test_rmvd_ubld_40'
filecode = 'Wd_x100_medium'
# filecode = 'xgb_Wd_sftmx_md10'
savecode = 'x10'
train_hp = {
    "lr":0.0001,
    "batch_size":50000,
    "N_epochs":10,
    "seed":0,
}
nodes = [20,20]


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



class Net(nn.Module):
    def __init__(self, n_features=7, nodes=[7,7], output_nodes=7,temp=0.0001):
        super(Net, self).__init__()
        self.temperature = temp
        # Build network
        n_nodes = [n_features] + nodes + [output_nodes]
        self.layers = nn.ModuleList()
        for i in range(len(n_nodes) - 1):
            linear_layer = nn.Linear(n_nodes[i], n_nodes[i+1])

            with torch.no_grad():
                linear_layer.weight.copy_(torch.eye(n_nodes[i+1], n_nodes[i]))
            with torch.no_grad():
                linear_layer.bias.zero_()
            self.layers.append(linear_layer)
            
            self.layers.append(nn.ReLU())
        
        


    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        return F.softmax(out)
        
    def set_temperature(self, temp):
        self.temperature = temp

def smooth_argmax(tensor, dim=-1, temperature=1.0):
    softmax_tensor =F.softmax(tensor / temperature, dim=dim)
    return softmax_tensor
def confusion_matrix(OC,label,weight,model):
    label_w = weight.unsqueeze(1)*label
    pred_matrix = smooth_argmax(model(OC),temperature=.0001,dim=1)
    confusion_matrix = torch.matmul(pred_matrix.t(),label_w)[1:,:]
    return confusion_matrix
def nll(theta1,OC,label,weight,model):
    cm = confusion_matrix(OC,label,weight,model)
    O = torch.sum(cm,dim=1)
    theta0 = torch.ones(1)
    theta = torch.cat([theta0,theta1])
    return -(O@(torch.log(cm@theta))-torch.sum((cm@theta)))
def InfAwareLoss(input,label,weight, model,theta_init = torch.ones(6)):
    theta = torch.tensor(theta_init)
    hess = torch.func.hessian(nll,0)(theta,input,label,weight,model)
    H = hess_to_tensor(hess)
    print(H)
    I = torch.inverse(H)
    return torch.trace(I)**(1/2)

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
    
    train_loss, test_loss = [], []
    mu_ini = torch.ones(6)
    ia_loss = lambda x,y,w,m: InfAwareLoss(x,y,w,m,mu_ini)

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
            

            model.eval()
            # for layer in model.layers:
            #     if isinstance(layer, nn.Linear):
            #         print(i_epoch, layer.weight.data)
            torch.save(model(torch.tensor(oc,dtype = torch.float64)),'/vols/cms/hw423/Data/Week14/oc_test_ia.pt')
            Loss = ia_loss
            train_loss.append(get_total_lossM(model, Loss, X_train, y_train,w_train)/1e4)
            test_loss.append(get_total_lossM(model, Loss, X_test, y_test,w_test)/1e4)
            
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
label = pd.read_pickle('/vols/cms/hw423/Data/Week14/label_InfA_DPrmvd.pkl')
dfy = pd.get_dummies(label)
dfw = pd.read_pickle('/vols/cms/hw423/Data/Week14/weight_InfA_DPrmvd.pkl')

# model_ia = Net(n_features=140, nodes=nodes, output_nodes=7)
# model_ia.load_state_dict(torch.load(f'/vols/cms/hw423/Data/Week14/model_b_u_x30x20_DPrmvd_100.pth'))

model_ia = Net(n_features=7, nodes=nodes, output_nodes=7)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(dfx, dfy,dfw, test_size=0.2, random_state=42)

model, train_loss_ia, test_loss_ia = train_network(model_ia, X_train, X_test, y_train, y_test,w_train,w_test, train_hp=train_hp)


# data = pd.read_parquet('/vols/cms/hw423/Data/Week14/df_InfA_RD_DPrmvd.parquet')
data = df
oc= model(torch.tensor(data.to_numpy(),dtype=torch.float64))
octest =pd.DataFrame(oc.detach(), index = data.index)
np.save(f'/vols/cms/hw423/Data/Week14/Post_octest_{filecode}_{savecode}.npy', np.array(octest))
np.save(f'/vols/cms/hw423/Data/Week14/Post_test_loss_{filecode}_{savecode}.npy',np.array(test_loss_ia))
np.save(f'/vols/cms/hw423/Data/Week14/Post_train_loss_{filecode}_{savecode}.npy',np.array(train_loss_ia))