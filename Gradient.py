import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


filecode = 'InfA_RD_DPrmvd'
oc = np.load(f'/vols/cms/hw423/Data/Week14/octest_{filecode}.npy')
Lb = pd.read_pickle('/vols/cms/hw423/Data/Week14/Label.pkl')
dfy = pd.get_dummies(Lb)
dfw = pd.read_pickle('/vols/cms/hw423/Data/Week14/weight.pkl')
true = torch.tensor(np.array(dfy).astype(int))

labels = torch.tensor(np.array(Lb))
OC = torch.tensor(oc)
weight = torch.tensor(np.array(dfw))



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
        
        
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.copy_(torch.eye(7))
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     print('INITIALED')


    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        return out
        # out = out / self.temperature
        # return torch.softmax(out, dim=1)
        
    def set_temperature(self, temp):
        self.temperature = temp

model = Net()