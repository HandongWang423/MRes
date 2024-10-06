import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

import mplhep as hep
hep.style.use("CMS")
import os

torch.set_default_dtype(torch.float64)


## Initialisation and Importing 
filecode = 'InfA_xgb_AJF_x2'
occode = 'InfA_RD_DPrmvd'
# filecode = f'{occode}_x6'
oc = np.load(f'/vols/cms/hw423/Data/Week14/Post_octest_{filecode}.npy')
Lb = pd.read_pickle('/vols/cms/hw423/Data/Week14/Label.pkl')
dfy = pd.get_dummies(Lb)
dfw = pd.read_pickle('/vols/cms/hw423/Data/Week14/weight.pkl')
true = torch.tensor(np.array(dfy).astype(int)).to(torch.float64)
labels = torch.tensor(np.array(Lb)).to(torch.float64)
OC = torch.tensor(oc).to(torch.float64)

oc_ce = np.load(f'/vols/cms/hw423/Data/Week14/octest_{occode}.npy')
OC_ce = torch.tensor(oc_ce).to(torch.float64)

weight = torch.tensor(np.array(dfw)).to(torch.float64)
test_loss = np.load(f'/vols/cms/hw423/Data/Week14/Post_test_loss_{filecode}.npy')
train_loss = np.load(f'/vols/cms/hw423/Data/Week14/Post_train_loss_{filecode}.npy')

folder_name = f"/vols/cms/hw423/Week14/plots/{filecode}"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"folder'{folder_name}' has made")
else:
    print(f"folder '{folder_name}' already exists")

col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']


###Loss function
plt.plot(0.907*test_loss/1e4,label='Test')
plt.plot(train_loss/1e4,label = 'Train')
plt.ylabel(r'Loss(1e4)')
plt.xlabel('Epochs')
plt.tight_layout()
plt.legend()
plt.savefig(f'{folder_name}/Loss.pdf')
plt.show()

## Prepare for Confusion_matrix
label = torch.tensor(np.array(dfy),dtype = torch.float64)
def smooth_argmax(tensor, dim=-1, temperature=1.0):
    softmax_tensor =F.softmax(tensor / temperature, dim=dim)
    return softmax_tensor
def confusion_matrix(OC,label,weight):
    label_w = 138000*weight.unsqueeze(1)*label
    pred_matrix = smooth_argmax(OC,temperature=.0000001,dim=1) 
    confusion_matrix = torch.matmul(pred_matrix.t(),label_w)[1:,:]
    return confusion_matrix

def nll(theta1,cm):
    O = torch.sum(cm,dim=1)
    theta0 = torch.ones(1)
    theta = torch.cat([theta0,theta1])
    return -(O@(torch.log(cm@theta))-torch.sum((cm@theta)))

# Confusion matrix
cm_ia = confusion_matrix(OC,label,weight)
cm_ia = torch.tensor(cm_ia,dtype=torch.float64)
cm_ce = confusion_matrix(OC_ce,label,weight)
cm_ce = torch.tensor(cm_ce,dtype=torch.float64)

def nll_prof(theta1,cm,i,val):
    O = torch.sum(cm,dim=1)
    val =torch.tensor(val,dtype=theta1.dtype)
    theta0 = torch.ones(1)
    theta = torch.cat([theta0,theta1])
    theta[i+1] = val
    return -(O@(torch.log(cm@theta))-torch.sum((cm@theta)))

def LL_prof(cm,j,a,b,n,steps=500):
    xspace=np.linspace(a,b,n)
    likelihood_prf = torch.tensor([]).view(-1)
    for i in xspace:
        theta = torch.ones(6,requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=0.001)
        for step in range(steps):
            optimizer.zero_grad()
            loss = nll_prof(theta,cm,j,i)
            loss.backward()
            optimizer.step()
        likelihood_prf=torch.cat([likelihood_prf,loss.view(-1)])
    lkpf = likelihood_prf.detach().numpy()
    return lkpf


col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues, filename='confusion_matrix',**kwargs):
    fig, ax = plt.subplots(figsize=(10,8))
    plot_params = {k: v for k, v in kwargs.items()}
    ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap,**plot_params)  # Using '.2f' for floating point format
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    ax.set_xticklabels(col[1:])
    ax.set_yticklabels(col[1:])
    fig.savefig(f'{folder_name}/{filename}.pdf', bbox_inches='tight')



    
print('PLOTS')


## IA cm
cm_w = pd.DataFrame(cm_ia[:,1:].detach().numpy()).T
CM = []
for i in range(6):
    cl = cm_w.iloc[:,i]/cm_w.iloc[:,i].sum()
    CM.append(cl)
cm = pd.DataFrame(CM).T
plot_confusion_matrix(cm, cmap=plt.cm.Oranges)


##XGB/CE cm
cm_w = pd.DataFrame(cm_ce[:,1:].detach().numpy()).T
CM = []
for i in range(6):
    cl = cm_w.iloc[:,i]/cm_w.iloc[:,i].sum()
    CM.append(cl)
cm = pd.DataFrame(CM).T
plot_confusion_matrix(cm, cmap=plt.cm.Oranges)

## Correlation 
def hess_to_tensor(H):
    hess_elements = []
    for i in range(len(H)):
        for j in range(len(H)):
            hess_elements.append(H[i][j].reshape(1))
    return torch.cat(hess_elements).reshape(len(H),len(H))
hess = torch.func.hessian(nll,0)(torch.ones(6),cm_ia)
H = hess_to_tensor(hess)
I = torch.inverse(H)
sigma = torch.diag(I)**0.5

Cor = torch.zeros(6,6)
for i in range(6):
    for j in range(6):
        Cor[i,j] = I[i,j]/(sigma[i]*sigma[j])
matrix = np.tril(Cor)

plot_confusion_matrix(Cor,cmap='bwr',title='Correlation Matrix',filename='Correlation',vmax=1,center=0,vmin=-1,mask = (matrix==False))

## Likelihoods
ratio = [0.1,1,2,10,2,100]
fig, axes = plt.subplots(2, 3, figsize=(30,20))
axes = axes.flatten()
for i in range(6):
    print(f'Begin to Plot {col[i+1]}')
    theta = torch.ones(6)
    thetas =  np.linspace(min(0.0,1-ratio[i]*0.5),ratio[i]*.5,50)
    theta_val = np.copy(theta[0])
    likelihood_ia = torch.tensor([])
    likelihood_ce = torch.tensor([])
    for j in thetas:
        theta[i] = theta_val+j
        nlls = nll(theta,cm_ia).view(-1)
        nlls_ce = nll(theta,cm_ce).view(-1)
        likelihood_ia = torch.cat([likelihood_ia,nlls])
        likelihood_ce = torch.cat([likelihood_ce,nlls_ce])
    lkpf_ia = LL_prof(cm_ia,i,min(0.0,1-ratio[i]*0.5),1+ratio[i]*.5,50,steps=1000)
    lkpf_ce = LL_prof(cm_ce,i,min(0.0,1-ratio[i]*0.5),1+ratio[i]*.5,50,steps=1000)
    axes[i].plot(thetas,likelihood_ia-likelihood_ia.min(), color='red',linestyle='--', alpha=0.7,label = 'IA scan')
    axes[i].plot(thetas,likelihood_ce-likelihood_ce.min(), color='blue',linestyle='--',alpha=0.7,label = 'XGB scan')
    axes[i].plot(thetas,lkpf_ia-lkpf_ia.min(),color = 'red',label = 'IA profiled')
    axes[i].plot(thetas,lkpf_ce-lkpf_ce.min(),color = 'blue',label = 'XGB profiled')
    axes[i].set_title(f'{col[i+1]}')
    axes[i].set_xlabel(f'$\mu_{i+1}$')
    axes[i].set_ylabel('2$\Delta$NLL')
    axes[i].legend()
# Adjust layout
plt.suptitle('The 2$\Delta$NLL plot for 6 signal strength parameter')
plt.tight_layout()
plt.savefig(f'{folder_name}/likelihood.pdf')




