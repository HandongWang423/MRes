import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import mplhep as hep
hep.style.use("CMS")
import os
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('png', 'pdf', 'svg')
set_matplotlib_formats('retina')
plt.rcParams['figure.dpi'] = 70     
plt.rcParams.update({'font.size': 24})

folder_name= os.getcwd()

col = ['$\gamma\gamma$','ggH','qqH','WH','ZH','ttH','tH']

torch.set_default_dtype(torch.float64)
def nll(theta1,cm):
    O = torch.sum(cm,dim=1)
    theta0 = torch.ones(1)
    theta = torch.cat([theta0,theta1])
    return -(O@(torch.log(cm@theta))-torch.sum((cm@theta)))
def nll_prof(theta1,cm,i,val):
    O = torch.sum(cm,dim=1)
    val =torch.tensor(val,dtype=theta1.dtype)
    theta0 = torch.ones(1)
    theta = torch.cat([theta0,theta1])
    theta[i+1] = val
    return -(O@(torch.log(cm@theta))-torch.sum((cm@theta)))
def LL_prof(cm,i,a,b,n,steps=500,lr = 0.001):
    xspace=np.linspace(a,b,n)
    likelihood_prf = torch.zeros(len(xspace),dtype=torch.float64)
    Theta = torch.zeros(6,n)
    for j,val in enumerate(xspace):
        theta = torch.ones(6,requires_grad=True)
        # theta[i]=val
        optimizer = torch.optim.Adam([theta], lr=lr)
        for step in range(steps):
            optimizer.zero_grad()
            loss = nll_prof(theta,cm,i,val)
            loss.backward()
            optimizer.step()
        likelihood_prf[j] = loss.item()
        Theta[:,j] = theta
    lkpf = likelihood_prf.detach().numpy()
    return lkpf,Theta.detach().numpy()


##画图##
def nll_prof_2(theta1,cm,j1,j2,val1,val2):
    O = torch.sum(cm,dim=1)
    val1 =torch.tensor(val1,dtype=theta1.dtype)
    val2 =torch.tensor(val2,dtype=theta1.dtype)
    theta0 = torch.ones(1)
    theta = torch.cat([theta0,theta1])
    theta[j1] = val1
    theta[j2] = val2
    return -(O@(torch.log(cm@theta))-torch.sum((cm@theta)))
def LL_prof_2(cm,j1,j2,x,y,n,steps=100):
    a,b = x[0],x[1]
    c,d = y[0],y[1]
    x_range=torch.linspace(a,b,n)
    y_range=torch.linspace(c,d,n)
    x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='ij')
    likelihood_prf = torch.zeros(x_grid.size())
    with tqdm(range(n)) as t:
        for i1 in t:
            for i2 in range(n):
                theta = torch.ones(6,requires_grad=True)
                optimizer = torch.optim.Adam([theta], lr=0.001)
                for step in range(steps):
                    optimizer.zero_grad()
                    loss = nll_prof_2(theta,cm,j1,j2,x_grid[i1,i2],y_grid[i1,i2])
                    loss.backward()
                    optimizer.step()
                likelihood_prf[i1,i2]=loss.item()
        lkpf = likelihood_prf.detach().numpy()
       
    return lkpf
def ContFields(i,j,cm,x,y,res):
    LP= LL_prof_2(cm,i,j,x,y,res,steps=300)
    deltLP = LP-LP.min()
    return deltLP

    
def PlotContG(i,j,x_ia,y_ia,x_new,y_new,x_xgb,y_xgb,res,deltLP56,deltLP56_xgb,deltLP56_new,xlim=None,ylim=None):
    x_range_ia = torch.linspace(x_ia[0],x_ia[1],res)
    y_range_ia = torch.linspace(y_ia[0],y_ia[1],res)
    x_range_xgb = torch.linspace(x_xgb[0],x_xgb[1],res)
    y_range_xgb = torch.linspace(y_xgb[0],y_xgb[1],res)
    x_range_new = torch.linspace(x_new[0],x_new[1],res)
    y_range_new = torch.linspace(y_new[0],y_new[1],res)
    

    x_grid_ia,y_grid_ia = torch.meshgrid(x_range_ia, y_range_ia, indexing='ij')
    x_grid_xgb,y_grid_xgb = torch.meshgrid(x_range_xgb, y_range_xgb, indexing='ij')
    x_grid_new,y_grid_new = torch.meshgrid(x_range_new, y_range_new, indexing='ij')
    # level=[1,2,4,90]
    level = [2.28,6.18,140]
    linestyles = ['-', ':','-.']
    
    alpha=[1,0.5,0.3,0.01]
    plt.figure(figsize=(6, 8))
    IA_c = plt.contour(x_grid_ia.numpy(), y_grid_ia.numpy(), deltLP56, levels=level,cmap = 'autumn',labels = 'IA',linestyles=linestyles,)
    XGB_c = plt.contour(x_grid_xgb.numpy(), y_grid_xgb.numpy(), deltLP56_xgb, levels=level,cmap = 'binary_r',labels = 'XGB',linestyles=linestyles)
    New_c = plt.contour(x_grid_new.numpy(), y_grid_new.numpy(), deltLP56_new, levels=level,cmap = 'winter',labels = 'Bayesian',linestyles=linestyles)
    # for contour in [IA_c,XGB_c,New_c]:
    #         for i, collection in enumerate(contour.collections):
    #             collection.set_alpha(alpha[i])
    plt.clabel(IA_c, inline=True, fontsize=8)
    plt.clabel(XGB_c, inline=True, fontsize=8)
    plt.clabel(New_c, inline=True, fontsize=8)
    plt.plot(1,1,color = 'red',label='IA')
    plt.plot(1,1,color = 'blue',label='Bayesian')
    plt.plot(1,1,color = 'black',label='XGB')
    
    plt.scatter(1,1,color = 'black',label='SM')
    plt.xlabel(f'{col[i]}')
    plt.ylabel(f'{col[j]}')
    plt.xlim(xlim)
    plt.ylim(ylim)
    # plt.legend()
    plt.savefig(f'{folder_name}/Contour_{i}{j}.pdf',bbox_inches='tight')
    
cm_xgb = torch.load(f'{folder_name}/cm_xgb.pt')
cm_ia_xgb = torch.load(f'{folder_name}/cm_ia_xgb.pt')
cm_xgb_new = torch.load(f'{folder_name}/cm_xgb_new.pt')
def PlotContours(i,j,res):
    x_ia = IA_xy[i-1]
    y_ia = IA_xy[j-1]
    x_xgb = XGB_xy[i-1]
    y_xgb = XGB_xy[j-1]
    x_new = NEW_xy[i-1]
    y_new = NEW_xy[j-1]

    deltLP56_ia = ContFields(i,j,cm_ia_xgb,x_ia,y_ia,res)
    deltLP56_new = ContFields(i,j,cm_xgb_new,x_new,y_new,res)
    deltLP56_xgb = ContFields(i,j,cm_xgb,x_xgb,y_xgb,res)

    np.save(f'{folder_name}/deltLP{i}{j}_xgb_ia.npy',deltLP56_ia)
    np.save(f'{folder_name}/deltLP{i}{j}_xgb_new.npy',deltLP56_new)
    np.save(f'{folder_name}/deltLP{i}{j}_xgb.npy',deltLP56_xgb)
    deltLP56_ia = np.load(f'{folder_name}/deltLP{i}{j}_xgb_ia.npy')
    deltLP56_new = np.load(f'{folder_name}/deltLP{i}{j}_xgb_new.npy')
    deltLP56_xgb = np.load(f'{folder_name}/deltLP{i}{j}_xgb.npy')
    PlotContG(i,j,x_ia,y_ia,x_new,y_new,x_xgb,y_xgb,res,deltLP56_ia,deltLP56_xgb,deltLP56_new)
IA_xy = [[0.7,1.3],[0,2],[0,3],[0,5],[0,3],[0,12]]
XGB_xy = [[0.6,1.4],[0,2.5],[-1,3],[0,6],[-1,3],[0,30]]
NEW_xy = [[0.7,1.3],[0,2],[0,4],[0,8],[0,3],[0,10]]


for i in range(6):
    for j in range(6):
        print(i+1,j+1)
        if i<j:
            PlotContours(i+1,j+1,50)