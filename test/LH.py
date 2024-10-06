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
def Likelihoods2(cm,cm_c,cm_n,ratio,title1='XGB',title2='IA',title3='Bayesian',num=100,xlim=-10,limx = [0,1,2,3,4,5,6],steps=140,lr=0.03,plim=[-3,3],color=['b','r']):
    fs = (20,20)
    color1,color2=color
    f = plt.figure(figsize=fs)
    gs0 = gridspec.GridSpec(2,3,figure=f,wspace=0.2, hspace=0.3)
    handles=[]
    labels=[]
    for i in range(6):
        xlimt = 0
        if i in limx:
            xlimt=xlim
        print(f'Begin to Plot {col[i+1]}')
        theta = torch.ones(6)
        thetam =  np.linspace(max(-1,-ratio[i]*0.5),ratio[i]*.5,num)
        thetac =  np.linspace(max(-10,-ratio[i]*0.5),ratio[i]*.5,num)
        thetas =  np.linspace(max(xlimt-1,-ratio[i]*0.5),ratio[i]*.5,num)
        theta_val = np.copy(theta[0])
        likelihood_ia = torch.tensor([])
        likelihood_ce = torch.tensor([])
        likelihood_n = torch.tensor([])
        for j in thetam:
            theta[i] = theta_val+j
            nlls = nll(theta,cm).view(-1)
            nlls_ce = nll(theta,cm_c).view(-1)
            nlls_n = nll(theta,cm_n).view(-1)
            likelihood_ia = torch.cat([likelihood_ia,nlls])
            likelihood_ce = torch.cat([likelihood_ce,nlls_ce])
            likelihood_n = torch.cat([likelihood_n,nlls_n])
        lkpf_ia,Theta_ia = LL_prof(cm,i,max(-9,1-ratio[i]*0.5),1+ratio[i]*.5,num,steps=steps,lr=lr)
        lkpf_ce,Theta_ce = LL_prof(cm_c,i,max(xlimt,1-ratio[i]*0.5),1+ratio[i]*.5,num,steps=steps,lr=lr)
        lkpf_n,Theta_n = LL_prof(cm_n,i,max(xlimt,1-ratio[i]*0.5),1+ratio[i]*.5,num,steps=steps,lr=lr)
        
        # 
        ax1 = f.add_subplot(gs0[i])
        ax1.plot(thetam+1,likelihood_ia-likelihood_ia.min(), 
                 color='black',linestyle='--', alpha=0.7,label = f'{title1} Scan')
        ax1.plot(thetam+1,likelihood_n-likelihood_n.min(),
                 color='blue',linestyle='--',alpha=0.7,label = f'{title3} scan')
        ax1.plot(thetam+1,likelihood_ce-likelihood_ce.min(), 
                 color='r',linestyle='--',alpha=0.7,label = f'{title2} scan')
        
        ax1.plot(thetac+1,lkpf_ia-lkpf_ia.min(),color = 'black',label = f'{title1} Profiled')
        ax1.plot(thetas+1,lkpf_n-lkpf_n.min(),color = color1,label = f'{title3} profiled')
        ax1.plot(thetas+1,lkpf_ce-lkpf_ce.min(),color = color2,label = f'{title2} profiled')
        
        xmax = thetac[(lkpf_ia-lkpf_ia.min())<1].max()+1
        xmin = thetac[(lkpf_ia-lkpf_ia.min())<1].min()+1
        xmax1 = thetas[(lkpf_ce-lkpf_ce.min())<1].max()+1
        xmin1 = thetas[(lkpf_ce-lkpf_ce.min())<1].min()+1
        xmax2 = thetas[(lkpf_n-lkpf_n.min())<1].max()+1
        xmin2 = thetas[(lkpf_n-lkpf_n.min())<1].min()+1
        ax1.axvspan(xmin, xmax, hatch ='..', facecolor='none', edgecolor='grey', label=f'$<1\sigma$')
        ax1.axvspan(xmin2, xmax2, color=color1, alpha=0.05, label=f'$<1\sigma$-Bayesian')
        ax1.axvspan(xmin1, xmax1, color=color2, alpha=0.1, label=f'$<1\sigma$-IA')
        
        ax1.text(1, 8.5, fr'$\mu_{{{i+1}}}=1^{{+{xmax-1:.2f}}}_{{{xmin-1:.2f}}}$',
         horizontalalignment='center', verticalalignment='center', fontsize=24, color='black')
        ax1.text(1, 9.5, fr'$\mu_{{{i+1}}}=1^{{+{xmax1-1:.2f}}}_{{{xmin1-1:.2f}}}$',
         horizontalalignment='center', verticalalignment='center', fontsize=24, color=color2)
        ax1.text(1, 9, fr'$\mu_{{{i+1}}}=1^{{+{xmax2-1:.2f}}}_{{{xmin2-1:.2f}}}$',
         horizontalalignment='center', verticalalignment='center', fontsize=24, color=color1)
        ax1.set_ylim(0,8)
        ax1.set_xlim(max(min(thetac+1),1+5*(xmin1-1)),min(max(thetac+1),1+5*(xmax1-1)))
        ax1.axhline(y=1,color='black',linestyle='-')
        ax1.axhline(y=4,color='black',linestyle='--')
        ax1.axvline(x=1,color='grey',linestyle='-')
        ax1.set_title(f'{col[i+1]}',loc='left',fontweight='bold')
        ax1.set_ylabel('2$\Delta$NLL')
        ax1.set_xlabel(fr'$\mu_{{{i+1}}}$')

        # ax1.legend()
        handles1, labels1 = ax1.get_legend_handles_labels() 
        # handles.append(handles2)


        # plt.setp(ax1.get_xticklabels(), visible=False)
        print(Theta_ia.max()) 
    f.legend(handles1, labels1, loc='upper right', bbox_to_anchor=(1.05, 0.87), 
            fancybox=True,edgecolor='black',framealpha=True,borderaxespad=0.)
    f.savefig(f'{folder_name}/LH_{title1}_{title2}_CM.pdf',bbox_inches='tight')
    
    




cm_xgb = torch.load(f'{folder_name}/cm_xgb.pt')
cm_ia_xgb = torch.load(f'{folder_name}/cm_ia_xgb.pt')
cm_xgb_new = torch.load(f'{folder_name}/cm_xgb_new.pt')
    
    
    

#######
Likelihoods2(cm_xgb,cm_ia_xgb,cm_xgb_new,ratio = [0.4,3,3,6,3,40],
             title1='XGB',title2='IA',title3='Bayesian',
             num=20,xlim=-0.4,limx=[5],color=['b','r'],steps=40)
#num是采样率，200比较大，可以适当调整。xlim和limx不用管，color是两个除了黑色曲线的颜色
########