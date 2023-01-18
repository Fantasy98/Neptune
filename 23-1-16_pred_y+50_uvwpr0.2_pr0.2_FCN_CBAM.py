import torch 
import os
from utils.prediction import Test_Eval,Grad_Eval,pred_save_dir
from utils.plots import Plot_2D_snapshots,PSD_single,Loss_Plot,Scatter_Plot,RSE_Surface,Grad_Surface,Save_Plot_dir
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

checkpoint = torch.load("/storage3/yuning/thesis/models/2023-01-16/y_plus_50-VARS-pr0.2_u_vel_v_vel_w_vel-TARGETS-pr0.2_flux_CBAM2_EPOCH=99.pt")
model = checkpoint["model"]
model.to(device)
print("Model checkpoint loaded")

var=['u_vel',"v_vel","w_vel","pr0.2"]
target=['pr0.2_flux']
normalized=False
y_plus=50
EPOCH = 100
model_name="CBAM2"
Test_Eval(model,EPOCH,y_plus,var,target,normalized,device,model_name=model_name)

pred_dir = pred_save_dir(EPOCH,y_plus,var,target,normalized,model_name=model_name)

train_loss = checkpoint["loss"]
val_loss = checkpoint["val_loss"]
fig_dir = Save_Plot_dir(EPOCH,y_plus,var,target,normalized,model_name=model_name)
loss_fig  = os.path.join(fig_dir,"Loss")
Loss_Plot(train_loss,val_loss,loss_fig)

glob_error = np.load(os.path.join(pred_dir,"glob.npy"))
rms_error = np.load(os.path.join(pred_dir,"rms.npy"))
fluct_error = np.load(os.path.join(pred_dir,"fluct.npy"))
root_squred_error = np.load(os.path.join(pred_dir,"ers.npy"))


Scatter_Plot(glob_error,rms_error,fluct_error,
               EPOCH,y_plus,var,target,normalized,model_name)

preds_array = np.load(os.path.join(pred_dir,"pred.npy"))
target_array=np.load(os.path.join(pred_dir,"y.npy"))

r,p = stats.pearsonr(preds_array.flatten(),target_array.flatten())
print(f"The PCR is {r}")
PSD_single(target_array,preds_array,os.path.join(fig_dir,"PSD"))

pred_mean = np.mean(preds_array,axis=0)
target_mean = np.mean(target_array,axis=0)


Plot_2D_snapshots(pred_mean,os.path.join(fig_dir,"pred_avg"))
Plot_2D_snapshots(target_mean,os.path.join(fig_dir,"target_avg"))


pred_fluct = pred_mean - np.mean(preds_array)
target_fluct = target_mean - np.mean(target_array)
Plot_2D_snapshots(pred_fluct,os.path.join(fig_dir,"pred_fluct_avg"))
Plot_2D_snapshots(target_fluct,os.path.join(fig_dir,"target_fluct_avg"))


snap_pred = preds_array[0,:,:]
snap_target = target_array[0,:,:]
rms_diff = 100*np.abs(snap_pred - snap_target)/np.mean(target_array)
Plot_2D_snapshots(snap_pred,os.path.join(fig_dir,"pred_snap"))
Plot_2D_snapshots(snap_target,os.path.join(fig_dir,"target_snap"))
Plot_2D_snapshots(rms_diff,os.path.join(fig_dir,"diff_snap"))

RSE_Surface(np.mean(root_squred_error,axis=0),os.path.join(fig_dir,"root_sqrt_surf"))

from utils.metrics import ERS
RSE_Surface(ERS(pred_mean,target_mean),os.path.join(fig_dir,"root_sqrt_surf_mean"))
from utils.metrics import ERS
RSE_Surface(ERS(pred_mean,target_mean),os.path.join(fig_dir,"root_sqrt_surf_mean"))

from utils.prediction import Grad_Eval
from utils.plots import Grad_Surface

Grad_Eval(model,EPOCH,y_plus,var,target,normalized,device,model_name=model_name)
grad = np.load(os.path.join(pred_dir,"grad.npy"))
for i, var in enumerate(vars):
    Grad_Surface(grad_array=grad.mean(0)[i,:,:],save_dir=os.path.join(fig_dir,var+"grad"))