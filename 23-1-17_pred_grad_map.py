import torch 
import os
from utils.prediction import Test_Eval,Grad_Eval,pred_save_dir
from utils.plots import Plot_2D_snapshots,PSD_single,Loss_Plot,Scatter_Plot,RSE_Surface,Grad_Surface,Save_Plot_dir
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from tqdm import tqdm
device = ("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

checkpoint = torch.load("/storage3/yuning/thesis/models/2023-01-10/y_plus_30-VARS-pr0.025_u_vel_v_vel_w_vel-TARGETS-pr0.025_flux_EPOCH=100.pt")
model = checkpoint["model"]
model.to(device)
print("Model checkpoint loaded")

var=['u_vel',"v_vel","w_vel","pr0.025"]
target=['pr0.025_flux']
normalized=False
y_plus=30
EPOCH = 100
model_name="FCN"

fig_dir = Save_Plot_dir(EPOCH,y_plus,var,target,normalized,model_name=model_name)
pred_dir = pred_save_dir(EPOCH,y_plus,var,target,normalized,model_name=model_name)

from utils.prediction import Grad_Eval
from utils.plots import Grad_Surface

Grad_Eval(model,EPOCH,y_plus,var,target,normalized,device,model_name=model_name)
grad = np.load(os.path.join(pred_dir,"grad.npy"))
var=['u_vel',"v_vel","w_vel","pr0025"]
for i, var in tqdm(enumerate(var)):
    Grad_Surface(grad_array=grad.mean(0)[i,:,:],save_dir=os.path.join(fig_dir,var+"_grad"))