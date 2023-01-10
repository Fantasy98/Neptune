import matplotlib.pyplot as plt
import numpy as np 
def Plot_2D_snapshots(avg,save_dir):
    if avg.shape != (256,256):
        
        print("Not valid for the function!")

    else:
        Re_Tau = 395 #Direct from simulation
        Re = 10400 #Direct from simulation
        nu = 1/Re #Kinematic viscosity
        u_tau = Re_Tau*nu

        xx, yy = np.mgrid[0:256:256j, 0:256:256j]


        x_range=12
        z_range=6

        gridpoints_x=int(255)+1
        gridponts_z=int(255)+1


        x_plus_max=x_range*u_tau/nu
        z_plus_max=z_range*u_tau/nu


        x_plus_max=np.round(x_plus_max).astype(int)
        z_plus_max=np.round(z_plus_max).astype(int)

        axis_range_x=np.array([0,950,1900,2850,3980,4740])
        axis_range_z=np.array([0,470,950,1420,1900,2370])


        placement_x=axis_range_x*nu/u_tau
        placement_x=np.round((placement_x-0)/(x_range-0)*(gridpoints_x-0)).astype(int)

        placement_z=axis_range_z*nu/u_tau
        placement_z=np.round((placement_z-0)/(z_range-0)*(gridponts_z-0)).astype(int)

        cm =1/2.54
        plt.figure(figsize=(20*cm,15*cm),dpi=500)
        clb=plt.contourf(xx, yy, np.transpose(avg), cmap='jet', edgecolor='none')
        plt.colorbar(clb)
        plt.xlabel(r'$x^+$',fontdict={"size":15})
        plt.ylabel(r'$z^+$',fontdict={"size":15})
        plt.xticks(placement_x)
        plt.yticks(placement_z)
        plt.savefig(save_dir,bbox_inches="tight")


def Loss_Plot(train_loss,val_loss,save_dir):
    print("INFO: Ploting Loss vs Epoch")
    import matplotlib.pyplot as plt 
    font_dict = {"fontsize":16}
    plt.figure(32,figsize=(16,9))
    plt.semilogy(train_loss,"r",lw=2.5,label="Train Loss")
    plt.semilogy(val_loss,"b",lw=2.5,label="Validation Loss")
    plt.grid()
    plt.xlabel("Epoch",fontdict=font_dict)
    plt.ylabel("MSE Loss",fontdict=font_dict)
    plt.legend(fontsize = 18    )
    plt.savefig(save_dir,bbox_inches="tight")

def Scatter_Plot(glob_error,rms_error,fluct_error,EPOCH,y_plus,var,target,normalized):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    """
    Plot scatter of all loss for each snap shot
    input: glob_error, rms_error,fluct_error should be numpy arrary for each error
    
    """
    mean_glob = np.mean(glob_error)
    mean_rms = np.mean(rms_error)
    mean_fluct = np.mean(fluct_error)
    print(f"The mean glob error is {mean_glob}")
    print(f"The mean rms error is {mean_rms}")
    print(f"The mean fluct error is {mean_fluct}")
    fig_dir = Save_Plot_dir(EPOCH,y_plus,var,target,normalized)
    scatter_fig = os.path.join(fig_dir,"Error Scatter")

    font_dict = {"fontsize":16}
    plt.figure(31,figsize=(16,9))
    plt.plot(glob_error,"rx",markersize= 5,label="Glob Error = {:.2f}%".format(mean_glob))
    plt.plot(rms_error,"bo",markersize= 5,label="RMS Error = {:.2f}%".format(mean_rms))
    plt.plot(fluct_error,"gs",markersize= 5,label="Fluct Error = {:.2f}%".format(mean_fluct))
    plt.xlabel("Snapshots",fontdict=font_dict)
    plt.ylabel("Error",fontdict=font_dict)
    plt.grid()
    plt.legend(fontsize=18)
    plt.savefig(scatter_fig,bbox_inches='tight')

def Save_Plot_dir(EPOCH,y_plus,var,target,normalized):
    from utils.toolbox import NameIt
    import os
    """
    return the path of saving pred result 
    """
    pred_path = "/storage3/yuning/thesis/fig"
    name = NameIt(y_plus,var,target,normalized)
    name = name + "_EPOCH="+str(EPOCH)
    save_path = os.path.join(pred_path,name)
    if os.path.exists(save_path) is False:
        print(f"Making Dir {save_path}")
        os.makedirs(save_path)

    return save_path