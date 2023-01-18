import matplotlib.pyplot as plt
import numpy as np 

def Save_Plot_dir(EPOCH,y_plus,var,target,normalized,model_name:None):
    from utils.toolbox import NameIt
    import os
    """
    return the path of saving pred result 
    """
    pred_path = "/storage3/yuning/thesis/fig"
    name = NameIt(y_plus,var,target,normalized)
    name = name + "_" + model_name + "_EPOCH="+str(EPOCH)
    save_path = os.path.join(pred_path,name)
    if os.path.exists(save_path) is False:
        print(f"Making Dir {save_path}")
        os.makedirs(save_path)

    return save_path

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
        plt.figure(4,figsize=(20*cm,15*cm),dpi=500)
        clb=plt.contourf(xx, yy, np.transpose(avg), cmap='jet', edgecolor='none')
        plt.colorbar(clb)
        plt.xlabel(r'$x^+$',fontdict={"size":15})
        plt.ylabel(r'$z^+$',fontdict={"size":15})
        plt.xticks(placement_x)
        plt.yticks(placement_z)
        plt.savefig(save_dir,bbox_inches="tight")
        plt.clf()


def PSD_single(y,pred,save_dir):
    import numpy as np 
    import matplotlib.pyplot as plt
    import matplotlib        as mpl
    Nx = 256 ; Nz  = 256 ;Lx  = 12 ;Lz  = 6
    # dx=Lx/Nx ;dz=Lz/Nz
    x_range=np.linspace(1,Nx,Nx)
    z_range=np.linspace(1,Nz,Nz)
    # x=dx*x_range;z=dz*z_range;[xx,zz]=np.meshgrid(x,z)
    dkx = 2*np.pi/Lx
    dkz = 2*np.pi/Lz
    kx = dkx * np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
    kz = dkz * np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])
    [kkx,kkz]=np.meshgrid(kx,kz)
    kkx_norm= np.sqrt(kkx**2)
    kkz_norm = np.sqrt(kkz**2)

    Re_Tau = 395 #Direct from simulation
    Re = 10400 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu

    # calculating wavelength in plus units 
    Lambda_x = (2*np.pi/kkx_norm)*u_tau/nu
    Lambda_z = (2*np.pi/kkz_norm)*u_tau/nu

    Theta_fluc_targ=y-np.mean(y)
    Theta_fluc_pred=pred-np.mean(pred)

    fourier_image_targ = np.fft.fftn(Theta_fluc_targ)
    fourier_image_pred = np.fft.fftn(Theta_fluc_pred)

    fourier_amplitudes_targ = np.mean(np.abs(fourier_image_targ)**2,axis=0)
    fourier_amplitudes_pred = np.mean(np.abs(fourier_image_pred)**2,axis=0)

    pct10=0.1*np.max(fourier_amplitudes_targ*kkx*kkz)
    pct50=0.5*np.max(fourier_amplitudes_targ*kkx*kkz)
    pct90=0.9*np.max(fourier_amplitudes_targ*kkx*kkz)
    pct100=np.max(fourier_amplitudes_targ*kkx*kkz)

    cmap = mpl.cm.Greys(np.linspace(0,1,20))
    cmap = mpl.colors.ListedColormap(cmap[5:,:-1])
    fig,ax=plt.subplots(1,1,dpi=1000)
    CP=plt.contourf(Lambda_x,Lambda_z,fourier_amplitudes_targ*kkx*kkz,[pct10,pct50,pct90,pct100],cmap=cmap)
    CS=plt.contour(Lambda_x,Lambda_z,fourier_amplitudes_pred*kkx*kkz,[pct10,pct50,pct90,pct100],colors='orange',linestyles='solid')
    plt.xscale('log')
    plt.yscale('log')
    ax.set_ylabel(r'$\lambda_{z}^+$')
    ax.set_xlabel(r'$\lambda_{x}^+$')
    ax.set_title(r'$k_x\ k_z\ \phi_{q_w}$')
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

def Scatter_Plot(glob_error,rms_error,fluct_error,EPOCH,y_plus,var,target,normalized,model_name):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.toolbox import NameIt
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
    fig_dir = Save_Plot_dir(EPOCH,y_plus,var,target,normalized,model_name)
    scatter_fig = os.path.join(fig_dir,"Error Scatter")
    name = NameIt(y_plus,var,target,normalized)
    font_dict = {"fontsize":17}
    plt.figure(31,figsize=(16,9))
    plt.plot(glob_error,"rx",markersize= 5,label="Glob Error = {:.2f}%".format(mean_glob))
    plt.plot(rms_error,"bo",markersize= 5,label="RMS Error = {:.2f}%".format(mean_rms))
    plt.plot(fluct_error,"gs",markersize= 5,label="Fluct Error = {:.2f}%".format(mean_fluct))
    plt.title(name,fontdict=font_dict)
    plt.xlabel("Snapshots",fontdict=font_dict)
    plt.ylabel("Error (%)",fontdict=font_dict)
    plt.grid()
    plt.legend(fontsize=18)
    plt.savefig(scatter_fig,bbox_inches='tight')


def RSE_Surface(res_array,save_dir):
    """
    Plot the Root squared error 3D surface
    Input: 
        res_array: np arrary of the root squared error 
        shape= 256*256
        save_dir: dir to save the fig
    """
    from matplotlib import cbook
    from matplotlib import cm
    from matplotlib.colors import LightSource
    import matplotlib.pyplot as plt
    import numpy as np

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

    cms = 1/2.54
    ig = plt.figure(6,figsize=(15*cms,10*cms),dpi=500)
    ax = plt.axes(projection="3d")
    mappable = cm.ScalarMappable(cmap=cm.jet)
    mappable.set_array(res_array)

    ls = LightSource(270, 45)
    surf = ax.plot_surface(xx, yy, res_array, rstride=1, cstride=1,cmap=mappable.cmap,
                        linewidth=0, antialiased=False, shade=False)
    plt.colorbar(surf,pad = 0.18)
    plt.tight_layout()
    ax.set_xlabel(r'$x^+$',labelpad=10)
    ax.set_ylabel(r'$z^+$',labelpad=5)
    ax.set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
    ax.set_xticks(placement_x)
    ax.set_xticklabels(axis_range_x)
    ax.set_yticks(placement_z)
    ax.set_yticklabels(axis_range_z)
    ax.set_box_aspect((2,1,1))
    ax.view_init(25, -70)
    plt.savefig(save_dir,bbox_inches="tight")
    plt.clf()

def Grad_Surface(grad_array,save_dir):
    """
    Plot the gradient 3D surface
    Input: 
        res_array: np arrary of gradient of varible w.r.t target
        shape= 256*256
        save_dir: dir to save the fig
    """
    from matplotlib import cbook
    from matplotlib import cm
    from matplotlib.colors import LightSource
    import matplotlib.pyplot as plt
    import numpy as np
    #Regularize the array
    z = (grad_array-grad_array.min())/(grad_array.max()-grad_array.min())
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

    cms = 1/2.54
    ig = plt.figure(6,figsize=(15*cms,10*cms),dpi=500)
    ax = plt.axes(projection="3d")
    mappable = cm.ScalarMappable(cmap=cm.jet)
    mappable.set_array(z)

    ls = LightSource(270, 45)
    surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1,cmap=mappable.cmap,
                        linewidth=0, antialiased=False, shade=False)
    plt.colorbar(surf,pad = 0.18)
    plt.tight_layout()
    ax.set_xlabel(r'$x^+$',labelpad=10)
    ax.set_ylabel(r'$z^+$',labelpad=5)
    # ax.set_zlabel(r'$E_{RS}\ [\%]$',labelpad=0)
    ax.set_xticks(placement_x)
    ax.set_xticklabels(axis_range_x)
    ax.set_yticks(placement_z)
    ax.set_yticklabels(axis_range_z)
    ax.set_box_aspect((2,1,1))
    ax.view_init(30, 140)
    plt.savefig(save_dir,bbox_inches="tight")
    plt.clf()