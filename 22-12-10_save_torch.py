from utils.datas import TF2Torch,mkdataset
var=['u_vel',"v_vel","w_vel","pr0.2"]
target=['pr0.2_flux']
normalized=False
y_plus=50
save_types= ["train","test","validation"]

for save_type in save_types:
    print(f"Saving {save_type} data!")
    root_path = "/storage3/yuning/thesis/tensor/"
    TF2Torch(root_path,y_plus,var,target,save_type,normalized=normalized)

    mkdataset(root_path,y_plus,var,target,save_type,normalized=False)