# package for predicting the results


def pred_save_dir(EPOCH,y_plus,var,target,normalized):
    from utils.toolbox import NameIt
    import os
    """
    return the path of saving pred result 
    """
    pred_path = "/storage3/yuning/thesis/pred"
    name = NameIt(y_plus,var,target,normalized)
    name = name + "_EPOCH="+str(EPOCH)
    save_path = os.path.join(pred_path,name)
    if os.path.exists(save_path) is False:
        print(f"Making Dir {save_path}")
        os.makedirs(save_path)

    return save_path

def Test_Eval(model,EPOCH,y_plus,var,target,normalized,device):
    import torch
    import os
    from torch.utils.data import DataLoader
    from utils.datas import slice_dir
    from utils.toolbox import NameIt
    from utils.metrics import Glob_error,RMS_error,Fluct_error
    from tqdm import tqdm
    import numpy as np
    """
    Evaluate the result of model
    input: 
        model: the trained model
        EPOCH: trained epoch corresponding to the model
        y_plus:value of wall distance
        var: list of input features
        target: list of targets
        normalized: boolean normalized or not 
        device: name of cuda device to use
    output:
        A dirct of all results
        Glob_error: np array of all glob error
        Rms_error: np array of all RMS error
        Fluct_error: np array of all Fluctuation error
        pred: np array of all prediction snapshots
    """
    model.eval()
    print("INFO: Test data evaluating !")
    root_path = "/storage3/yuning/thesis/tensor/"
    test_path = slice_dir(root_path,y_plus,var,target,"test",normalized)
    print(f"Data loaded from: \n {test_path}")
    test_dl = DataLoader(torch.load(test_path+"/test.pt"),batch_size=1,shuffle=True)
    RMSErrors = []
    GlobErrors = []
    FluctErrors = []
    preds = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(test_dl):
            x,y = batch
            x = x.float().to(device); y = y.float().squeeze().numpy()
            pred = model(x)
            pred = pred.float().squeeze().cpu().numpy()

            rms_error = RMS_error(pred,y)
            glb_error = Glob_error(pred,y)
            fluct_error = Fluct_error(pred,y)
        
            RMSErrors.append(rms_error)
            GlobErrors.append(glb_error)
            FluctErrors.append(fluct_error)
            preds.append(pred)
            targets.append(y)
    
    save_path = pred_save_dir(EPOCH,y_plus,var,target,normalized)
    
    Glob_error = np.array(GlobErrors)
    np.save(os.path.join(save_path,"glob.npy"),Glob_error)
    print(f"Glob error array saved, shape is {Glob_error.shape}")
    
    Rms_error = np.array(RMSErrors)
    np.save(os.path.join(save_path,"rms.npy"),Rms_error)
    print(f"Rms error array saved, shape is {Rms_error.shape}")
    
    
    Fluct_error = np.array(FluctErrors)
    np.save(os.path.join(save_path,"fluct.npy"),Fluct_error)
    print(f"Fluct error array saved, shape is {Fluct_error.shape}")

    Pred_array = np.array(preds)
    np.save(os.path.join(save_path,"pred.npy"),Pred_array)
    print(f"Prediction array saved, shape is {Pred_array.shape}")

    y_array = np.array(targets)
    np.save(os.path.join(save_path,"y.npy"),Pred_array)
    print(f"Ground Truth array saved, shape is {y_array.shape}")
            














# def predict(model,y_plus,var,target,normalized=False):
    
    
    
#     file_path = slice_loc(y_plus,var,target,normalized=False)
#     path_test = os.path.join(file_path,"test")
#     print(path_test)
#     dataset = tf.data.TFRecordDataset(
#                                         filenames=path_test,
#                                         compression_type="GZIP",
#                                         num_parallel_reads=tf.data.experimental.AUTOTUNE
#                                         )

#     feature_dict = feature_description(file_path)
    
#     TARGET = []
#     INPUTS = []
#     for snap in dataset:
        
#         (dict_for_dataset,target_array) = read_tfrecords(snap,feature_dict,target)
#         inputs = [ np.expand_dims(item,0) for item in dict_for_dataset.values() ]
#         names = dict_for_dataset.keys()
       
#         pr = inputs[0]
#         inputs.pop(0)
#         inputs.append(pr)
        
#         INPUTS.append(inputs)
#         TARGET.append(target_array)

#     print("Totally {} test snapshots".format(len(TARGET)))

#     PRED = []
#     for input in tqdm (INPUTS):
#             pred_pr = model.predict(input,verbose= 0)
#             PRED.append(pred_pr)

#     pred_array = np.array(PRED)
#     preds = np.squeeze(pred_array)
#     targets = np.array(TARGET)
#     if preds.shape == targets.shape:
#         print("targets and prediction shape are matched!")

#     return (preds,targets)