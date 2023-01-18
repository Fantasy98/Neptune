import numpy as np

def RMS_error(y_pred,y_true):
  """
  Root Mean Square Error as metrics of prediction
  Input:
    y_pred: np.array with shape of 256,256
    y_ture: np.array with shape of 256,256
  output:
    error: RMS error scalar
  """
  if y_pred.shape == y_true.shape and len(y_pred.shape)==2:
    error = 100*np.sqrt( np.mean( (y_pred-y_true)**2 ) )/np.mean(y_true)
    return error
  else:
    print("Expected shape of (256,256)")

def Glob_error(y_pred,y_true):
  """
  Global Error as metrics of prediction
  Input:
    y_pred: np.array with shape of 256,256
    y_ture: np.array with shape of 256,256
  output:
    error: global error scalar
  """
  if y_pred.shape == y_true.shape and len(y_pred.shape)==2:  
    return 100*(np.mean(y_pred)-np.mean(y_true))/np.mean(y_true)
  else:
    print("Expected shape of (256,256)")

def Fluct_error(y_pred,y_true):
  """
  Fluctuation Error as metrics of prediction
  Input:
    y_pred: np.array with shape of 256,256
    y_ture: np.array with shape of 256,256
  output:
    error: fluctuation error
  """
  if y_pred.shape == y_true.shape and len(y_pred.shape)==2:  
    pred = y_pred-np.mean(y_pred)
    true = y_true-np.mean(y_true)
    return 100*( np.std(pred)-np.std(true))/np.std(true)
  else:
    print("Expected shape of (256,256)")


def ERS(y_pred,y_true):
  """
  Piexelwise Root squared Error as metrics of prediction
  Input:
    y_pred: np.array with shape of 256,256
    y_ture: np.array with shape of 256,256
  output:
    error: root squared error with shape of 256*256
  """
  if y_pred.shape == y_true.shape and len(y_pred.shape)==2:  
    return 100 * np.abs(y_pred - y_true)/np.mean(y_true)
  else:
    print("Expected shape of (256,256)")