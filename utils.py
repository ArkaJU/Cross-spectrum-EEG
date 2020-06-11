from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from constants import NUM_SLEEP_STAGES, NUM_FEATURES


def remove_nan(data):

  #print(data.shape)
  df = pd.DataFrame(np.array(list(data[:, 1]), dtype=np.float))
  if df.isnull().values.any():
    print(f'Data not OK, removing nan values..')
    print()
    nan_values = []
    indices = list(np.arange(NUM_FEATURES))
    for j in range(df.shape[1]):
      nan_values.append(df[j].isnull().sum().sum())
    
    print(f'Before:')
    print(f"Indices:    {indices}")      #index of feature   
    print(f"NaN values: {nan_values}")   #number of nan values corresponding to each feature
    print()

    df = df.fillna(df.median())  #replacing nan with median
    df2 = df.to_numpy()
    for j in range(df2.shape[0]):
      data[j][1] = df2[j]
    

    nan_values = []
    indices = list(np.arange(NUM_FEATURES))
    for j in range(df.shape[1]):
      nan_values.append(df[j].isnull().sum().sum())

    print(f'After:')
    print(f"Indices:    {indices}")        #index of feature
    print(f"NaN values: {nan_values}")     #number of nan values corresponding to each feature
    print()

  else:
    print(f"Data is OK")
  
  return data


#@profile
def correntropy(x, y):
    #N = len(x)
    X = preprocess(x)
    Y = preprocess(y)
    s = np.std(X, axis=0)
    #print(f"std dev: {s}")
    V = np.exp(-0.5*np.square(X - Y)/s**2)
    #CIP = 0.0 # mean in feature space should be subtracted!!
    #for i in range(0, N):
        #CIP += np.average(np.exp(-0.5*(x- y[i])**2/s**2))/N
    return V


#@profile
def get_sums(W):
  path = '/content/matrix_masks/'
  
  row_mask = np.load(path + 'row_mask.npy', allow_pickle=True)  #mask matrices have fixed shape for same scale and time i.shape/j.shape=(263,3750)
  column_mask = np.load(path + 'column_mask.npy', allow_pickle=True)  #for dj=1/24 and 30 second segments
  
  accum = np.multiply(W, np.multiply(row_mask, column_mask))
  accum = np.sum(accum)
  accum_sq = np.multiply(W, np.multiply(row_mask**2, column_mask**2))
  accum_sq = np.sum(accum_sq)
  
  return accum, accum_sq


#@profile
def get_sums2(total_scales, total_time, W):

  ones = np.ones((total_scales, total_time))
  x = np.arange(total_scales).reshape(-1, 1)
  y = np.arange(total_time).reshape(1,-1)
  i = np.multiply(ones, x).astype(int)
  j = np.multiply(ones, y).astype(int)
  accum = np.multiply(W, np.multiply(i, j))  #kind of masking
  accum = np.sum(accum)
  accum_sq = np.multiply(W, np.multiply(i**2, j**2))
  accum_sq = np.sum(accum_sq)
  
  return accum, accum_sq


def get_len_dict(eeg_dict):  
  len_dict = {}
  for i in eeg_dict.keys():
    len_dict[i] = len(eeg_dict[i])
  print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in len_dict.items()) + "}")


def get_X_test(dic):
  X_0 = []
  X_1 = []
  X_2 = []
  X_3 = []
  X_4 = []
  X_5 = []

  for tup in dic[0]:   
    X_0.append(tup[1])
  for tup in dic[1]:   
    X_1.append(tup[1]) 
  for tup in dic[2]:   
    X_2.append(tup[1]) 
  for tup in dic[3]:   
    X_3.append(tup[1]) 
  for tup in dic[4]:   
    X_4.append(tup[1]) 
  for tup in dic[5]:   
    X_5.append(tup[1])  

  X = [X_0, X_1, X_2, X_3, X_4, X_5]
  return X


def get_Y_test(dic):
  svm_id = np.random.randint(NUM_SLEEP_STAGES) #emphasizing that it doesn't  matter which ref label we'll use because the label of the randomly selected sample will be same for all keys in the dict
  Y = []
  for tup in dic[svm_id]:   
    Y.append(tup[0]) 
  return Y


def split_dataset(dic, clf_id):     
  """dic -> ref_label wise list of 
  (selected_seg_label, avg feature_vector of ref_label: selected_seg X ref_seg)
  
  clf_id -> signifies which SVM this data is meant for
  """
  X = []
  Y = []
  for tup in dic[clf_id]:   
    Y.append(tup[0])
    X.append(tup[1])

  X = np.array(X)
  Y = np.array(Y)
  # print("Original labels:")
  # print(np.unique(Y, return_counts=True))
  # print(f"clf_id:{clf_id}")
  pos_indices = np.where(Y == clf_id)[0]
  Y[np.where(Y != clf_id)[0].tolist()] = -1
  Y[np.where(Y == clf_id)[0].tolist()] = 1
  Y[np.where(Y == -1)[0].tolist()] = 0
  assert np.all(np.where(Y == 1)[0] == pos_indices)
  # print("Binarized labels:")
  # print(np.unique(Y, return_counts=True))
  return X, Y


def split_datalist(data_list, clf_id):     
  """
  np.ndarray -> list of (selected_seg_label, avg feature_vector of ref_label: selected_seg X ref_seg)
  clf_id -> signifies which SVM this data is meant for
  """
  #X = np.array(data_list[:, 1])
  X = np.array(list(data_list[:, 1]), dtype=np.float)
  Y = np.array(data_list[:, 0]).astype('int')
  # for tup in data_list:   
  #   Y.append(tup[0])
  #   X.append(tup[1])

  # print("Original labels:")
  # print(np.unique(Y, return_counts=True))

  # print(f"clf_id:{clf_id}")
  pos_indices = np.where(Y == clf_id)[0]
  Y[np.where(Y != clf_id)[0].tolist()] = -1
  Y[np.where(Y == clf_id)[0].tolist()] = 1
  Y[np.where(Y == -1)[0].tolist()] = 0

  assert np.all(np.where(Y == 1)[0] == pos_indices)

  # print("Binarized labels:")
  # print(np.unique(Y, return_counts=True))

  # print(X.shape)
  # print(Y.shape)
  
  return X, Y

#used for training and in correntropy calculation
def preprocess(X):
  #data = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
  #data = X/np.max(X, axis=0)
  m = np.mean(X, axis=0)
  s = np.std(X, axis=0)
  data = (X - m)/s
  return data


def preprocess_test(X):
  #data = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
  #data = X/np.max(X, axis=0)
  m = np.mean(X, axis=1)[:, np.newaxis, :]
  s = np.std(X, axis=1)[:, np.newaxis, :]
  data = (X - m)/s
  return data