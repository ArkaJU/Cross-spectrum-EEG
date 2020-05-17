from sklearn.model_selection import train_test_split
import numpy as np
from constants import NUM_SLEEP_STAGES

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

def preprocess(X):
  #data = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
  #data = X/np.max(X, axis=0)
  data = (X - np.mean(X, axis=0))/np.std(X, axis=0)
  return data