from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from constants import NUM_SLEEP_STAGES, NUM_FEATURES


def describe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df.mean().rename('mean'),
                      df.median().rename('median'),
                      df.max().rename('max'),
                      df.min().rename('min')
                     ], axis=1).T

                     
def out_std(s, nstd=3.0, return_thresholds=False):
    """
    Return a boolean mask of outliers for a series
    using standard deviation, works column-wise.
    param nstd:
        Set number of standard deviations from the mean
        to consider an outlier
    :type nstd: ``float``
    param return_thresholds:
        True returns the lower and upper bounds, good for plotting.
        False returns the masked array 
    :type return_thresholds: ``bool``
    """
    data_mean, data_std = s.mean(), s.std()
    cut_off = data_std * nstd
    lower, upper = data_mean - cut_off, data_mean + cut_off
    if return_thresholds:
        return lower, upper
    else:
        return [True if x < lower or x > upper else False for x in s]


def out_iqr(s, k=1.5, return_thresholds=False):
    """
    Return a boolean mask of outliers for a series
    using interquartile range, works column-wise.
    param k:
        some cutoff to multiply by the iqr
    :type k: ``float``
    param return_thresholds:
        True returns the lower and upper bounds, good for plotting.
        False returns the masked array 
    :type return_thresholds: ``bool``
    """
    # calculate interquartile range
    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q75 - q25
    # calculate the outlier cutoff
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    if return_thresholds:
        return lower, upper
    else: # identify outliers
        return [True if x < lower or x > upper else False for x in s]


def remove_nan(df: pd.DataFrame) -> pd.DataFrame:

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

    nan_values = []
    indices = list(np.arange(NUM_FEATURES))
    for j in range(df.shape[1]):
      nan_values.append(df[j].isnull().sum().sum())

    print(f'After:')
    print(f"Indices:    {indices}")        #index of feature
    print(f"NaN values: {nan_values}")     #number of nan values corresponding to each feature
    print()

  else:
    print(f"Data has no NaN values")
  
  return df


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
  
  row_mask = np.load(path + 'row_mask_6.npy', allow_pickle=True)  #mask matrices have fixed shape for same scale and time i.shape/j.shape=(263,3750)
  column_mask = np.load(path + 'column_mask_6.npy', allow_pickle=True) 
  
  accum = np.multiply(W, np.multiply(row_mask+1, column_mask+1))
  accum = np.sum(accum)
  accum_sq = np.multiply(W, np.multiply((row_mask+1)**2, (column_mask+1)**2))
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


def split_dataset(data_dict: dict) -> [np.ndarray, list]:
  """
  data_dict -> labelwise dict of (selected_seg_label, avg feature_vector of ref_label: selected_seg X ref_seg)
  """
  X_0, X_1, X_2, X_3, X_4, X_5 = ([] for _ in range(NUM_SLEEP_STAGES)) #initializing 6 empty strings
  X = [X_0, X_1, X_2, X_3, X_4, X_5]
  
  #features_to_keep = [0,1,2,5,6,7,9,13,16,18,19,20,21,25,26,27,28,29,30,31]
  features_to_keep = list(range(17))+list(range(24,32))
  features_to_delete = []
  for i in range(32):
    if i not in features_to_keep:
      features_to_delete.append(i) 

  print(f"Features kept: {features_to_keep}")

  for i in range(NUM_SLEEP_STAGES):
    for tup in data_dict[i]:  
      #x = np.delete(tup[1], 8)
      x = tup[1]       #uncomment if nothing to delete
      X[i].append(x) 

  Y = []
  clf_id = np.random.randint(NUM_SLEEP_STAGES) #emphasizing that it doesn't  matter which ref label we'll use because the label of the randomly selected sample will be same for all keys in the dict
  for tup in data_dict[clf_id]:   
    Y.append(tup[0]) 

  return np.array(X), Y         #(num_sleep_stages, total_samples, num_features), num_samples


def split_datalist(data_list: np.ndarray, clf_id: int) -> [np.ndarray, np.ndarray]:    
  """
  data_list -> list of (selected_seg_label, avg feature_vector of ref_label: selected_seg X ref_seg)
  clf_id -> signifies which SVM this data is meant for
  """
  # print(f"clf_id:{clf_id}")

  X = np.array(list(data_list[:, 1]), dtype=np.float)
  Y = np.array(data_list[:, 0]).astype('int')

  # print("Original labels:")
  # print(Y)

  pos_indices = np.where(Y == clf_id)[0]
  Y[np.where(Y != clf_id)[0].tolist()] = -1
  Y[np.where(Y == clf_id)[0].tolist()] = 1
  Y[np.where(Y == -1)[0].tolist()] = 0

  assert np.all(np.where(Y == 1)[0] == pos_indices)

  # print("Binarized labels:")
  # print(Y)
  # print(X.shape)
  # print(Y.shape)
  
  return X, Y                #(total_samples, num_featues), (total_samples,) 


#used for training and in correntropy calculation
def preprocess(X: np.ndarray) -> np.ndarray:
  #data = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
  m = np.mean(X, axis=0)
  s = np.std(X, axis=0)

  data = (X - m)/s
  return data               #(total_samples, num_featues)


def preprocess_test(X: np.ndarray) -> np.ndarray:
  #data = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
  m = np.mean(X, axis=1)[:, np.newaxis, :]
  s = np.std(X, axis=1)[:, np.newaxis, :]
  
  data = (X - m)/s
  return data                   #(num_sleep_stages, total_samples, num_features)
