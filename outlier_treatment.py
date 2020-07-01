import numpy as np
import pandas as pd
from IPython.display import display
from utils import out_std, out_iqr, describe
from constants import NUM_SLEEP_STAGES, NUM_FEATURES

def trimming(df, y, get_cutoffs, details):
  indices_to_be_removed = []
  if details: 
    print(f"Skew Before")
    for i in range(NUM_FEATURES):
      print(f"{i}: {df[i].skew()}")
      print("#####################")
      print()
  for i in range(NUM_FEATURES):
    lower, upper = get_cutoffs(df[i], return_thresholds=True)
    indices_to_be_removed += [j for j in range(df.shape[0]) if df[i][j]>upper or df[i][j]<lower]
  
  indices_to_be_removed = list(set(indices_to_be_removed))
  print(f"Number of rows to be dropped: {len(indices_to_be_removed)}")

  df.drop(df.index[indices_to_be_removed], inplace=True)
  y = np.delete(y, indices_to_be_removed, axis=0)
  if details: 
    print(f"Skew After")
    for i in range(NUM_FEATURES):
      print(f"{i}: {df[i].skew()}")
      print("#####################")
      print()
  return df, y


def flooring_capping(df, get_cutoffs, details):
  for i in range(NUM_FEATURES):
    if details: print(f"Skew before: {df[i].skew()}")
    lower, upper = get_cutoffs(df[i], return_thresholds=True)
    df[i] = np.where(df[i]<lower, lower, df[i])
    df[i] = np.where(df[i]>upper, upper, df[i])

    if details: 
      print(f"Skew After: {df[i].skew()}")
      print("#####################")
      print()
  return df


def replace_with_median(df, get_cutoffs, details):

  for i in range(NUM_FEATURES):
    if details: print(f"{i}:")
    if details: print(f"Skew before: {df[i].skew()}")

    median = df[i].quantile(0.50)
    lower, upper = get_cutoffs(df[i], return_thresholds=True)
    df[i] = np.where(df[i]<lower, median, df[i])
    df[i] = np.where(df[i]>upper, median, df[i])

    if details: 
      print(f"Skew After: {df[i].skew()}")
      print("#####################")
      print()
  return df


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


def treat_outliers(df, identification, treatment, y = None, details=True):

  df = remove_nan(df)    #removing nan(if any) before outlier treatment

  if identification=='iqr': get_cutoffs = out_iqr
  if identification=='std': get_cutoffs = out_std

  
  if treatment=='trimming':
    assert y is not None
    df, y = trimming(df, y, get_cutoffs, details)

  if treatment=='flooring_capping':
    df = flooring_capping(df, get_cutoffs, details)

  if treatment=='replace_with_median':
    #params = {}
    df = replace_with_median(df, get_cutoffs, details)
  
  if y is not None: 
    return df, y  
  else: 
    return df


def treat_train_outliers(identification, treatment, details=True):

  for label in range(NUM_SLEEP_STAGES):
    data = np.load(f'/content/original_data/clf{label}.npy', allow_pickle=True)
    X = np.array(list(data[:, 1]), dtype=np.float)
    y = np.array(data[:, 0]).astype('int')
    df = pd.DataFrame(data=X)
    if details: print(f"Label: {label}:")

    if details: display(describe(df))

    if treatment=='trimming':
      df, y = treat_outliers(df, identification, treatment, y, details=details)
    else:
      df = treat_outliers(df, identification, treatment, y=None, details=details)
    
    if details: display(describe(df))

    data = []
    for f, l in zip(df.to_numpy(), y):
      data.append((l, f))

    np.save(f'/content/cleaned_data/clf{label}.npy', data)
    


def treat_test_outliers(identification, treatment, details=True):
  test_set = np.load('/content/original_data/test_set_balanced.npy', allow_pickle=True)
  test_set_dic = test_set.reshape(-1,1)[0][0]
  
  for i in range(NUM_SLEEP_STAGES):
    data = np.array(test_set_dic[i])
    X = np.array(list(data[:, 1]), dtype=np.float)
    y = np.array(data[:, 0]).astype('int')
    df = pd.DataFrame(data=X)

    #display(describe(df))
    if treatment=='trimming': #NOT TO BE USED FOR TESTSET AS NUMBER OF DROPPED ROWS WILL DIFFER FOR EACH OF THE 6 LISTS
      df, y = treat_outliers(df, identification, treatment, y, details=details)
    else:
      df = treat_outliers(df, identification, treatment, y=None, details=details)
        #display(describe(df))
    #print()

    data = []
    for f, l in zip(df.to_numpy(), y):
      data.append((l, f))

    test_set_dic[i] = data
  np.save(f'/content/cleaned_data/test.npy', test_set_dic)




if __name__ == "__main__":
  identification = 'std'
  treatment = 'flooring_capping'
  treat_train_outliers(identification, treatment, details=False)
  treat_test_outliers(identification, treatment, details=False)
