import numpy as np
import pandas as pd
from IPython.display import display
from utils import out_std, out_iqr, describe


def treat_outliers(df, identification, treatment, details=True):
  if df.isnull().values.any():
    print(f'Data not OK')

    nan_values= []
    for j in range(df.shape[1]):
        nan_values.append(df[j].isnull().sum().sum())
    print(f"NaN values: {nan_values}")   #number of nan values corresponding to each feature
    print()

  for i in range(32):
   
    if details: print(f"{i}:")
    

    if identification=='iqr':
      lower, upper = out_iqr(df[i], return_thresholds=True)
    if identification=='std':
      lower, upper = out_std(df[i], nstd=3.0, return_thresholds=True)

    if details: print(f"Skew before: {df[i].skew()}")

    if treatment=='flooring_capping':
      df[i] = np.where(df[i]<lower, lower, df[i])
      df[i] = np.where(df[i]>upper, upper, df[i])

    if treatment=='median':
      median = df[i].quantile(0.50)
      if details: print(f"Median: {median}") 
      df[i] = np.where(df[i]<lower, median, df[i])
      df[i] = np.where(df[i]>upper, median, df[i])

    if details: 
      print(f"Skew After: {df[i].skew()}")
      print("#####################")
      print()

  return df


def train_remove_outliers(identification, treatment, details=True):

  for label in range(6):
    data = np.load(f'/content/original_data/clf{label}.npy', allow_pickle=True)
    X = np.array(list(data[:, 1]), dtype=np.float)
    y = np.array(data[:, 0]).astype('int')
    df = pd.DataFrame(data=X)
    if details: print(f"Label: {label}:")

    if details: display(describe(df))
    df = treat_outliers(df, identification, treatment, details=details)
    if details: display(describe(df))

    data = []
    for f, l in zip(df.to_numpy(), y):
      data.append((l, f))

    np.save(f'/content/cleaned_data/clf{label}.npy', data)
    


def test_remove_outliers(identification, treatment, details=True):
  test_set = np.load('/content/original_data/test_set_balanced (2).npy', allow_pickle=True)
  test_set_dic = test_set.reshape(-1,1)[0][0]
  
  for i in range(6):
    data = np.array(test_set_dic[i])
    X = np.array(list(data[:, 1]), dtype=np.float)
    y = np.array(data[:, 0]).astype('int')
    df = pd.DataFrame(data=X)

    #display(describe(df))
    df = treat_outliers(df, identification, treatment, details=details)
    #display(describe(df))
    #print()

    data = []
    for f, l in zip(df.to_numpy(), y):
      data.append((l, f))

    test_set_dic[i] = data
  np.save(f'/content/cleaned_data/test.npy', test_set_dic)




if __name__ == "__main__":
  train_remove_outliers(identification='iqr', treatment='median', details=False)
  test_remove_outliers(identification='iqr', treatment='median', details=False)
