import numpy as np
import pandas as pd


def display_histogram_train(path):
  for i in range(6):
    data = np.load(path+f'clf{i}.npy', allow_pickle=True)
    X = np.array(list(data[:, 1]), dtype=np.float)
    df = pd.DataFrame(data=X)
    df.hist(figsize=(60,60))

def display_histogram_test(path):
  test_set = np.load(path, allow_pickle=True)
  test_set_dic = test_set.reshape(-1,1)[0][0]
  for i in range(6):
    data = np.array(test_set_dic[i])
    X = np.array(list(data[:, 1]), dtype=np.float)
    df = pd.DataFrame(data=X)
    df.hist(figsize=(60,60))

