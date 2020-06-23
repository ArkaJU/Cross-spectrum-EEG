from imblearn.over_sampling import SMOTE
import numpy as np
import warnings

from constants import NUM_SLEEP_STAGES
from utils import remove_nan

warnings.filterwarnings("ignore") #suppressing warnings, don't understand what they are but could be important


n = 3500
desired_samples = {0:n, 1:n, 2:n, 3:n, 4:n, 5:n}
for i in range(NUM_SLEEP_STAGES):
  print(f"LABEL {i}")
  data = np.load(f'/content/original_data/clf{i}.npy', allow_pickle=True)
  
  X = np.array(list(data[:, 1]), dtype=np.float)
  y = np.array(data[:, 0]).astype('int')
  

  desired_samples[i] *= 5
  #print(desired_samples)
  print(f"y:       {np.unique(y, return_counts=True)}")
  sm = SMOTE(random_state=42, sampling_strategy=desired_samples)
  X_smote, y_smote = sm.fit_resample(X, y)
  print(f"y_smote: {np.unique(y_smote, return_counts=True)}")
  print("###########################################################")

  desired_samples[i] //= 5
  
  data = []
  for features, label in zip(X_smote, y_smote):
    data.append((label, features))
  np.save(f"/content/cleaned_data/clf_smote{i}.npy", data)