from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import time
import numpy as np

from constants import *
from utils import get_len_dict, split_dataset

start = time.time()


dataset = np.load('dataset.npy', allow_pickle=True)
dic = dataset.reshape(-1,1)[0][0]
test_size = 0.20
        
SVMs = [svm.SVC(kernel='linear') for i in range(NUM_SLEEP_STAGES)]

for svm_id in range(NUM_SLEEP_STAGES):
  X_train, X_test, Y_train, Y_test = split_dataset(dic, svm_id, test_size=test_size)
  SVMs[svm_id].fit(X_train, Y_train)
  Y_pred = SVMs[svm_id].predict(X_test)
  print("*******************************")
  print(f"Test performance of SVM-{svm_id}:")
  print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
  print(f"Confusion matrix: {confusion_matrix(Y_test, Y_pred)}")
  print(f"Classification Report: {classification_report(Y_test, Y_pred)}")
  print("*******************************")

print(f"The whole process took {time.time()-start} seconds")
