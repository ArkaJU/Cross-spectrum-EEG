from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight

import time
import numpy as np
import pickle

from constants import *
from utils import *

start = time.time()

train = True
test = False

if train == True:
  start = time.time()
  sleep_stages = [0] #edit to add more svm_ids to train

  for svm_id in sleep_stages:
    

    #dataset = np.load(f'/content/svm{svm_id}.npy', allow_pickle=True)
    #dic = dataset.reshape(-1,1)[0][0]
    #X_train, Y_train = split_dataset(dic, svm_id)

    data_list = np.load(f'/content/drive/My Drive/Cross-spectrum-EEG/svm{svm_id}.npy', allow_pickle=True)
    X_train, Y_train = split_datalist(data_list, svm_id)
    print("*****************************************************")
    print(f"Example of training feature vector: {X_train[svm_id]}")
    print(f"It's corresponding label: {Y_train[svm_id]}")
    print("*****************************************************")

    params = {
              'class_weight': 'balanced',
              'classes': [0, 1],
              'y': list(Y_train) 
              }

    weights = compute_class_weight(**params)
    weights_dict = {}
    for i, w in enumerate(weights):
      weights_dict[i] = w

    print(weights_dict)

    clf = SGDClassifier(loss='hinge', verbose=1, class_weight=weights_dict)

    clf.fit(X_train, Y_train)
    print(f"Training svm_{svm_id} complete!")

    print("Saving model..")
    print("\n")
    pickle.dump(clf, open(f'/content/drive/My Drive/Cross-spectrum-EEG/trained_models/svm_{svm_id}.sav','wb'))

    print(f"Total time taken for this sleep stage: {time.time()-start} seconds")
  print(f"Total training time: {time.time()-start} seconds")

if test == True: #NOT READY
  for svm_id in range(NUM_SLEEP_STAGES):
    SVMs[svm_id] = pickle.load(open(f'/content/drive/My Drive/Cross-spectrum-EEG/trained_models/svm_{svm_id}.sav','rb'))
  X_test = get_X_test(dic)
  Y_test = get_Y_test(dic)
  Y_pred = []
  for i in range(len(X_test[0])): 
    probs = []
    for svm_id in range(NUM_SLEEP_STAGES):
      probs.append(SVMs[svm_id].predict_proba(X_test[svm_id][i].reshape(1, -1)))
    probs = np.array(probs)
    #print(probs.shape)
    probs = probs.squeeze()
    #print(probs.shape)
    Y_pred.append(probs[:, 1].argmax())
  print(Y_pred)
  print("#####################################################")
  print(Y_test)
  print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
  print(f"Confusion matrix: \n{confusion_matrix(Y_test, Y_pred)}")
  print(f"Classification Report:\n {classification_report(Y_test, Y_pred)}")
  print("*****************************************************************************")

print(f"The whole process took {time.time()-start} seconds")
