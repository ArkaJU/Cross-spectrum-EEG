from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression

import time
import numpy as np
import pickle

from constants import *
from utils import *

start = time.time()

train = True
test = True

if train == True:
  start = time.time()
  sleep_stages = [0,1,2,3,4,5] #edit to add more clf_ids to train

  for clf_id in sleep_stages:
    
    print("*****************************************************")
    print(f"CLF_ID:{clf_id}")
    #dataset = np.load(f'/content/svm{clf_id}.npy', allow_pickle=True)
    #dic = dataset.reshape(-1,1)[0][0]
    #X_train, Y_train = split_dataset(dic, clf_id)

    data_list = np.load(f'/content/drive/My Drive/Cross-spectrum-EEG/datasets/clf{clf_id}.npy', allow_pickle=True)
    X_train, Y_train = split_datalist(data_list, clf_id)
    
    # print(f"Example of training feature vector: {X_train[clf_id]}")
    # print(f"It's corresponding label: {Y_train[clf_id]}")
    # print("*****************************************************")
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

    #clf = SGDClassifier(loss='hinge', verbose=1, class_weight=weights_dict)
    clf = LogisticRegression(class_weight=weights_dict, max_iter=10000)
    clf.fit(X_train, Y_train)
    print(f"Training clf_{clf_id} complete!")

    print("Saving model..")
    print("\n")
    pickle.dump(clf, open(f'/content/drive/My Drive/Cross-spectrum-EEG/trained_models/clf_{clf_id}.sav','wb'))

    print(f"Total time taken for this sleep stage: {time.time()-start} seconds")
    print("*****************************************************")
  print(f"Total training time: {time.time()-start} seconds")

if test == True: #NOT READY
  test_set = np.load('/content/drive/My Drive/Cross-spectrum-EEG/datasets/test_set.npy', allow_pickle=True)
  test_set_dic = test_set.reshape(-1,1)[0][0]
  path = '/content/drive/My Drive/Cross-spectrum-EEG/trained_models/'

  #loading trained models
  clf_0 = pickle.load(open(path + 'clf_0.sav','rb'))
  clf_1 = pickle.load(open(path + 'clf_1.sav','rb'))
  clf_2 = pickle.load(open(path + 'clf_2.sav','rb'))
  clf_3 = pickle.load(open(path + 'clf_3.sav','rb'))
  clf_4 = pickle.load(open(path + 'clf_4.sav','rb'))
  clf_5 = pickle.load(open(path + 'clf_5.sav','rb'))

  CLF = [clf_0, clf_1, clf_2, clf_3, clf_4, clf_5]    # list of classifiers

  X_test = get_X_test(test_set_dic)
  Y_test = get_Y_test(test_set_dic)

  probs = []
  for clf_id in range(6):
    probs.append(CLF[clf_id].predict_proba(X_test[clf_id]))

  probs = np.array(probs)  
  #print(probs.shape)         #(6, test_size, 2)
  
  Y_preds = []
  for i in range(probs.shape[1]):
    #print(probs[:, i, 1])
    #print(np.argmax(probs[:, i, 1]))
    Y_preds.append(np.argmax(probs[:, i, 1]))

  print(f"Y_preds: {Y_preds}")
  print("#####################################################")
  print(f"Y_test: {Y_test}")
  print(f"Accuracy: {accuracy_score(Y_test, Y_preds)}")
  print(f"Confusion matrix: \n{confusion_matrix(Y_test, Y_preds)}")
  print(f"Classification Report:\n {classification_report(Y_test, Y_preds)}")
  print("*****************************************************************************")

print(f"The whole process took {time.time()-start} seconds")
