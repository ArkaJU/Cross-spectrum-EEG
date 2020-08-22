import copy
import time
import numpy as np
import pickle
from sklearn.svm import SVC
from scipy.stats import mode
from itertools import combinations 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from constants import *
from utils import *


start = time.time()

train = True
test = True
preprocessing = 'standardize'
#preprocessing = 'normalize'

#val
# hyperparam_list = [ {'C':10,  'gamma':0.001, 'kernel':'rbf'},
#                     {'C':100, 'gamma':0.001, 'kernel':'rbf'}, 
#                     {'C':10,  'gamma':0.001, 'kernel':'rbf'},
#                     {'C':10,  'gamma':0.001, 'kernel':'rbf'},
#                     {'C':1,   'gamma':0.01,  'kernel':'rbf'},
#                     {'C':1,   'gamma':0.01,  'kernel':'rbf'},
#                     {'C':1,   'gamma':0.01,  'kernel':'rbf'},
#                     {'C':1,   'gamma':0.01,  'kernel':'rbf'},
#                     {'C':10,  'gamma':0.01,  'kernel':'rbf'},
#                     {'C':1,   'gamma':0.01,  'kernel':'rbf'},]

#test
# hyperparam_list = [ {'C':100, 'gamma':0.001, 'kernel':'rbf'},
#                     {'C':10,  'gamma':0.01,  'kernel':'rbf'}, 
#                     {'C':10,  'gamma':0.001, 'kernel':'rbf'},
#                     {'C':10,  'gamma':0.001, 'kernel':'rbf'},
#                     {'C':1,   'gamma':0.01,  'kernel':'rbf'},
#                     {'C':1,   'gamma':0.01,  'kernel':'rbf'},
#                     {'C':10,  'gamma':0.001, 'kernel':'rbf'},
#                     {'C':1,   'gamma':0.01,  'kernel':'rbf'},
#                     {'C':10,  'gamma':0.01,  'kernel':'rbf'},
#                     {'C':1,   'gamma':0.01,  'kernel':'rbf'},]

if train == True:
  t1 = time.time()
  sleep_stages = [0,1,2,3,4] #clf_ids

  print("Training model....")

  combos = list(combinations(list(range(NUM_SLEEP_STAGES)), 2))
  #combos = list(combinations(list(range(6)), 2))
  
  for idx, (clf_id1, clf_id2) in enumerate(combos):

    t2 = time.time()
    print("*****************************************************")
    print(f"{idx}. CLF_ID1: {clf_id1}, CLF_ID2: {clf_id2}")

    data_list1 = np.load(f'/content/cleaned_data/clf{clf_id1}.npy', allow_pickle=True)
    data_list2 = np.load(f'/content/cleaned_data/clf{clf_id2}.npy', allow_pickle=True)

    X_train, Y_train = split_datalist(data_list1, clf_id1, data_list2, clf_id2)

    
    print(X_train.shape)
    print(f"Y_train: {np.unique(Y_train, return_counts=True)}")
    print(f"preprocessing: {preprocessing}")
    X_train = preprocess(X_train, preprocessing)
    
    classes = [-1, clf_id1, clf_id2]
    params = {
          'class_weight': 'balanced',
          'classes': classes,
          'y': list(Y_train) 
          }

    weights = compute_class_weight(**params)
    weights_dict = {}
    for c, w in zip(classes, weights):
      weights_dict[c] = w

    print(weights_dict)
    # hparams = hyperparam_list[idx]
    # print(hparams)
    clf = SVC(class_weight=weights_dict)
    clf.fit(X_train, Y_train)

    print(f"Training clf_{clf_id1}{clf_id2} complete!")
    print("Saving model..")
    
    pickle.dump(clf, open(f'/content/drive/My Drive/Cross-spectrum-EEG/trained_models/clf_{clf_id1}{clf_id2}.sav','wb'))
    #pickle.dump(clf, open(f'/content/clf_{clf_id1}{clf_id2}.sav','wb'))

    print(f"Total time taken for this sleep stage: {time.time()-t2} seconds")
    print("*****************************************************")
    print("\n")
  print(f"Total training time: {time.time()-t1} seconds")





if test == True: 

  #path = '/content/'
  path = '/content/drive/My Drive/Cross-spectrum-EEG/trained_models/'

  #loading trained models
  clf_01 = pickle.load(open(path + 'clf_01.sav','rb'))
  clf_02 = pickle.load(open(path + 'clf_02.sav','rb'))
  clf_03 = pickle.load(open(path + 'clf_03.sav','rb'))
  clf_04 = pickle.load(open(path + 'clf_04.sav','rb'))
  clf_12 = pickle.load(open(path + 'clf_12.sav','rb'))
  clf_13 = pickle.load(open(path + 'clf_13.sav','rb'))
  clf_14 = pickle.load(open(path + 'clf_14.sav','rb'))
  clf_23 = pickle.load(open(path + 'clf_23.sav','rb'))
  clf_24 = pickle.load(open(path + 'clf_24.sav','rb'))
  clf_34 = pickle.load(open(path + 'clf_34.sav','rb'))

  CLF = [clf_01, clf_02, clf_03, clf_04, clf_12, clf_13, clf_14, clf_23, clf_24, clf_34]    # list of classifiers

  test_set = np.load('/content/drive/My Drive/ref5_dj6/test_set_balanced_ref5.npy', allow_pickle=True)
  test_set_dict = test_set.reshape(-1,1)[0][0]
  X_test, Y_test = split_dataset(test_set_dict)
  X_test = preprocess_test(X_test, preprocessing)
  print(X_test.shape)
  preds = []
  combos = list(combinations(list(range(NUM_SLEEP_STAGES)), 2))

  for idx, (clf_id1, clf_id2) in enumerate(combos):
    print(idx, clf_id1, clf_id2)
    pred = CLF[idx].predict(np.concatenate((X_test[clf_id1], X_test[clf_id2]), axis=1))
    print(np.unique(pred, return_counts=True))
    preds.append(pred)

  preds = np.array(preds)
  preds = preds.T

  print()
  print("Accuracies of individual classifiers:")
  for i, (clf_id1,clf_id2) in enumerate(combos):
    test_Y = copy.deepcopy(Y_test)
    test_Y[np.where((test_Y != clf_id1) & (test_Y != clf_id2))[0].tolist()] = -1 #none
    predictions = preds[:, i]
    print(f"clf{clf_id1}{clf_id2}: {accuracy_score(test_Y, predictions)}")
  print()

  Y_preds = []
  for i in range(preds.shape[0]):  #over sample axis
    U = preds[i, :]
    x = mode(U[U!=-1])
    if x[0].shape==(0,):         #if all -1
      md = np.random.randint(5)
    else:
      md = x[0][0]
    # if md!=Y_test[i]:
    #   print(U)
    #   print(md)
    #   print(Y_test[i])
    #   print()
    Y_preds.append(md)      #Take the mode over values other than -1(none)

  print(f"Y_preds: {Y_preds}")
  print("#####################################################")
  print(f"Y_test: {Y_test}")
  print(f"Accuracy: {accuracy_score(Y_test, Y_preds)}")
  print(f"Confusion matrix: \n{confusion_matrix(Y_test, Y_preds)}")
  print(f"Classification Report:\n {classification_report(Y_test, Y_preds)}")
  print()

  print("*****************************************************************************")

print(f"The whole process took {time.time()-start} seconds")
