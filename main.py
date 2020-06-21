from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import time
import numpy as np
import pickle

from constants import *
from utils import *

start = time.time()

train = True
test = True

if train == True:
  t1 = time.time()
  sleep_stages = [0,1,2,3,4,5] #edit to add more clf_ids to train

  for clf_id in sleep_stages:
    t2 = time.time()
    print("*****************************************************")
    print(f"CLF_ID:{clf_id}")

    #data_list = np.load(f'/content/drive/My Drive/Cross-spectrum-EEG/datasets/clf{clf_id}.npy', allow_pickle=True)
    data_list = np.load(f'/content/cleaned_data/clf_smote{clf_id}.npy', allow_pickle=True)
    
    X_train, Y_train = split_datalist(data_list, clf_id)
    print(f"Y_train: {np.unique(Y_train, return_counts=True)}")
    X_train = preprocess(X_train)
    
    # print(f"Example of training feature vector: {X_train[clf_id]}")
    # print(f"It's corresponding label: {Y_train[clf_id]}")
    # print("*****************************************************")
    # params = {
    #           'class_weight': 'balanced',
    #           'classes': [0, 1],
    #           'y': list(Y_train) 
    #           }

    # weights = compute_class_weight(**params)
    # weights_dict = {}
    # for i, w in enumerate(weights):
    #   weights_dict[i] = w

    # print(weights_dict)

    #clf = SGDClassifier(loss='hinge', verbose=1, class_weight=weights_dict)
    #clf = LogisticRegression(class_weight=weights_dict, max_iter=10000)
    #clf = RandomForestClassifier(class_weight=weights_dict, max_depth=100, random_state=0)
    #clf = SVC(kernel='linear', gamma='auto', class_weight=weights_dict)
    #clf.fit(X_train, Y_train)
    
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                  'C': [1, 10, 100, 1000]},
    #                 {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # print("Starting Grid Search...")
    #clf = GridSearchCV(SVC(), tuned_parameters)
    clf = SVC()
    clf.fit(X_train, Y_train)
    # print("Best parameters set found on development set:")
    # print(clf.best_params_)
    
    print(f"Training clf_{clf_id} complete!")
    print("Saving model..")
    print("\n")
    pickle.dump(clf, open(f'/content/drive/My Drive/Cross-spectrum-EEG/trained_models/clf_{clf_id}.sav','wb'))

    print(f"Total time taken for this sleep stage: {time.time()-t2} seconds")
    print("*****************************************************")
  print(f"Total training time: {time.time()-t1} seconds")





if test == True: 
  #test_set = np.load('/content/drive/My Drive/Cross-spectrum-EEG/datasets/test_set.npy', allow_pickle=True)
  test_set = np.load('/content/cleaned_data/test.npy', allow_pickle=True)
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
  X_test = preprocess_test(X_test)
  print(f"X_test.shape :{X_test.shape}")        
 
  Y_test = get_Y_test(test_set_dic)
  
  distances_from_hyperplane = []
  for clf_id in range(NUM_SLEEP_STAGES):
    distances_from_hyperplane.append(CLF[clf_id].decision_function(X_test[clf_id]))

  #(+ve) distance->on the positive side of hyperplane, confidence is directly proportional to the absolute value of the distance
  distances_from_hyperplane = np.array(distances_from_hyperplane) 
  print(f"distances_from_hyperplane.shape :{distances_from_hyperplane.shape}")        #(6, test_size)
  
  Y_preds = []    #highest
  Y_preds2 = []   #2nd highest

  #loop over the sample axis
  for i in range(distances_from_hyperplane.shape[1]):
    # print(i, distances_from_hyperplane[:, i])  #distances of a particular sample from hyperplanes of corresponding classifiers-> (6, )
    # print(f"Actual label: {Y_test[i]}")
    # print(f"Distance1: {np.argmax(distances_from_hyperplane[:, i])}->{np.max(distances_from_hyperplane[:, i])}")
    # print(f"Distance2: {np.argsort(distances_from_hyperplane[:, i])[-2]}->{np.sort(distances_from_hyperplane[:, i])[-2]}")
    # print()
    Y_preds.append(np.argmax(distances_from_hyperplane[:, i]))    #prediction is the label corresponding to which highest distance is obtained
    Y_preds2.append(np.argsort(distances_from_hyperplane[:, i])[-2])    #prediction is the label corresponding to which 2nd highest distance is obtained


  print(f"Y_preds: {Y_preds}")
  # np.save('Y_test.npy', Y_test)
  # np.save('preds.npy', Y_preds)
  # np.save('preds2.npy', Y_preds2)
  print("#####################################################")
  print(f"Y_test: {Y_test}")
  print(f"Accuracy: {accuracy_score(Y_test, Y_preds)}")
  print(f"Confusion matrix: \n{confusion_matrix(Y_test, Y_preds)}")
  print(f"Classification Report:\n {classification_report(Y_test, Y_preds)}")
  print()
  # print(f"Accuracy: {accuracy_score(Y_test, Y_preds2)}")
  # print(f"Confusion matrix: \n{confusion_matrix(Y_test, Y_preds2)}")
  # print(f"Classification Report:\n {classification_report(Y_test, Y_preds2)}")
  print("*****************************************************************************")

print(f"The whole process took {time.time()-start} seconds")
