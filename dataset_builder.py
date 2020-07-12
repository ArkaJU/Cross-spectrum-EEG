import os
import time
import random
import resource
import numpy as np

from extract import extract_anns, extract_data
from constants import *


start = time.time()

class TrainDataset():
  def __init__(self, num_patients):

    self.num_patients = num_patients
    self.data_path = TRAIN_DATA_PATH
    self.ann_path = TRAIN_ANN_PATH
    self.patient_list = sorted(os.listdir(self.data_path))
    self.trainset_list = []                                 #list for trainset
    self.segs_global = []

  #@profile
  def extract_random_segments_for_given_patient(self, patient_no, num_segs_chosen_per_patient):   #helper

    current_patient = self.patient_list[patient_no]  
    patient_ann = current_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(self.ann_path + patient_ann)
    eeg_dict, info_dict = extract_data(self.data_path + current_patient, ann, onset, duration[-1])
    len_dict = {}
    
    for i in eeg_dict.keys(): 
      len_dict[i] = len(eeg_dict[i])

    #print(len_dict)

    tuples = []    #all (label, segment)
    for label in eeg_dict.keys():
      for seg in range(len_dict[label]): 
        tuples.append((int(label), eeg_dict[label][seg]))


    # l = []
    # for t in tuples:
    #   l.append(t[0])
    # print(f"tuples: {np.unique(l, return_counts=True)}")
    random.shuffle(tuples)

    labels = []
    selected_tuples = []

    for _ in range(num_segs_chosen_per_patient):
    #for _ in range(num_segs_chosen_per_patient):
      t = tuples.pop()
      selected_tuples.append(t)
      labels.append(t[0])

    del tuples

    self.segs_global.extend(labels)
    self.trainset_list.extend(selected_tuples)
    del selected_tuples


  #@profile
  def create(self, num_segs_chosen_per_patient):      #main
  
    for p in range(self.num_patients): 
      if (p+1)%10==0:
        print(f"patient_no: {p+1}")
        print(f"Time taken so far: {time.time()-start} seconds")
        print(f"segs: {np.unique(self.segs_global, return_counts=True)}")
        print(f"RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")
        print("\n")

      self.extract_random_segments_for_given_patient(patient_no=p, num_segs_chosen_per_patient=num_segs_chosen_per_patient)
    

train_set = TrainDataset(num_patients=160)  
train_set.create(num_segs_chosen_per_patient=300)
np.save(f"train_set.npy", train_set.trainset_list)

print(f"Total time: {time.time()-start} seconds")