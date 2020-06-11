import os
import time
import random
import numpy as np

from extract import extract_anns, extract_data
from constants import *
from features import feature_gen

start = time.time()

class DatasetBuilder:
  
  #@profile
  def __init__(self): 
    self.ref_segments = np.load('/content/reference_segments_10.npy', allow_pickle=True).reshape(-1, 1)[0][0]
    print(f"Refs loaded in {time.time()-start} seconds")
    
  #@profile
  def extract_random_segments_for_given_patient_during_warning(self, segment_label, patient_no):   #during warning related to AR(l)__autocorrelation lag

    current_patient_ = self.patient_list[patient_no]  
    print(f'\nCurrent Patient: {current_patient_}')
    patient_ann_ = current_patient_[:-4] + '-nsrr.xml'
    ann_, onset_, duration_ = extract_anns(self.ann_path + patient_ann_)
    eeg_dict_, info_dict_ = extract_data(self.data_path + current_patient_, ann_, onset_, duration_[-1])
    #print(np.random.choice(len(eeg_dict_[segment_label])-1), len(eeg_dict_[segment_label]))
    return (int(segment_label), eeg_dict_[segment_label][np.random.choice(len(eeg_dict_[segment_label])-1)])


  #@profile
  def extract_random_segments_for_given_patient(self, patient_no, num_segs_chosen_per_patient):   #helper

    current_patient = self.patient_list[patient_no]  
    print(f'\nCurrent Patient: {current_patient}')
    patient_ann = current_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(self.ann_path + patient_ann)
    eeg_dict, info_dict = extract_data(self.data_path + current_patient, ann, onset, duration[-1])
    len_dict = {}
    
    for i in eeg_dict.keys(): 
      len_dict[i] = len(eeg_dict[i])

    print(len_dict)

    tuples = []    #all (label, segment)
    for label in eeg_dict.keys():
      # flag = (label == 0  or label == 2)
      for seg in range(len_dict[label]): 
        tuples.append((int(label), eeg_dict[label][seg]))
        # if flag and np.random.random() < 0.3:
        #   tuples.append((int(label), eeg_dict[label][seg]))
        # if not flag:
        #   tuples.append((int(label), eeg_dict[label][seg]))

    # l = []
    # for t in tuples:
    #   l.append(t[0])
    # print(f"tuples: {np.unique(l, return_counts=True)}")
    random.shuffle(tuples)

    selected_tuples = []
    for i in range(num_segs_chosen_per_patient):
      selected_tuples.append(tuples.pop())
      # t = tuples.pop()
      # if t[0] == self.ref_label:
      #   selected_tuples.append(t)   #popping after shuffling equivalent to sampling randomly
      # if t[0] != self.ref_label and np.random.random()<0.2:
      #   selected_tuples.append(t)

    del tuples
    
    # l = []
    # for t in selected_tuples:
    #   l.append(t[0])

    # print(f"selected_tuples: {np.unique(l, return_counts=True)}")

    for t in selected_tuples:
      yield t


  #@profile
  def create(self, num_segs_chosen_per_patient):      #main
  
    segs_global = []
    for p in range(self.num_patients): 
      segs = []
      print(f"REF_ID: {self.ref_label}, patient_no: {p}")
      segment_tuple_generator = self.extract_random_segments_for_given_patient(patient_no=p, num_segs_chosen_per_patient=num_segs_chosen_per_patient)
      t1 = time.time()
      for i, selected_tuple in enumerate(segment_tuple_generator):
        #print(f"{i+1}. Selected label: {selected_tuple[0]}")
        t2 = time.time()
        self.generate_features_with_ref_segments(selected_tuple, p)    #different for both the child classes
        #print(time.time()-t2)
        #print()
        segs.append(selected_tuple[0])  
      print(f"Time taken for this patient: {time.time()-t1}")

      segs_global.extend(segs)
      print(f"segs: {np.unique(segs, return_counts=True)}")  #for this patient only
      print(f"segs_global: {np.unique(segs_global, return_counts=True)}")
      print(f"Time taken so far: {time.time()-start} seconds")
      print("\n")
      
    print("########################")
    print("\n")
    
    print(f"segs_global: {np.unique(segs_global, return_counts=True)}")    #accumulating over all patients

#************************************************************************************************************

class TestDataset(DatasetBuilder):
  def __init__(self, num_patients):
    super().__init__()
    self.ref_label = "N/A"
    self.num_patients = num_patients
    self.data_path = TEST_DATA_PATH
    self.ann_path = TEST_ANN_PATH
    self.testset_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}      #dict for testset
    self.patient_list = sorted(os.listdir(self.data_path))
    
  #@profile
  def generate_features_with_ref_segments(self, selected_tuple, patient_no, subs_flag=False):
    #print("Inside test")
    if subs_flag: print("Using Substitute")
    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    s1 = np.array(selected_segment)
    for ref in range(NUM_SLEEP_STAGES):            #extra loop over all ref labels
      F_avg = []
      for ref_segment in self.ref_segments[ref]:
        
        s2 = np.array(ref_segment)

        try:
          F = feature_gen(s1, s2)
          F_avg.append(F)
        except Warning:
          print("Warning encountered..")
          print("*************************")
          substitute_tuple = self.extract_random_segments_for_given_patient_during_warning(selected_label, patient_no)
          self.generate_features_with_ref_segments(substitute_tuple, patient_no, subs_flag=True)    #recursively call this function till Warningless segment is found (in practice Warning is rarely encountered, hence more than 1 recursive call is extremely rare)
          return    #important, else the local s1(which is the source of Warning) will continue executing, thus calling the functions in except block again and again
      
      self.testset_dict[ref].append((selected_label, np.mean(F_avg, axis=0)))
      np.save(f"test.npy", self.testset_dict)
#************************************************************************************************************

class TrainDataset(DatasetBuilder):
  def __init__(self, ref_label, num_patients):
    super().__init__()
    self.ref_label = ref_label
    self.num_patients = num_patients
    self.data_path = TRAIN_DATA_PATH
    self.ann_path = TRAIN_ANN_PATH
    self.patient_list = sorted(os.listdir(self.data_path))
    self.trainset_list = []                                 #list for trainset

  #@profile
  def generate_features_with_ref_segments(self, selected_tuple, patient_no, subs_flag=False):
    #print("Inside train")
    if subs_flag: print("Using Substitute")
    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    s1 = np.array(selected_segment)
    F_avg = []

    for ref_segment in self.ref_segments[self.ref_label]:
      
      s2 = np.array(ref_segment)

      try:
        F = feature_gen(s1, s2)
        F_avg.append(F)
      except Warning:
        print("Warning encountered..")
        print("*************************")
        substitute_tuple = self.extract_random_segments_for_given_patient_during_warning(selected_label, patient_no)
        self.generate_features_with_ref_segments(substitute_tuple, patient_no, subs_flag=True)    #recursively call this function till Warningless segment is found (in practice Warning is rarely encountered, hence more than 1 recursive call is extremely rare)
        return    #important, else the local s1(which is the source of Warning) will continue executing, thus calling the functions in except block again and again
      
    self.trainset_list.append((selected_label, np.mean(F_avg, axis=0)))
    np.save(f"clf{self.ref_label}.npy", self.trainset_list)
#************************************************************************************************************

train = True
test = False

if train == True:
  ref_label = 0
  train_set = TrainDataset(ref_label=ref_label, num_patients=160)  
  train_set.create(num_segs_chosen_per_patient=50)
  np.save(f"clf{ref_label}.npy", train_set.trainset_list)


if test == True:
  test_set = TestDataset(num_patients=40)  
  test_set.create(num_segs_chosen_per_patient=25)
  np.save(f"test.npy", test_set.testset_dict)


print(f"Total time: {time.time()-start} seconds")