import numpy as np
import time
import resource
import os
import random
from collections import Counter
import h5py

from extract import extract_anns, extract_data
from constants import *
from features import feature_gen

excess_segments_needed_per_patient=4
start = time.time()
patient_list = sorted(os.listdir(TEST_DATA_PATH))
labels_global=[]
no_of_errors_encountered=0

class TestSetBuilder:

  def __init__(self, size): #size->number of patients from which segment bank will be made
    
    self.size = size
    self.testset_dict ={0:[],1:[],2:[],3:[],4:[],5:[]}
    self.ref_segments = np.load('/content/reference_segments_10.npy', allow_pickle=True).reshape(-1, 1)[0][0]
    
    print(f"Refs loaded in {time.time()-start} seconds")

  def extract_random_segments_for_given_patient_during_warning(self,segment_label, patient_no):   #during warning related to AR(l)__autocorrelation lag

    current_patient_ = patient_list[patient_no]  
    patient_ann_ = current_patient_[:-4] + '-nsrr.xml'
    ann_, onset_, duration_ = extract_anns(TEST_ANN_PATH + patient_ann_)
    eeg_dict_, info_dict_ = extract_data(TEST_DATA_PATH + current_patient_, ann_, onset_)
    return eeg_dict_[segment_label][np.random.choice(len(eeg_dict[segment_label]))]

  
  def generate_features_with_ref_segments(self, selected_tuple,patient_no):
      
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
          global no_of_errors_encountered
          no_of_errors_encountered+=1
          substitute=extract_random_segments_for_given_patient_during_warning(selected_label,patient_no)
          self.generate_features_with_ref_segments(substitute,patient_no)

      self.testset_dict[ref].append((selected_label, np.mean(F_avg, axis=0)))

        
    
    #print(f"Shape of data_list[{self.ref_label}]={number_of_segments_for_ith_key}")
    #print(f"Feature vector extracted:{self.data_list[number_of_segments_for_ith_key-1]}")

 

  def extract_test_segments_for_given_patient(self, patient_no):   #helper

    current_patient = patient_list[patient_no]  
    patient_ann = current_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(TEST_ANN_PATH + patient_ann)
    eeg_dict, info_dict = extract_data(TEST_DATA_PATH + current_patient, ann, onset,duration[-1])
    len_dict = {}
    
    for i in eeg_dict.keys(): 
      len_dict[i] = len(eeg_dict[i])
      #print(f"eeg_dict{i} is {eeg_dict[i]}")
      #random.shuffle(eeg_dict[i])
    #print(len_dict)

    selected_tuples=[]
    for _ in range(4):    #approx 4 segments of each label from each patient
      for i in eeg_dict.keys():
        if len_dict[i]!=0:
          selected_tuples.append((int(i),eeg_dict[i][np.random.choice(len(eeg_dict[i]))]))
          if i==4:
            for j in range(min(5,len_dict[i])):
              selected_tuples.append((int(i), eeg_dict[i][j]))

    for _ in range(excess_segments_needed_per_patient):   
      for i in eeg_dict.keys():
        if i==4:
          continue
        if len_dict[i]!=0:
          selected_tuples.append((int(i),eeg_dict[i][np.random.choice(len(eeg_dict[i]))]))
    print(f"RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")   
  
    for t in selected_tuples:
      yield t


  def create_testset(self):      #main
  
    segs_global = []
    for p in range(self.size): 
      print(f"Patient ID: {p}")
      segs = []
      segment_generator = self.extract_test_segments_for_given_patient(patient_no=p)
      for segment in segment_generator:
        #print(segment[0])
        self.generate_features_with_ref_segments(segment,p)
        segs.append(segment[0])  #just appending the label for now, can save segment[1] for actual segment
      segs_global = segs_global + segs
      print(f"segs: {np.unique(segs, return_counts=True)}")  #for this patient only
      print(f"segs_global: {np.unique(segs_global, return_counts=True)}")
      print(f"Time taken so far: {time.time()-start} seconds")
      print("\n")
      np.save(f"/content/test_set_balanced.npy", dataset.testset_dict)

      
    print("########################")
    print("\n")
    
    print(f"segs_global: {np.unique(segs_global, return_counts=True)}")    #accumulating over all patients



dataset = TestSetBuilder(size=40)  #size->number of patients from which random segment  will be chosen

dataset.create_testset()
#print(f"saving svm{ref_label} dataset...")

np.save(f"/content/test_set_balanced.npy", dataset.testset_dict)

print(f"Total errors encounterd: {no_of_errors_encountered}")
print(f"Total time: {time.time()-start} seconds")