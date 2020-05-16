import numpy as np
import time
import resource
import os
import random
import psutil
import humanize
from collections import Counter
import h5py

from extract import extract_anns, extract_data
from constants import *
from features import feature_gen


start = time.time()
patient_list = sorted(os.listdir(TRAIN_DATA_PATH))
labels_global=[]

class DatasetBuilder:

  def __init__(self, ref_label, size): #size->number of patients from which segment bank will be made
    
    self.size = size
    self.data_list = []
    self.ref_label = ref_label
    self.ref_segments = np.load('/content/drive/My Drive/Cross-spectrum-EEG/reference_segments.npy', allow_pickle=True).reshape(-1, 1)[0][0]
    
    print(f"Refs loaded in {time.time()-start} seconds")
  
  
  def generate_features_with_ref_segments(self, selected_tuple):
      
    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    t, s = np.arange(len(selected_segment)), np.array(selected_segment)
    #l = 1 if selected_label == self.ref_label else 0
    F_avg = []
    for ref_segment in self.ref_segments[self.ref_label]:
      t1, s1 = t, s
      t2, s2 = np.arange(len(ref_segment)), np.array(ref_segment)
      # print(s1.shape)
      # print(s2.shape)

      #converting segments to equal lengths
      S1 = s1[np.argwhere((t1 >= min(t2)) & (t1 <= max(t2))).flatten()]
      S2 = s2[np.argwhere((t2 >= min(t1)) & (t2 <= max(t1))).flatten()]

      # print(s1.shape)
      # print(s2.shape)
      # print("*************************")
      try:
        F = feature_gen(t1, S1, t2, S2)
      except Warning:
        num_features = 13
        print("Warning encountered..")
        print(f"s1.shape: {s1.shape}")
        print(f"s2.shape: {s2.shape}")
        print(f"S1.shape: {S1.shape}")
        print(f"S2.shape: {S2.shape}")
        print("*************************")
        F = np.zeros(num_features)
      F_avg.append(F)
    
    self.data_list.append((selected_label, np.mean(F_avg, axis=0)))
    #print(f"Shape of data_list[{self.ref_label}]={number_of_segments_for_ith_key}")
    #print(f"Feature vector extracted:{self.data_list[number_of_segments_for_ith_key-1]}")



  def extract_random_segments_for_given_patient(self, patient_no, num_segs_chosen):   #helper

    current_patient = patient_list[patient_no]  
    patient_ann = current_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
    eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + current_patient, ann, onset)
    len_dict = {}
    
    for i in eeg_dict.keys(): 
      len_dict[i] = len(eeg_dict[i])
      #print(f"eeg_dict{i} is {eeg_dict[i]}")
      #random.shuffle(eeg_dict[i])
    print(len_dict)

    tuples = []    #all (label, segment)
    for label in eeg_dict.keys():
      for seg in range(len_dict[label]): 
        tuples.append((int(label), eeg_dict[label][seg]))


    random.shuffle(tuples)

    selected_tuples = []
    for i in range(num_segs_chosen):
      selected_tuples.append(tuples.pop())   #popping after shuffling equivalent to sampling randomly
    
    #print(f"RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")   
    del tuples
    #print(f"RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")   
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), \
          " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    for t in selected_tuples:
      yield t


  def create_dataset_of_particular_stage(self, num_segs_chosen):      #main
  
    segs_global = []
    for p in range(self.size): 
      segs = []
      print(f"SVM_id: {self.ref_label}, patient_no: {p}")
      segment_generator = self.extract_random_segments_for_given_patient(patient_no=p, num_segs_chosen=num_segs_chosen)
      for segment in segment_generator:
        print(segment[0])
        self.generate_features_with_ref_segments(segment)
        segs.append(segment[0])  #just appending the label for now, can save segment[1] for actual segment
      segs_global = segs_global + segs
      print(f"segs: {np.unique(segs, return_counts=True)}")  #for this patient only
      print(f"segs_global: {np.unique(segs_global, return_counts=True)}")
      print(f"Time taken so far: {time.time()-start} seconds")
      print("\n")
      
    print("########################")
    print("\n")
    
    print(f"segs_global: {np.unique(segs_global, return_counts=True)}")    #accumulating over all patients


print(f"Start time:{time.time()}")
ref_label = 5
dataset = DatasetBuilder(size=10, ref_label=ref_label)  #size->number of patients from which random segment  will be chosen

dataset.create_dataset_of_particular_stage(num_segs_chosen=50)
print(f"saving svm{ref_label} dataset...")

np.save(f"svm{ref_label}.npy", dataset.data_list)


print(f"Total time: {time.time()-start} seconds")