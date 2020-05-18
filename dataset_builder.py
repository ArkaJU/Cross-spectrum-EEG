import numpy as np
import time
import resource
import os
import random
import psutil
import humanize
from collections import Counter

from extract import extract_anns, extract_data
from constants import *
from features import feature_gen
from line_profiler import LineProfiler

start = time.time()
patient_list = sorted(os.listdir(TRAIN_DATA_PATH))
labels_global=[]

class DatasetBuilder:
  
  #@profile
  def __init__(self, ref_label, num_patients): #num_patients->number of patients from which segment bank will be made
    
    self.ref_label = ref_label
    self.num_patients = num_patients
    self.data_list = []
    self.ref_segments = np.load('/content/drive/My Drive/Cross-spectrum-EEG/datasets/reference_segments.npy', allow_pickle=True).reshape(-1, 1)[0][0]
    
    print(f"Refs loaded in {time.time()-start} seconds")

  #@profile
  def extract_random_segments_for_given_patient(self, patient_no, num_segs_chosen_per_patient):   #helper

    current_patient = patient_list[patient_no]  
    patient_ann = current_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
    eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + current_patient, ann, onset, duration[-1])
    len_dict = {}
    
    for i in eeg_dict.keys(): 
      len_dict[i] = len(eeg_dict[i])

    print(len_dict)

    tuples = []    #all (label, segment)
    for label in eeg_dict.keys():
      for seg in range(len_dict[label]): 
        tuples.append((int(label), eeg_dict[label][seg]))


    random.shuffle(tuples)

    selected_tuples = []
    for i in range(num_segs_chosen_per_patient):
      selected_tuples.append(tuples.pop())   #popping after shuffling equivalent to sampling randomly
    
    #print(f"RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")   
    del tuples
    #print(f"RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")   
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), \
          " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    for t in selected_tuples:
      yield t

  #@profile
  def create_dataset_of_particular_stage(self, num_segs_chosen_per_patient):      #main
  
    segs_global = []
    for p in range(self.num_patients): 
      segs = []
      print(f"SVM_id: {self.ref_label}, patient_no: {p}")
      segment_tuple_generator = self.extract_random_segments_for_given_patient(patient_no=p, num_segs_chosen_per_patient=num_segs_chosen_per_patient)
      s1 = time.time()
      for i, selected_tuple in enumerate(segment_tuple_generator):
        print(f"{i+1}. Selected label: {selected_tuple[0]}")
        #print(len(selected_tuple[1])/SAMPLE_RATE)
        s2 = time.time()
        # lp = LineProfiler()
        # lp_wrapper = lp(self.generate_features_with_ref_segments)
        # lp_wrapper(selected_tuple)
        # lp.print_stats(output_unit=1e-3)
        self.generate_features_with_ref_segments(selected_tuple)
        print(time.time()-s2)
        print()
        # print()
        # print()
        segs.append(selected_tuple[0])  
      print(time.time()-s1)

      segs_global.extend(segs)
      print(f"segs: {np.unique(segs, return_counts=True)}")  #for this patient only
      print(f"segs_global: {np.unique(segs_global, return_counts=True)}")
      print(f"Time taken so far: {time.time()-start} seconds")
      print("\n")
      
    print("########################")
    print("\n")
    
    print(f"segs_global: {np.unique(segs_global, return_counts=True)}")    #accumulating over all patients

  #@profile
  def generate_features_with_ref_segments(self, selected_tuple):
      
    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    #t, s = np.arange(len(selected_segment)), np.array(selected_segment)
    s1 = np.array(selected_segment)
    F_avg = []
    for ref_segment in self.ref_segments[self.ref_label]:
      #print(len(ref_segment)/SAMPLE_RATE)
      # t1, s1 = t, s
      # t2, s2 = np.arange(len(ref_segment)), np.array(ref_segment)

      s2 = np.array(ref_segment)

      #converting segments to equal lengths
      # S1 = s1[np.argwhere((t1 >= min(t2)) & (t1 <= max(t2))).flatten()]
      # S2 = s2[np.argwhere((t2 >= min(t1)) & (t2 <= max(t1))).flatten()]

      # print(f"s1.shape: {s1.shape}")
      # print(f"s2.shape: {s2.shape}")
      # print(f"S1.shape: {S1.shape}")
      # print(f"S2.shape: {S2.shape}")
      
      try:
        # lp = LineProfiler()
        # lp_wrapper = lp(feature_gen)
        # lp_wrapper(s1, s2)
        # lp.print_stats(output_unit=1e-3)
        F = feature_gen(s1, s2)
        #F = feature_gen(t1, s1, t2, s2)
        #F = feature_gen(t1, S1, t2, S2)
      except Warning:
        num_features = 13
        # print(s1[:50])
        # print(s2[:50])
        print("Warning encountered..")
        # print(f"s1.shape: {s1.shape}")
        # print(f"s2.shape: {s2.shape}")
        # print(f"S1.shape: {S1.shape}")
        # print(f"S2.shape: {S2.shape}")
        print("*************************")
        F = np.zeros(num_features)
      F_avg.append(F)
    self.data_list.append((selected_label, np.mean(F_avg, axis=0)))


ref_label = 0

dataset = DatasetBuilder(ref_label=ref_label, num_patients=1)  
dataset.create_dataset_of_particular_stage(num_segs_chosen_per_patient=2)


# dataset = DatasetBuilder(ref_label=ref_label, num_patients=NUM_CHOSEN_PATIENTS)  
# dataset.create_dataset_of_particular_stage(num_segs_chosen_per_patient=50)
# print(f"saving clf{ref_label} dataset...")

# np.save(f"clf{ref_label}.npy", dataset.data_list)


print(f"Total time: {time.time()-start} seconds")