import os
import time
import random
import numpy as np

from extract import extract_anns, extract_data
from constants import TEST_DATA_PATH, TEST_ANN_PATH, NUM_SLEEP_STAGES
from features import feature_gen

start = time.time()
patient_list = sorted(os.listdir(TEST_DATA_PATH))

class TestSetBuilder:

  def __init__(self, size): #size->number of patients from which segment bank will be made
    
    self.size = size
    self.testset_dict ={0:[],1:[],2:[],3:[],4:[],5:[]}
    self.ref_segments = np.load('/content/drive/My Drive/Cross-spectrum-EEG_2/datasets/ref-EEG/reference_segments_10_EEG_channel.npy', allow_pickle=True).reshape(-1, 1)[0][0]
    
    print(f"Refs loaded in {time.time()-start} seconds")


  def generate_features_with_ref_segments(self, selected_tuple, patient_no, mean, std):
      
    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    s1 = np.array(selected_segment)
    for ref in range(NUM_SLEEP_STAGES):            #extra loop over all ref labels
      F_avg = []
      for ref_segment in self.ref_segments[ref]:
        
        s2 = np.array(ref_segment)

        try:
          F = feature_gen(s1, s2, mean, std)
          F_avg.append(F)
        except Warning:
          print("Warning encountered")
          return

      self.testset_dict[ref].append((selected_label, np.mean(F_avg, axis=0)))

        
  def extract_test_segments_for_given_patient(self, patient_no):   #helper

    current_patient = patient_list[patient_no]  
    patient_ann = current_patient[:-4] + '-nsrr.xml'
    ann, onset, duration = extract_anns(TEST_ANN_PATH + patient_ann)
    preprocess = None  #no-preprocessing
    eeg_dict, info_dict = extract_data(TEST_DATA_PATH + current_patient, ann, onset,duration[-1], preprocess=preprocess)
    
    len_dict = {}
    for i in eeg_dict.keys(): 
      len_dict[i] = len(eeg_dict[i])

    selected_tuples=[]

    for i in eeg_dict.keys():
      if len_dict[i]!=0:
        #print(f"Label: {i}: {len_dict[i]}")
        if i==4:
          seg_indices = np.random.choice(len(eeg_dict[i]), min(60, len(eeg_dict[i])), replace=False)
          for j in seg_indices:
            selected_tuples.append((int(i), eeg_dict[i][j]))
        else:
          seg_indices = np.random.choice(len(eeg_dict[i]), min(7, len(eeg_dict[i])), replace=False)
          for j in seg_indices:
            selected_tuples.append((int(i),eeg_dict[i][j]))

        #print(seg_indices)
    
   #print()
          
    for t in selected_tuples:
      yield t


  def create_testset(self):      #main
  
    segs_global = []
    for p in range(self.size): 
      print(f"Patient ID: {p}")
      segs = []

      stats = np.load('/content/drive/My Drive/Cross-spectrum-EEG_2/datasets/stats/stats_test.npy', allow_pickle=True)
      mean = stats[p, 2]
      std =  stats[p, 3]
      segment_generator = self.extract_test_segments_for_given_patient(patient_no=p)
      for segment in segment_generator:
        self.generate_features_with_ref_segments(segment,p,mean,std)
        segs.append(segment[0])  #just appending the label for now, can save segment[1] for actual segment
      segs_global = segs_global + segs
      print(f"segs: {np.unique(segs, return_counts=True)}")  #for this patient only
      print(f"segs_global: {np.unique(segs_global, return_counts=True)}")
      print(f"Time taken so far: {time.time()-start} seconds")
      print("\n")
      np.save(f"/content/drive/My Drive/test_set_balanced.npy", dataset.testset_dict)

      
    print("########################")
    print("\n")
    
    print(f"segs_global: {np.unique(segs_global, return_counts=True)}")    #accumulating over all patients



dataset = TestSetBuilder(size=40)  
dataset.create_testset()

np.save(f"/content/drive/My Drive/test_set_balanced.npy", dataset.testset_dict)

print(f"Total time: {time.time()-start} seconds")