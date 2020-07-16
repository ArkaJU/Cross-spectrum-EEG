import os
import time
import random
import numpy as np

from extract import extract_anns, extract_data
from constants import *
from features import feature_gen

start = time.time()

class TrainSetBuilder:
  
  #@profile
  def __init__(self, ref_label): 

    self.base_dataset = np.load('/content/drive/My Drive/data_set.npy', allow_pickle=True)
    self.ref_segments = np.load('/content/drive/My Drive/Cross-spectrum-EEG_2/datasets/reference_segments_10.npy', allow_pickle=True).reshape(-1, 1)[0][0]
    print(f"Number of references: {len(self.ref_segments[0])}")
    print(f"Base dataset and references loaded in {time.time()-start} seconds")
    
    self.ref_label = ref_label
    print(f"REF ID: {self.ref_label}")
    self.trainset_list = []                                 #list for trainset

  #@profile
  def generate_features_with_ref_segments(self, selected_tuple, subs_flag=False):

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
        #substitute_tuple = self.extract_random_segments_for_given_patient_during_warning(selected_label, patient_no)
        #self.generate_features_with_ref_segments(substitute_tuple, patient_no, subs_flag=True)    #recursively call this function till Warningless segment is found (in practice Warning is rarely encountered, hence more than 1 recursive call is extremely rare)
        return    #important, else the local s1(which is the source of Warning) will continue executing, thus calling the functions in except block again and again
    #print(f"Avg vector shape: {np.mean(F_avg, axis=0).shape}")  
    self.trainset_list.append((selected_label, np.mean(F_avg, axis=0)))
    np.save(f"/content/drive/My Drive/clf{self.ref_label}.npy", self.trainset_list)


  #@profile
  def create(self):      #main

    for i, selected_tuple in enumerate(self.base_dataset):
      if (i+1)%1000==0:
        print(f"{i+1}: Time taken so far is {time.time()-start} seconds")
      self.generate_features_with_ref_segments(selected_tuple)    
    
    print(f"{i+1}: Time taken so far is {time.time()-start} seconds")
#************************************************************************************************************


ref_label = 1
train_set = TrainSetBuilder(ref_label=ref_label)  
train_set.create()
np.save(f"/content/drive/My Drive/clf{ref_label}.npy", train_set.trainset_list)
