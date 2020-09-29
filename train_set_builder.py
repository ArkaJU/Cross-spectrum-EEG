import os
import time
import random
import numpy as np

from features import feature_gen
from constants import NUM_SEG_CHOSEN_PER_PATIENT, NUM_CHOSEN_PATIENTS, DJ

start = time.time()
save_path = "/content/drive/My Drive"


class TrainSetBuilder:
  
  #@profile
  def __init__(self, ref_label): 

    self.base_dataset = np.load('/content/drive/My Drive/data_set_3stage.npy', allow_pickle=True)
    self.ref_segments = np.load('/content/drive/My Drive/Cross-spectrum-EEG/datasets/ref-EEG/reference_segments_3stage_5_EEG_channel_raw.npy', allow_pickle=True).reshape(-1, 1)[0][0]
    print(f"Number of references: {len(self.ref_segments[0])}")
    print(f"Base dataset and references loaded in {time.time()-start} seconds")
    
    self.ref_label = ref_label
    print(f"REF ID: {self.ref_label}")
    self.trainset_list = []                                 #list for trainset


  def generate_features_with_ref_segments(self, selected_tuple):
  
    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    s1 = np.array(selected_segment)
    F_avg = []

    # choice_indices = np.random.choice(len(self.ref_segments[self.ref_label]), 5, replace=False)
    # chosen_refs = [self.ref_segments[self.ref_label][i] for i in choice_indices]

    for ref_segment in self.ref_segments[self.ref_label]:
      
      s2 = np.array(ref_segment)

      try:
        F = feature_gen(s1, s2)
        F_avg.append(F)
      except Warning:
        print("Warning encountered..")
        print("*************************")
        return

    #print(np.mean(F_avg, axis=0).shape)
    self.trainset_list.append((selected_label, np.mean(F_avg, axis=0)))
    np.save(os.path.join(save_path, f"dj6_60f_3stage", f"clf{self.ref_label}.npy"), self.trainset_list)


  #@profile
  def create(self):      #main
    for i, selected_tuple in enumerate(self.base_dataset):
      self.generate_features_with_ref_segments(selected_tuple)

      if (i+1)%200==0: 
        print(f"{i+1}: Time taken so far is {time.time()-start} seconds")

    print(f"{i+1}: Time taken so far is {time.time()-start} seconds")
#************************************************************************************************************


ref_label = 1
train_set = TrainSetBuilder(ref_label=ref_label)  
train_set.create()
np.save(os.path.join(save_path, "dj6_60f_3stage", f"clf{ref_label}.npy"), train_set.trainset_list)