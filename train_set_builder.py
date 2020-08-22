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

    self.base_dataset = np.load('/content/drive/My Drive/data_set2.npy', allow_pickle=True)
    self.ref_segments = np.load('/content/Cross-spectrum-EEG/datasets/ref-EEG/reference_segments_5_EEG_channel.npy', allow_pickle=True).reshape(-1, 1)[0][0]
    print(f"Number of references: {len(self.ref_segments[0])}")
    print(f"Base dataset and references loaded in {time.time()-start} seconds")
    
    self.ref_label = ref_label
    print(f"REF ID: {self.ref_label}")
    self.trainset_list = []                                 #list for trainset

  #@profile
  def generate_features_with_ref_segments(self, selected_tuple, mean, std):
  #def generate_features_with_ref_segments(self, selected_tuple, mx, mn):

    selected_label = selected_tuple[0]
    selected_segment = selected_tuple[1]
    s1 = np.array(selected_segment)
    F_avg = []

    for ref_segment in self.ref_segments[self.ref_label]:
      
      s2 = np.array(ref_segment)

      try:
        F = feature_gen(s1, s2, mean, std)
        #F = feature_gen(s1, s2, mx, mn)
        F_avg.append(F)
      except Warning:
        print("Warning encountered..")
        print("*************************")
        return

    #print(np.mean(F_avg, axis=0).shape)
    self.trainset_list.append((selected_label, np.mean(F_avg, axis=0)))
    np.save(os.path.join(save_path, f"ref{len(self.ref_segments[0])}_dj{DJ}", f"clf{self.ref_label}_2.npy"), self.trainset_list)


  #@profile
  def create(self):      #main
    stats = np.load('/content/Cross-spectrum-EEG/datasets/stats/stats2.npy', allow_pickle=True)
    mean = stats[0, 2]
    std =  stats[0, 3]
    # mx = stats[0, 0]
    # mn = stats[0, 1]
    for i, selected_tuple in enumerate(self.base_dataset):
      self.generate_features_with_ref_segments(selected_tuple, mean, std)   
      #self.generate_features_with_ref_segments(selected_tuple, mx, mn)     
      if (i+1)%NUM_SEG_CHOSEN_PER_PATIENT==0:
        p = (i+1)//NUM_SEG_CHOSEN_PER_PATIENT

        if p==NUM_CHOSEN_PATIENTS: break

        mean = stats[p, 2]
        std =  stats[p, 3]
        # mx = stats[p, 0]
        # mn = stats[p, 1]
        print(f"{i+1}: Time taken so far is {time.time()-start} seconds")

    print(f"{i+1}: Time taken so far is {time.time()-start} seconds")
#************************************************************************************************************


ref_label = 1
train_set = TrainSetBuilder(ref_label=ref_label)  
train_set.create()
np.save(os.path.join(save_path, f"ref5_dj{DJ}", f"clf{ref_label}_2.npy"), train_set.trainset_list)