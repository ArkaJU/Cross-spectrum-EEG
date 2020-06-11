import os
import numpy as np
import time
from extract import extract_anns, extract_data
from constants import *


patient_list = sorted(os.listdir(TRAIN_DATA_PATH))


#a dict is said to to be fit if it has atleast min_seg_per_stage_reqd segments of each sleep stage
class FitDictFinder:
  
  def __init__(self, min_seg_per_stage_reqd):
    self.min_seg_per_stage_reqd = min_seg_per_stage_reqd

  def is_dict_fit(self, eeg_dict):         #whether the patient has segments of all types of sleep stages
    len_dict = {}
    for label in eeg_dict.keys():
      if len(eeg_dict[label]) < self.min_seg_per_stage_reqd: return False
      len_dict[label] = len(eeg_dict[label])
    #print(len_dict)
    return True 

  def get_fit_dict_indices(self):
    fit_dict_indices = []
    
    for i in range(len(patient_list)):
      current_patient = patient_list[i]
      patient_ann = current_patient[:-4] + '-nsrr.xml'
      ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
      eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + current_patient, ann, onset, duration[-1])
      flag = self.is_dict_fit(eeg_dict)
      print(i, flag)
      if flag: 
        fit_dict_indices.append(i)
        print(fit_dict_indices)
    
    #return fit_dict_indices


class ReferenceBuilder:

  def __init__(self, num_patients, num_segs_chosen_per_patient_per_stage):
    self.num_patients = num_patients
    self.num_segs_chosen_per_patient_per_stage = num_segs_chosen_per_patient_per_stage 
    self.reference_segments = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]} 


  def get_ref_dicts(self):

    #indices of fit dicts, found beforehand, can use FitDictFinder class to get desired dicts
    fit_ref_list = [3, 10, 19, 21, 22, 26, 28, 33, 36, 38, 40, 42, 43, 51, 52, 53, 55, 56, 57, 60, 62,\
                    64, 70, 72, 73, 82, 83, 85, 87, 89, 90, 93, 95, 101, 103, 107, 109, 113, 115, 116,\
                    117, 123, 130, 131, 135, 138, 139, 141, 148, 155, 156, 157, 159]
    selected_ref_indices = list(np.random.choice(fit_ref_list, size=self.num_patients, replace=False))
    
    for i in selected_ref_indices:
      ref = patient_list[i]
      ann_ref = ref[:-4] + '-nsrr.xml'
      
      ann, onset, duration = extract_anns(TRAIN_ANN_PATH + ann_ref)
      eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + ref, ann, onset, duration[-1])
      
      print(ref)
      print(ann_ref)

      yield eeg_dict   


  def each_reference_data(self, eeg_dict):
    for label in range(NUM_SLEEP_STAGES):
      seg_indices = np.random.choice(len(eeg_dict[label]), size=self.num_segs_chosen_per_patient_per_stage, replace=False)
      print(label, seg_indices, len(eeg_dict[label]))
      for s in seg_indices:
        self.reference_segments[label].append(eeg_dict[label][s])
    print()


  def build_refs(self):
    fit_eeg_dict_generator = self.get_ref_dicts()

    for i, ref_dict in enumerate(fit_eeg_dict_generator):
      print(f"Patient {i+1}:")
      self.each_reference_data(ref_dict)

start = time.time()
# fdf = FitDictFinder(min_seg_per_stage_reqd=1)
# fdf.get_fit_dict_indices()
num_patients=10
num_segs_chosen_per_patient_per_stage=1
ref_builder = ReferenceBuilder(num_patients=num_patients, num_segs_chosen_per_patient_per_stage=num_segs_chosen_per_patient_per_stage)
ref_builder.build_refs()
np.save(f'reference_segments_{num_patients*num_segs_chosen_per_patient_per_stage}.npy', ref_builder.reference_segments)
# print(f"Total time taken for reference building:{start-time.time()}")