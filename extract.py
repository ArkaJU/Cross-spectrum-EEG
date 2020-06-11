import mne
from xml.etree import ElementTree
import numpy as np
from constants import *

#for reading annotations from xml file
def extract_anns(path):
  def parse_nsrr_annotations(file_path):
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    return root


  def parse_nsrr_scored_events(file_path):
      root = parse_nsrr_annotations(file_path)
      scored_events = root.find('ScoredEvents').getchildren()
      return scored_events


  def parse_nsrr_sleep_stages(file_path):
      events = parse_nsrr_scored_events(file_path)
      sleep_stages = [event for event in events if event.find('EventType').text == 'Stages|Stages']
      return sleep_stages

  def nsrr_sleep_stage_components(xml_file_path):
      stages_elements = parse_nsrr_sleep_stages(xml_file_path)

      stage = [elem.find('EventConcept').text for elem in stages_elements]
      onset = [elem.find('Start').text for elem in stages_elements]
      duration = [elem.find('Duration').text for elem in stages_elements]

      onset = np.array(onset, dtype=float)
      duration = np.array(duration, dtype=float)

      return stage, onset, duration

  stage, onset, duration = nsrr_sleep_stage_components(path)
  annotations = mne.Annotations(onset=onset, duration=duration, description=stage, orig_time=None)

  ann = annotations.description
  onset = onset.astype(np.int)


  return ann, onset, duration


#getting the EEG data for each patient->outputs labelwise dict of segments of EEG and another info dict
#@profile
def extract_data(path, ann, onset, last_seg_duration, preprocess='std'):
  raw = mne.io.read_raw_edf(path, verbose=False)
  
  try:
    data = raw.get_data(picks=['EEG(sec)'])    #taking 3rd channel(EEG)
  except ValueError:
    print(f"Channel error at: {path}")
    data = raw.get_data(picks=['EEG2']) 

  if preprocess=='norm': data = (data - np.min(data))/(np.max(data) - np.min(data))      #normalizing, using unique stats for each patient
  if preprocess=='std': data = (data - np.mean(data))/np.std(data)

  x = data.tolist()[0]
  
  eeg_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
  # for i in range(len(onset)-1):           
  #   label = SLEEP_STAGES[ann[i]]
  #   #print(onset[i], onset[i+1], label)
  #   eeg_dict[label].append(x[SAMPLE_RATE * onset[i] : SAMPLE_RATE * onset[i+1]])
  for i in range(len(onset)-1):           
    label = SLEEP_STAGES[ann[i]]
    #print(onset[i], onset[i+1], label)
    for j in range(onset[i], onset[i+1], DURATION_OF_EACH_SEGMENT):
      eeg_dict[label].append(x[j*SAMPLE_RATE:(j+DURATION_OF_EACH_SEGMENT)*SAMPLE_RATE])
  
  #TAKING CARE OF THE LAST SEGMENT 
  #try-except for the weird 'Unscored-9' stage in some patients
  try:
    last_label = SLEEP_STAGES[ann[-1]]
    for j in range(onset[-1], onset[-1]+int(last_seg_duration), DURATION_OF_EACH_SEGMENT):
        eeg_dict[last_label].append(x[j*SAMPLE_RATE : (j+DURATION_OF_EACH_SEGMENT)*SAMPLE_RATE])
  except KeyError:
    print("KeyError")
    pass
    
  info_dict = {}
  for i in range(NUM_SLEEP_STAGES):
    info_dict[SLEEP_STAGES_INV[i]] = len(eeg_dict[i])    #info regarding how many segments of each type in each patient's EEG

  return eeg_dict, info_dict   

