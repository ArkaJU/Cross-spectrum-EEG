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
  #raw.set_annotations(annotations)

  return ann, onset, duration


#getting the EEG data for each patient->outputs labelwise dict of segments of EEG and another info dict
def extract_data(path, ann, onset):
  raw = mne.io.read_raw_edf(path, verbose=False)
  row_idx = np.array([2])      #taking 3rd channel(EEG)
  data = raw.get_data()
  data = data[row_idx, :]
  data = (data - np.min(data))/(np.max(data) - np.min(data))      #normalizing, using unique stats for each patient
  x = list(data.reshape(-1,))
  
  label_to_signal_mapping = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
  for i in range(len(onset)-1):           
    label = SLEEP_STAGES[ann[i]]
    #print(onset[i], onset[i+1], label)
    label_to_signal_mapping[label].append(x[SAMPLE_RATE * onset[i] : SAMPLE_RATE * onset[i+1]])

  info = {}
  for i in range(NUM_SLEEP_STAGES):
    info[SLEEP_STAGES_INV[i]] = len(label_to_signal_mapping[i])    #info regarding how many segments of each type in each patient's EEG

  return label_to_signal_mapping, info   

