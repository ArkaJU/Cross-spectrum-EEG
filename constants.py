SLEEP_STAGES = {'Wake|0':0, 'Stage 1 sleep|1':1, 'Stage 2 sleep|2':2, 'Stage 3 sleep|3':3, 
                'Stage 4 sleep|4':4, 'REM sleep|5':5}
SLEEP_STAGES_INV = {v: k for k, v in SLEEP_STAGES.items()}
SAMPLE_RATE = 125    #fixed for shhs1
NUM_SLEEP_STAGES = 6   #fixed
#NUM_SEG_PROCESSED_PER_PATIENT = 36

TRAIN_DATA_PATH = r"/content/drive/My Drive/NSRR/Data/train/"
TRAIN_ANN_PATH = r"/content/drive/My Drive/NSRR/Annotations/train/"

TEST_DATA_PATH = r"/content/drive/My Drive/NSRR/Data/test/"
TEST_ANN_PATH = r"/content/drive/My Drive/NSRR/Annotations/test/"

NUM_CHOSEN_PATIENTS = 10  #num_patients->number of patients from which random segment  will be chosen
DURATION_OF_EACH_SEGMENT = 30 #seconds

NUM_FEATURES = 32

