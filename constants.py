SLEEP_STAGES = {'Wake|0':0, 
                'Stage 1 sleep|1':1, 
                'Stage 2 sleep|2':2, 
                'Stage 3 sleep|3':3, 
                'Stage 4 sleep|4':3, 
                'REM sleep|5':4}
#SLEEP_STAGES_INV = {v: k for k, v in SLEEP_STAGES.items()}
SAMPLE_RATE = 125    #fixed for shhs1
NUM_SLEEP_STAGES = 5   #fixed
DURATION_OF_EACH_SEGMENT = 30 #seconds

TRAIN_DATA_PATH = r"/content/drive/My Drive/NSRR2/Data/train/"
TRAIN_ANN_PATH = r"/content/drive/My Drive/NSRR2/Annotations/train/"

TEST_DATA_PATH = r"/content/drive/My Drive/NSRR/Data/test/"
TEST_ANN_PATH = r"/content/drive/My Drive/NSRR/Annotations/test/"


NUM_CHOSEN_PATIENTS = 200  
NUM_SEG_CHOSEN_PER_PATIENT = 200
DJ = 6