from easydict import EasyDict as edict
import os
from misc.utils import up

# init
__C_LDN = edict()
cfg_data = __C_LDN

# Image sizes
__C_LDN.STD_SIZE = (1024, 1024)
__C_LDN.TRAIN_SIZE = (1024, 1024) # 2D tuple or 1D scalar

# Filepaths
root_folder = up(os.path.realpath(__file__), 5)
data_folder = os.path.join(root_folder, 'data')
__C_LDN.DATA_PATH = os.path.join(data_folder, 'london')
__C_LDN.TRAIN_JSON_FILEPATH = os.path.join(__C_LDN.DATA_PATH, 'train.json')
__C_LDN.TEST_JSON_FILEPATH = os.path.join(__C_LDN.DATA_PATH, 'test.json')

__C_LDN.MEAN_STD = (
    [0.4063, 0.4952, 0.3289],
    [0.1718, 0.1585, 0.1603]
)

__C_LDN.LABEL_FACTOR = 1
__C_LDN.LOG_PARA = 100.
__C_LDN.RESUME_MODEL = ''  # model path
__C_LDN.TRAIN_BATCH_SIZE = 1  # imgs
__C_LDN.VAL_BATCH_SIZE = 1  # must be 1


