import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from model import config as C

from generators import getDataPairGenerator, getDataGenerator

PATH_TO_THIS_DIR = os.path.dirname(__file__)
PATH_TO_DATA_DIR = os.path.join(PATH_TO_THIS_DIR, '../data')
PATH_TO_DATASET = os.path.join(PATH_TO_DATA_DIR, 'furniture')

#datagenerator = getDataPairGenerator(PATH_TO_DATASET)
datagenerator = getDataGenerator(PATH_TO_DATASET)
print(datagenerator)

a = datagenerator.__len__()
print(str(a))

def datagen():
  for i in range(datagenerator.__len__()):
    yield datagenerator.__getitem__(i)

test_gen = datagen()

for batch in test_gen:
  print('a')
  batch = batch / 255

