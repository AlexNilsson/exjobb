import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))

from core.Generators import Generators
from model import config as C

data_generator = Generators(C).getDataGenerator('C:/Users/alex/Documents/GitHub/exjobb/data/windows')

for a in range(300):
  batch = next(data_generator)
  print(batch.shape)