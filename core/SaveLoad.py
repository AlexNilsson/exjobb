import sys
import numpy as np

def LoadWeights(model, weights, by_name = False):
  """ Routine for loading saved weights to a model"""
  try:
    current_weights = model.get_weights()
    model.load_weights(weights, by_name = by_name)
    loaded_weights = model.get_weights()
  
  except Exception as e:
    print('Failed to load saved weights: ')
    sys.exit(str(e))
  
  else:
    print('Weights loaded successfully!')

    if(np.array_equal(current_weights, loaded_weights)):
      print('...but identical to old weights!')
