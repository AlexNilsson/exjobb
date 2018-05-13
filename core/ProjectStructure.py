import os
from datetime import datetime

def _timestamp_id(prefix='ID'):
  t = datetime.now()
  return "{}--{}-{}-{}--{}-{}-{}".format(prefix, t.year, t.month, t.day, t.hour, t.minute, t.second)

class TrainingSessionOutputDirectory:
  def __init__(self, root, title):
    # Main directory with a unique name
    THIS_ID = _timestamp_id(title)
    self.PATH_TO_OUT_DIRECTORY = os.path.join(root, 'out', THIS_ID)
    os.makedirs(self.PATH_TO_OUT_DIRECTORY, exist_ok = True)

    # sub directories
    self.PATH_TO_LOGS_DIRECTORY = os.path.join(self.PATH_TO_OUT_DIRECTORY, 'logs')
    os.makedirs(self.PATH_TO_LOGS_DIRECTORY, exist_ok = True)

    self.PATH_TO_EPOCH_PLOTS = os.path.join(self.PATH_TO_OUT_DIRECTORY, 'epoch_plots')
    os.makedirs(self.PATH_TO_EPOCH_PLOTS, exist_ok = True)

    self.PATH_TO_LOSS_PLOTS = os.path.join(self.PATH_TO_OUT_DIRECTORY, 'loss_plots')
    os.makedirs(self.PATH_TO_LOSS_PLOTS, exist_ok = True)

    self.PATH_TO_SAVED_WEIGHTS = os.path.join(self.PATH_TO_OUT_DIRECTORY, 'saved_weights')
    os.makedirs(self.PATH_TO_SAVED_WEIGHTS, exist_ok = True)

    self.PATH_TO_SAVED_MODELS = os.path.join(self.PATH_TO_OUT_DIRECTORY, 'saved_models')
    os.makedirs(self.PATH_TO_SAVED_MODELS, exist_ok = True)

class InferenceSessionOutputDirectory:
  def __init__(self, root, title):
    # Main directory with a unique name
    THIS_ID = _timestamp_id('lsexp_{}'.format(title))
    self.PATH_TO_OUT_DIRECTORY = os.path.join(root, 'out_exp', THIS_ID)
    os.makedirs(self.PATH_TO_OUT_DIRECTORY, exist_ok = True)