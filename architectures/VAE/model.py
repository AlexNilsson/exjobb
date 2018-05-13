import importlib

""" Setup """
# name of model to load: ./models/<name>
NAME = 'VAE_Radford2'

""" Load Model Config & Architecture """
REL_PATH_TO_MODELS_DIR = 'models'
REL_PATH_TO_MODEL_DIR = REL_PATH_TO_MODELS_DIR + '/' + NAME

config = importlib.import_module('.config', package = REL_PATH_TO_MODELS_DIR + '.' + NAME)
architecture = importlib.import_module('.architecture', package = REL_PATH_TO_MODELS_DIR + '.' + NAME)
