import os, shutil, argparse

""" clearTrainingSession 'PATH/TO/FOLDER' -n 2 """

#PATH_TO_THIS_DIR = os.path.dirname(__file__) 

def removeEmptySessions( session_directory, min_epoch_plots=2 ):
  """ Removes training session folders containg less than <min_epoch_plots> epoch plots from the directory"""
  # For all sub-directories at this location
  for d in os.listdir( session_directory ):
    
    # A potential sub-directory containg the results of a specific model training session
    SUB_TRAIN_SESS_DIR = os.path.join( session_directory, d)
    if os.path.isdir(SUB_TRAIN_SESS_DIR):

      # Remove the folder if it contains less than <min_epoch_plots> epoch plots
      PATH_TO_EPOCH_PLOTS = os.path.join(SUB_TRAIN_SESS_DIR, 'epoch_plots')
      if os.path.exists(PATH_TO_EPOCH_PLOTS) == False or len(os.listdir(PATH_TO_EPOCH_PLOTS)) < min_epoch_plots:
        print('Remove dir: {}'.format(SUB_TRAIN_SESS_DIR))
        shutil.rmtree(SUB_TRAIN_SESS_DIR, ignore_errors=True)


# Parse command line input
parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str, help="target directory to clear sessions in")
parser.add_argument("-n", "--number_of_plots", type=int, help="Min number of plots required to keep the folder")
args = parser.parse_args()

# Handle Defaults
number_of_plots = args.number_of_plots if args.number_of_plots != None else 2

# Execute commands
removeEmptySessions(args.directory, min_epoch_plots = number_of_plots)

