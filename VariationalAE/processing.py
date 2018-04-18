import os
from PIL import Image

def preProcessImages(in_directory, out_directory, convert_to_grayscale=False, resize_to=(40,40)):
  pot_files = os.listdir(in_directory)
  n_files = len(pot_files)

  # For all files (assumed to be images) in_directory
  for i, f in enumerate(pot_files):
    path_to_file = os.path.join(in_directory, f)

    # if the file is a file and not a folder
    if os.path.isfile(path_to_file):
      im = Image.open(path_to_file)

      # Resize
      #if im.size != resize_to:
      im = im.resize(resize_to)
      print('resized: {}/{}'.format(i, n_files))

      # Convert to Grayscale
      if convert_to_grayscale:
        im = im.convert('L')

      # Save to_directory
      im.save(os.path.join(out_directory, f))
