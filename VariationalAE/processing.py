import os
from PIL import Image

def preProcessImages(in_directory, out_directory, convert_to_grayscale=False, resize_to=(40,40)):
  # For all files (assumed to be images) in_directory
  for f in os.listdir(in_directory):
    if os.path.isfile(f):
      im = Image.open( os.path.join(in_directory, f))

      # Resize
      im.resize(resize_to)

      # Convert to Grayscale
      if convert_to_grayscale:
        im = im.convert('L')

      # Save to_directory
      im.save(os.path.join(out_directory, f))
