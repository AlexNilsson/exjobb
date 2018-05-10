#!/usr/bin/python
from PIL import Image
import os, sys

path = '.'
resize_to = 720

for f in os.listdir(path):
  if os.path.splitext(f)[1] != '.py':
    file_path = os.path.join(path, f)
    if os.path.isfile(file_path):
      try:
        im = Image.open(file_path)
        imResize = im.resize((resize_to,resize_to), Image.ANTIALIAS)
        imResize.save(file_path)
        print('resizing {} to {}x{}px'.format(f, resize_to, resize_to))
      except Exception as e:
        print(str(e))