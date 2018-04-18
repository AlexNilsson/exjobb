import os
import cv2 as cv

from keras.callbacks import Callback
from visualization import plotLatentSpace2D

class PlotLatentSpaceProgress(Callback):
  def __init__(self, model, tiling=15, tile_size = 40, zoom = 1, show_plot=True, save_plot=True, path_to_save_directory='./epoch_plots', save_name='image'):
    self.plot = lambda : plotLatentSpace2D(model, tiling = tiling, tile_size = tile_size, zoom = zoom, show_plot=True )
    self.save_plot = save_plot
    self.path_to_save_directory = path_to_save_directory
    self.save_name = save_name

  def on_epoch_begin(self, epoch, logs):

    latentSpacePlot = self.plot()

    if self.save_plot:
      # reshape to get the right range when saving image to file
      figure_to_file = latentSpacePlot * 255
      figure_to_file = figure_to_file.astype('uint8')
      file_path = os.path.join(self.path_to_save_directory, '{}-'.format(epoch), self.save_name, '.jpg')
      cv.imwrite(file_path, figure_to_file)

  def on_epoch_end(self, epoch, logs):
    cv.destroyAllWindows()
