import os
import cv2 as cv

from keras.callbacks import Callback
from visualization import plotLatentSpace2D

from model import config as C

class PlotLatentSpaceProgress(Callback):
  def __init__(self, model, tiling=15, img_size = 720, max_dist_from_mean = 1, show_plot = True, save_plot = True, path_to_save_directory = './epoch_plots', save_name = 'image'):

    channels = 3 if C.COLOR_MODE == 'rgb' else 1

    self.plot = lambda : plotLatentSpace2D(model, tiling = tiling, img_size = img_size, max_dist_from_mean = max_dist_from_mean, show_plot = show_plot, channels = channels )
    self.save_plot = save_plot
    self.path_to_save_directory = path_to_save_directory
    self.save_name = save_name

  def on_epoch_begin(self, epoch, logs=None):
    if epoch % C.PLOT_LATENT_SPACE_EVERY == 0:
      latentSpacePlot = self.plot()

      if self.save_plot:
        # reshape to get the right range when saving image to file
        figure_to_file = latentSpacePlot * 255
        figure_to_file = figure_to_file.astype('uint8')
        file_path = os.path.join(self.path_to_save_directory, '{}-{}.jpg'.format(epoch, self.save_name))
        cv.imwrite(file_path, figure_to_file)

  def on_epoch_end(self, epoch, logs=None):
    cv.destroyAllWindows()
