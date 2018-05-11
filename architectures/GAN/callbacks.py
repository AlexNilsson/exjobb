import os, math
import cv2 as cv

import matplotlib.pyplot as plt

from keras.callbacks import Callback
from core.Visualizer import Visualizer

class PlotLatentSpaceProgress(Callback):
  def __init__(self, model, config, tiling=15, img_size = 720, max_dist_from_mean = 1, show_plot = True, save_plot = True, path_to_save_directory = './epoch_plots', save_name = 'image'):
    self.config = config

    channels = 3 if C.COLOR_MODE == 'rgb' else 1

    self.plot = lambda : Visualizer(config).plotLatentSpace2D(model, tiling = tiling, img_size = img_size, max_dist_from_mean = max_dist_from_mean, show_plot = show_plot, channels = channels )
    self.save_plot = save_plot
    self.path_to_save_directory = path_to_save_directory
    self.save_name = save_name

  def on_epoch_begin(self, epoch, logs):
    C = self.config

    if epoch % C.PLOT_LATENT_SPACE_EVERY == 0:
      latentSpacePlot = self.plot()

      if self.save_plot:
        # reshape to get the right range when saving image to file
        figure_to_file = (latentSpacePlot + 1) * 0.5 * 255
        figure_to_file = figure_to_file.astype('uint8')
        file_path = os.path.join(self.path_to_save_directory, '{}-{}.jpg'.format(epoch, self.save_name))
        cv.imwrite(file_path, figure_to_file)

  def on_epoch_end(self, epoch, logs):
    cv.destroyAllWindows()

class PlotLosses(Callback):
  def __init__(self, config, path_to_save_directory = './loss_plots'):
    self.config = config
    self.path_to_save_directory = path_to_save_directory

    self.d_loss = []
    self.g_loss = []
    self.baseline = []
    self.base_line_value = -math.log(0.5)

    self.fig = plt.figure()
    plt.ion()
    plt.show()

  def on_epoch_end(self, epoch, logs={}):
    C = self.config

    self.d_loss.append(logs.get('d_loss'))
    self.g_loss.append(logs.get('g_loss'))
    self.baseline.append(self.base_line_value)

    x = [i+1 for i in range(len(self.d_loss))]

    plt.clf()
    plt.plot(x, self.d_loss, label="Discriminator Loss")
    plt.plot(x, self.g_loss, label="Generator Loss")
    plt.plot(x, self.baseline, label="50/50 Line")
    plt.legend()
    plt.draw()
    plt.pause(0.001)

    if epoch % C.PLOT_LOSS_EVERY == 0:
      plt.savefig( os.path.join(self.path_to_save_directory, 'loss_plot_{}.png'.format(epoch)))
