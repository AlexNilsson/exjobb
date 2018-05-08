import os, math
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

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

  def on_epoch_begin(self, epoch, logs):
    if epoch % C.PLOT_LATENT_SPACE_EVERY == 0:
      latentSpacePlot = self.plot()

      if self.save_plot:
        # reshape to get the right range when saving image to file
        figure_to_file = latentSpacePlot * 255
        figure_to_file = figure_to_file.astype('uint8')
        file_path = os.path.join(self.path_to_save_directory, '{}-{}.jpg'.format(epoch, self.save_name))
        cv.imwrite(file_path, figure_to_file)

  def on_epoch_end(self, epoch, logs):
    cv.destroyAllWindows()

class PlotLosses(Callback):
  def __init__(self, path_to_save_directory = './loss_plots'):

    self.path_to_save_directory = path_to_save_directory

    self.loss = []
    self.loss_rm20 = []

    self.fig = plt.figure()
    plt.ion()
    plt.show()

  def on_batch_end(self, batch, logs={}):

    self.loss.append(logs.get('loss'))
    self.loss_rm20.append(np.mean(self.loss[-20:]))

    loss = self.loss[-100:]
    loss_rm20 = self.loss_rm20[-100:]
    x = [i+1 for i in range(len(loss))]

    plt.clf()
    plt.semilogy(x, loss, label="Loss")
    plt.semilogy(x, loss_rm20, label="Loss rm20")
    plt.legend()
    plt.draw()
    plt.pause(0.001)

  def on_epoch_end(self, epoch, logs={}):
    if epoch % C.SAVE_LOSS_PLOT_FREQ == 0:
      plt.savefig( os.path.join(self.path_to_save_directory, 'loss_plot_{}.png'.format(epoch)))

class LogLosses(Callback):
  def __init__(self, log_file = './logs/loss.log'):
    self.logfile = log_file

  def on_epoch_end(self, epoch, logs={}):
    with open(self.logfile, "a+") as f:
      f.write("{}, {}\n".format(epoch, logs.get('loss'))) # Todo comma separated string of multiple values if applicable