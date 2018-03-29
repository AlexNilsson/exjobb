import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import itertools

def plotConfusionMatrix(test_labels, rounded_predictions, plot_labels=['Class_0','Class_1'], plot_title='Confusion Matrix' ):
  plt.figure()
  # Function from:
  # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
  def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
      """
      This function prints and plots the confusion matrix.
      Normalization can be applied by setting `normalize=True`.
      """
      if normalize:
          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
          print("Normalized confusion matrix")
      else:
          print('Confusion matrix, without normalization')

      print(cm)

      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      plt.title(title)
      plt.colorbar()
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=45)
      plt.yticks(tick_marks, classes)

      fmt = '.2f' if normalize else 'd'
      thresh = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')

  cm = confusion_matrix(test_labels, rounded_predictions)

  plot_confusion_matrix(cm, plot_labels, title=plot_title)


def plotImages(images, labels, rows=2, axis='Off', fontsize=14):

  cols = math.ceil(len(images) / rows)

  f, axarr = plt.subplots(rows, cols)

  if type(images[0]) is np.ndarray:
      images = np.array(images).astype(np.uint8)
      if (images.shape[-1] != 3):
        images = images.transpose((0,2,3,1))

  for i in range(rows):
    for j in range(cols):
      n = i*cols + j
      axarr[i,j].imshow(images[n])
      axarr[i,j].axis(axis)
      axarr[i,j].set_title(labels[n], fontsize=fontsize)
