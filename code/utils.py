import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from tensorflow.core.framework import summary_pb2
import matplotlib
import matplotlib.pyplot as plt
import itertools

def add_summary(tf_writer, tag, raw_value, global_step):
  value = summary_pb2.Summary.Value(tag=tag, simple_value=raw_value)
  summary = summary_pb2.Summary(value=[value])
  tf_writer.add_summary(summary, global_step)


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, init_lr):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = init_lr * (0.1 ** (epoch // 30))
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr
  return lr

def mean_average_precision(outputs, targets, num_classes=101):
  maps = np.zeros((num_classes, ))
  for c in range(num_classes):
      target = (targets == c)
      output = outputs[:, c]
      maps[c] = average_precision_score(target, output)
  #preds = np.argmax(outputs, axis=1)
  #conf_matrix = confusion_matrix(preds, targets)
  return np.mean(maps)*100.0

def conf_matrix(outputs, targets, num_classes=101):
  preds = np.argmax(outputs, axis=1)
  conf_matrix = confusion_matrix(targets, preds)
  return conf_matrix

def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          image_name='confusion_matrix.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontproperties=matplotlib.font_manager.FontProperties(size=8))
    plt.yticks(tick_marks, classes, fontproperties=matplotlib.font_manager.FontProperties(size=8))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, cm[i, j],
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black",
    #             fontproperties=matplotlib.font_manager.FontProperties(size='x-small') )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(image_name)