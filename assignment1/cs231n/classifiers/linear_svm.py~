import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0  
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    counter = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        counter += 1
        dW[j,:] += X[:,i]
    dW[y[i],:] += X[:,i] * counter * -1
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[1]
  scores = W.dot(X)
  #correct_scores = np.diagonal(scores[y])
  correct_scores = scores[y, range(len(y))]
  margins = np.maximum(0, scores - correct_scores + 1)
  margins[y, range(len(y))] = 0
  loss = np.sum(margins) / num_train
  loss += 0.5 * reg * np.sum(W * W)

  margins[margins > 0] = 1
  count = np.sum(margins, axis = 0)
  dW = margins.dot(X.T)
  # review
  # can it be fully vectorized ?
  for i in range(y.shape[0]):
    dW[y[i]] -= count[i] * X[:,i]
#  dW[y] -= count.reshape((1,count.shape[0])).dot(X.T)

  dW /= num_train

  return loss, dW
