import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0  
  
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = W.dot(X)
  num_train = X.shape[1]
  num_class = W.shape[0]
  for i in xrange(num_train):
    score = scores[:,i]
    score -=  np.max(score)
    correct_score = score[y[i]]
    loss += -correct_score + np.log(np.sum(np.exp(score)))
    score = np.exp(score)
    score /= np.sum(score)
    for j in xrange(num_class):
      dW[j] += score[j] * X[:,i]
    dW[y[i]] -= X[:,i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[1]
  num_class = W.shape[0]
  scores = W.dot(X)
  scores -= np.max(scores, axis = 0)
  correct_scores = scores[y, range(len(y))]
  scores = np.exp(scores)
  loss = np.sum(-correct_scores + np.log(np.sum(scores, axis = 0)))
  #loss = np.sum(-np.log(np.exp(correct_scores)/np.sum(scores, axis = 0)))

  scores /= np.sum(scores,axis = 0)
  scores[y, range(len(y))] -= 1
  dW = scores.dot(X.T)
  #for i in xrange(y.shape[0]):
  #  dW[y[i],:] -= X[:,i]
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
