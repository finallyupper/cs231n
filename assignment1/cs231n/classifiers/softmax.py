from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0] 
    num_classes = W.shape[1]

    for i in range(num_train):
      scores = X[i].dot(W) # shape = (N, C)
      scores -= np.max(scores)

      scores_exp = np.sum(np.exp(scores))
      correct_exp = np.exp(scores[y[i]]) 

      loss -= np.log(correct_exp / scores_exp)

      for j in range(num_classes):
        if j == y[i]:
          continue
        dW[:, j] += np.exp(scores[j])/scores_exp * X[i]  ############## 이해가 부족한 파트
      dW[:, y[i]] -= (scores_exp - correct_exp)/scores_exp * X[i] ###############

    loss /= num_train
    loss += (reg* np.sum(W*W))
 
    dW /= num_train
    dW += reg*2*W 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0] # X shape = (N, D)

    scores = X.dot(W) # shape = (N, C)
    scores -= np.max(scores)

    scores_exp = np.exp(scores)
    scores_expsum = np.sum(scores_exp, axis = 1)
    correct_exp = scores_exp[range(num_train), y]

    loss = correct_exp / scores_expsum
    loss = -np.sum(np.log(loss)) 

    temp = np.divide(scores_exp, scores_expsum.reshape(num_train, 1)) 
    temp[range(num_train), y] = -(scores_expsum - correct_exp) / scores_expsum
    
    dW = X.T.dot(temp) # (D, C)

    loss /= num_train
    loss += (reg* np.sum(W*W))
    dW/= num_train;  
    dW += reg*2*W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
