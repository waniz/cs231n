import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        incorrect_classes = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                incorrect_classes += 1
                loss += margin
                dW[:, j] = dW[:, j] + X[i, :]

        dW[:, y[i]] = dW[:, y[i]] - X[i, :] * incorrect_classes

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]
    scores = np.dot(X, W)
    correct_class_scores = np.choose(y, scores.T)

    # Need to remove correct class scores as we dont calculate loss/margin for those.
    mask = np.ones(scores.shape, dtype=bool)
    mask[range(scores.shape[0]), y] = False
    scores_ = scores[mask].reshape(scores.shape[0], scores.shape[1] - 1)

    # Calculate our margins all at once.
    margin = scores_ - correct_class_scores[..., np.newaxis] + 1

    # Only add margin to our loss if it's greater than 0, let's make
    # negative margins =0 so they dont change our loss.
    margin[margin < 0] = 0

    # Average our data loss over the size of batch and add reg. term to the loss.
    loss = np.sum(margin) / num_train
    loss += reg * np.sum(W * W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    original_margin = scores - correct_class_scores[..., np.newaxis] + 1

    # Mask to identiy where the margin is greater than 0 (all we care about for gradient).
    pos_margin_mask = (original_margin > 0).astype(float)

    # Count how many times >0 for each image but dont count correct class hence -1
    sum_margin = pos_margin_mask.sum(1) - 1

    # Make the correct class margin be negative total of how many > 0
    pos_margin_mask[range(pos_margin_mask.shape[0]), y] = -sum_margin

    # Now calculate our gradient.
    dW = np.dot(X.T, pos_margin_mask)

    # Average over batch and add regularisation derivative.
    dW = dW / num_train + 2 * reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
