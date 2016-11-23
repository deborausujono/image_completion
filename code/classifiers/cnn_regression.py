import numpy as np

from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *


class ThreeLayerConvNetRegression(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - l2

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(4, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, output_dim=(3, 32, 32), weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - output_dim: Tuple (C, H, W) giving size of output data
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    num_channels, input_height, input_width = input_dim
    w1 = init_filters(num_filters, num_channels, filter_size, filter_size, weight_scale)
    b1 = init_bias(num_filters)

    # Calculate input dim of the affine layer
    conv_stride = 1
    conv_pad = (filter_size - 1) / 2
    conv_out_height = 1 + (input_height + 2 * conv_pad - filter_size) / conv_stride
    conv_out_width = 1 + (input_width + 2 * conv_pad - filter_size) / conv_stride

    pool_height = 2
    pool_width = 2
    pool_stride = 2
    pool_out_height = (conv_out_height - pool_height)/pool_stride + 1
    pool_out_width = (conv_out_width - pool_width)/pool_stride + 1

    affine_input_dim = num_filters * pool_out_height * pool_out_width
    w2 = init_weights(hidden_dim, affine_input_dim, weight_scale)
    b2 = init_bias(hidden_dim)

    output_dim = np.prod(output_dim)
    w3 = init_weights(output_dim, hidden_dim, weight_scale)
    b3 = init_bias(output_dim)

    self.params = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2, 'W3': w3, 'b3': b3}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional regression
    network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    w1 = self.params['W1']
    b1 = self.params['b1']
    w2 = self.params['W2']
    b2 = self.params['b2']
    w3 = self.params['W3']
    b3 = self.params['b3']

    h1, cache_h1 = conv_relu_pool_forward(X, w1, b1, conv_param, pool_param)
    h2, cache_h2 = affine_relu_forward(h1, w2, b2)
    scores, cache_scores = affine_forward(h2, w3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    #loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx = l2_loss(scores, y)
    loss += 0.5 * self.reg * np.sum(w1 * w1)
    loss += 0.5 * self.reg * np.sum(w2 * w2)
    loss += 0.5 * self.reg * np.sum(w3 * w3)

    dx3, dw3, db3 = affine_backward(dx, cache_scores)
    dw3 += self.reg * w3
    dx2, dw2, db2 = affine_relu_backward(dx3, cache_h2)
    dw2 += self.reg * w2
    _, dw1, db1 = conv_relu_pool_backward(dx2, cache_h1)
    grads = {'W1': dw1, 'b1': db1, 'W2': dw2, 'b2': db2, 'W3': dw3, 'b3': db3}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads