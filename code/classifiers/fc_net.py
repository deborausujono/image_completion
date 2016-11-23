import numpy as np

from asgn2.layers import *
from asgn2.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.

  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    w1 = init_weights(hidden_dim, input_dim, weight_scale)
    b1 = init_bias(hidden_dim)

    w2 = init_weights(num_classes, hidden_dim, weight_scale)
    b2 = init_bias(num_classes)

    self.params = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    #scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    w1 = self.params['W1']
    b1 = self.params['b1']
    w2 = self.params['W2']
    b2 = self.params['b2']

    h1, cache_h1 = affine_relu_forward(X, w1, b1) # np.maximum(0, X.dot(W1) + b1)
    scores, cache_scores = affine_forward(h1, w2, b2) # h1.dot(W2) + b2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    #loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    loss += 0.5 * self.reg * np.sum(w1 * w1)
    loss += 0.5 * self.reg * np.sum(w2 * w2)

    dx2, dw2, db2 = affine_backward(dx, cache_scores)
    dw2 += self.reg * w2
    _, dw1, db1 = affine_relu_backward(dx2, cache_h1)
    dw1 += self.reg * w1
    grads = {'W1': dw1, 'b1': db1, 'W2': dw2, 'b2': db2}
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    input_dims = [input_dim] + hidden_dims
    output_dims = hidden_dims + [num_classes]
    #key_init_pairs = [('W', init_weights), ('b', init_bias)]
    #self.params = {k + str(i + 1): f(d2, d1, weight_scale)
    #               for i, (d1, d2) in enumerate(zip(d1s, d2s))
    #               for k, f in key_init_pairs}

    self.params = {}
    for i, (in_d, out_d) in enumerate(zip(input_dims, output_dims)):
      idx = str(i + 1)
      self.params['W' + idx] = init_weights(out_d, in_d, weight_scale)
      self.params['b' + idx] = init_bias(out_d)
      if self.use_batchnorm and i < self.num_layers - 1:
        self.params['gamma' + idx] = init_gamma(out_d)
        self.params['beta' + idx] = init_beta(out_d)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    #scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    cache_list = []
    if self.use_dropout: dropout_cache_list = []

    # Hidden layer params
    if self.use_batchnorm:
      params = [(self.params['W' + str(i)], self.params['b' + str(i)],
                 self.params['gamma' + str(i)], self.params['beta' + str(i)],
                 self.bn_params[i - 1])
                for i in range(1, self.num_layers)]
    else:
      params = [(self.params['W' + str(i)], self.params['b' + str(i)])
                for i in range(1, self.num_layers)]

    # Output layer params
    params.append((self.params['W' + str(self.num_layers)],
                   self.params['b' + str(self.num_layers)]))
    
    # Compute hidden layers
    h = X
    for param in params[:-1]:
      if self.use_batchnorm:
        h, cache = affine_bn_relu_forward(h, *param)
      else:
        h, cache = affine_relu_forward(h, *param)
      cache_list.append(cache)

      if self.use_dropout:
        h, dropout_cache =  dropout_forward(h, self.dropout_param)
        dropout_cache_list.append(dropout_cache)

    # Compute output layer
    w, b = params[-1]
    scores, cache = affine_forward(h, w, b)
    cache_list.append(cache)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    #loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # Find data loss
    data_loss, dx = softmax_loss(scores, y)

    # Find regularization loss
    w_params = np.array([param[0] for param in params])
    reg_loss = 0.5 * self.reg * np.sum([np.sum(a) for a in w_params * w_params])

    # Calculate total loss
    loss = data_loss + reg_loss

    # Calculate gradients
    i = self.num_layers
    dx, dw, db = affine_backward(dx, cache_list[i - 1])
    dw += self.reg * w_params[i - 1]
    grads = {'W' + str(i): dw, 'b' + str(i): db}

    for i in range(self.num_layers - 1, 0, -1):
      if self.use_dropout:
        dx = dropout_backward(dx, dropout_cache_list[i - 1])

      if self.use_batchnorm:
        dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, cache_list[i - 1])
        grads['gamma' + str(i)] = dgamma
        grads['beta' + str(i)] = dbeta
      else:
        dx, dw, db = affine_relu_backward(dx, cache_list[i - 1])

      dw += self.reg * w_params[i - 1]
      grads['W' + str(i)] = dw
      grads['b' + str(i)] = db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform, followed by a 
  batch normalization, followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  fc_out, fc_cache = affine_forward(x, w, b)
  bn_out, bn_cache = batchnorm_forward(fc_out, gamma, beta, bn_param)
  relu_out, relu_cache = relu_forward(bn_out)
  cache = (fc_cache, bn_cache, relu_cache)
  return relu_out, cache

def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the affine-bn-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  relu_dx = relu_backward(dout, relu_cache)
  bn_dx, dgamma, dbeta = batchnorm_backward(relu_dx, bn_cache)
  fc_dx, dw, db = affine_backward(bn_dx, fc_cache)
  return fc_dx, dw, db, dgamma, dbeta
