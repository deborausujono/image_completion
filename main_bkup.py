import numpy as np
import theano
import theano.tensor as T
import lasagne

from code.data_utils import get_CIFAR10_data

# Set constants
labels = range(256)
num_labels = len(labels)

# Sequence Length
SEQ_LENGTH = 20

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 128 #512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 20 #128

NUM_LABELS = 256

# Load data
data = get_CIFAR10_data(mode=1)
"""
data is a dictionary with the following keys and value shapes:
X_val:  (1000, 4, 32, 32)
X_train:  (49000, 4, 32, 32)
X_test:  (1000, 4, 32, 32)
y_val:  (1000, 3, 32, 32)
y_train:  (49000, 3, 32, 32)
y_test:  (1000, 3, 32, 32)
"""

# Work with a small subset of data
num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

# Lasagne seed for reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

def gen_sequences(images):
	"""
	images: during training, complete images of size (batch_size, 3, 32, 32)
	
	Returns:
	(x_r, x_g, x_b, y_r, y_g, y_b)
	"""
	pass

def test(images):
	"""
	images: incomplete images of size (batch_size, 3, 32, 32)
	"""
	pass

def main():
	print 'Compiling network'

	# Input layers
	input_r = lasagne.layers.InputLayer(shape=(None, 15, 3))

	# First hidden layers
	h_forward_r1 = lasagne.layers.LSTMLayer(
        input_r, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

	# Second hidden layers
	h_forward_r2 = lasagne.layers.LSTMLayer(
        h_forward_r1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True)

	# Output layer
	output_r = lasagne.layers.DenseLayer(h_forward_r2, num_units=NUM_LABELS,
		W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

	# Theano tensor for the targets
    target_values = T.ivector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(output_r)

	# The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(output_r,trainable=True)

    # Define update rule for training
    updates = lasagne.updates.rmsprop(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    train = theano.function([input_r.input_var, target_values], cost, updates=updates, allow_input_downcast=True)

    # Probabilities over the next red pixels
    probs = theano.function([input_r.input_var], network_output, allow_input_downcast=True)

    print 'Training'

    # One image from training set to monitor training progress
    img = small_data['X_train'][0][np.newaxis, :,:,:]

    try:
    	# Since num_iters = 100 * num_epochs / BATCH_SIZE
        for itr in xrange(100 * num_epochs / BATCH_SIZE):
            test(img) # Generate text using the p^th character as the start. 
            
            avg_cost = 0;
            x, y = gen_sequences(train_feeder())                

            cost = train(x, y)
            print 'Epoch {} loss = {}'.format(itr * BATCH_SIZE / 100, cost)
                    
    except KeyboardInterrupt:
        pass
