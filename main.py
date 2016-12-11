import numpy as np
import theano
import theano.tensor as T
import lasagne

from code.data_utils import get_CIFAR10_data

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

def load_data(small_data=False, verbose=False):
    """
    Output: A dictionary with the following keys and value shapes
    if small_data is True:
        X_train (100, 32, 32, 4)
        y_train (100, 32, 32, 3)
        X_val (10, 32, 32, 4)
        y_val (10, 32, 32, 3)
    else (small_data is False):
        X_train (49000, 32, 32, 4)
        y_train (49000, 32, 32, 3)
        X_val (1000, 32, 32, 4)
        y_val (1000, 32, 32, 3)
        X_test (1000, 32, 32, 4)
        y_test (1000, 32, 32, 3)
    """
    print 'Loading data'
    data = get_CIFAR10_data(mode=1)
    if small_data:
        num_train = 100
        data = {
          'X_train': data['X_train'][:num_train],
          'y_train': data['y_train'][:num_train],
          'X_val': data['X_val'][:(num_train / 10)],
          'y_val': data['y_val'][:(num_train / 10)],
        }

    if verbose:
        for k, v in data.iteritems():
            print '%s: ' % k, v.shape

    return data

def make_minibatch(X_train, y_train, batch_size):
    # Make a minibatch of training data
    num_train = X_train.shape[0]
    batch_mask = np.random.choice(num_train, batch_size)
    X_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]

    return X_batch, y_batch

def data_feeder(data, images_per_batch, min_seq_len, max_seq_len):
    """
    Output:
    - X (batch_size, max_seq_len, num_features): observed rows of pixels so far
    - X_mask (batch_size, max_seq_len): required in lasagne for variable seq_len
    - y (batch_size, num_features): next row of pixels

    More details:
    Each image is expanded into H - min_seq_len (sequence, target) pairs, so
    batch_size = images_per_batch * (H - min_seq_len).

    A sequence is seq = (f_1, ..., f_k), where k = min_seq_len, ..., max_seq_len
    and f_t = (pixelR_1, pixelG_1, pixelB_1, ..., pixelB_H). In other words,
    it is the RGB pixels of rows 1 to k.

    The target is f_{k + 1} or the RGB pixels of the next row after k.

    Since the sequence length k varies from min_seq_len to max_seq_len, lasagne
    LSTMLayer requires a mask array of ones for col <= k and zero otherwise.

    A minibatch of data is batch_size pairs of sequences, their masks, and
    their targets.
    """
    # Randomly get a minibatch of data
    X_train, y_train = data['X_train'], data['y_train']
    _, images = make_minibatch(X_train, y_train, images_per_batch)

    _, H, W, num_channels = images.shape # (images_per_batch, H, W, num_channels)
    num_repeats = H - min_seq_len
    batch_size = images_per_batch * num_repeats
    num_features = W * num_channels

    # Flatten width and channel dimensions, so the features at each time step become
    # f_t = (pixelR_1, pixelG_1, pixelB_1, ..., pixelB_H)
    images = images.reshape(images_per_batch, H, num_features)
    
    # Create labels (the next row of pixels)
    y = images[:, min_seq_len:, :] # (images_per_batch, H - min_seq_len, num_features)
    y = y.reshape(batch_size, num_features)

    # Repeat each image num_repeats times
    images = np.repeat(images, num_repeats, axis=0)

    # Create feature sequences (observed rows of pixels)
    X = images[:, :-1, :]

    # Create mask (zeros out future rows of pixels)
    X_mask = np.ones((num_repeats, max_seq_len))       # (27, 31)
    X_mask = np.tril(X_mask, min_seq_len - 1)
    X_mask = np.array(list(X_mask) * images_per_batch) # (batch_size, 31)

    return (X, X_mask, y)

def test_one(img, predict):
    H, W, num_channels = img.shape
    missing_start = int(H * 0.25)  # 8
    missing_end = int(H * 0.75)    # 24

    mask = img[:, :, -1]
    img = img[:, :, :-1] * mask

    assert np.all(img[missing_start:missing_end, missing_start:missing_end, :] == 0)
    print img[missing_start:missing_end, missing_start:missing_end, :]

    """
    predict() needs X of shape (num_data, max_seq_len, num_features)
    """

    seq = img[:missing_start, :, :]
    seq = seq.reshape(missing_start, W * num_channels)
    new_px = predict(X, X_mask)

def main(min_seq_len, max_seq_len, H, W, num_channels, images_per_batch, seed,
         use_small_data):
    # Set seed for reproducibility if provided
    if seed is not None:
        lasagne.random.set_rng(np.random.RandomState(1))

    data = load_data(small_data=use_small_data, verbose=True)

    # Define network
    print 'Compiling network'
    num_train = data['X_train'].shape[0]
    num_features = W * num_channels
    batch_size = images_per_batch * (W - min_seq_len)

    # Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, max_seq_len, num_features))
    mask_in = lasagne.layers.InputLayer(shape=(None, max_seq_len))

    # First hidden layer
    l_forward1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, mask_input=mask_in, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    # Second hidden layer
    l_forward2 = lasagne.layers.LSTMLayer(
        l_forward1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True)

    # Output layer
    l_out = lasagne.layers.DenseLayer(l_forward2, num_units=num_features,
        W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.rectify)

    # Matrix of target pixels (batch_size, num_features)
    target_output = T.imatrix('target_output')

    # Get output from l_out (batch_size, num_features)
    network_output = lasagne.layers.get_output(l_out)

    # Compute mean-squared error loss
    loss = T.mean(np.square(network_output - target_output))

    # Compute gradients
    params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.rmsprop(loss, params, LEARNING_RATE)

    # Define Theano helper functions
    print 'Compiling functions'
    # Loss
    train = theano.function([l_in.input_var, mask_in.input_var, target_output],
        loss, updates=updates, allow_input_downcast=True)

    # Predict next row of pixels
    predict = theano.function([l_in.input_var, mask_in.input_var],
        network_output, allow_input_downcast=True)

    # Train network
    print 'Training'

    #img = small_data['X_train'][0]#[np.newaxis, :,:,:]
    try:
        num_iters = num_train * NUM_EPOCHS / images_per_batch
        for itr in xrange(num_iters):
            # Check training progress
            #test_one(img, predict)
            
            X, X_mask, y = data_feeder(data, images_per_batch, min_seq_len, max_seq_len)
            new_loss = train(X, X_mask, y)
            print 'Epoch {} loss = {}'.format(itr * images_per_batch / num_train, new_loss)
                    
    except KeyboardInterrupt:
        pass

    # Test network
    #test(data['X_test'])

if __name__ == '__main__':
    min_seq_len = 5
    max_seq_len = 31
    H = 32
    W = 32
    num_channels = 3
    images_per_batch = 20
    seed = None
    use_small_data = True

    main(min_seq_len, max_seq_len, H, W, num_channels, images_per_batch, seed,
         use_small_data)