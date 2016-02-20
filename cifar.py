import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from fuel.datasets.cifar10 import CIFAR10


def load_dataset():
    # loading the data from CIFAR10 and dividing it into 
    # training, validation and test sets
    
    cifar_train = CIFAR10(("train",), subset=slice(None, 40000))
    cifar_validation = CIFAR10(("train",), subset=slice(40000, None))
    cifar_test = CIFAR10(("test",))

    X_train = (cifar_train.data_sources[0] / 255.0).astype(np.single)
    Y_train = cifar_train.data_sources[1].ravel()
    X_val = (cifar_validation.data_sources[0] / 255.0).astype(np.single)
    Y_val = cifar_validation.data_sources[1].ravel()
    X_test = (cifar_test.data_sources[0] / 255.0).astype(np.single)
    Y_test = cifar_test.data_sources[1].ravel()

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def build_net(input_var=None, batchsize=100):
    # the architecture of the network

    network = lasagne.layers.InputLayer(shape=(batchsize, 3, 32, 32), 
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(network, num_filters=128, 
                                        filter_size=(5, 5), 
                                        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, 
                                        filter_size=(5, 5),
                                        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=0.5), 
                                        num_units=256, 
                                        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=0.5), 
                                        num_units=10, 
                                        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    # partitioning the input data into minibatches

    assert len(inputs) == len(targets)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(num_epochs=175, batchsize=100):
    # both the number of epochs and the size of a minibatch 
    # were found experimentally

    print "Loading data..."
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset()

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print "Building model and compiling functions..."

    network = build_net(input_var, batchsize)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)

    learning_rate = 0.01
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, Y_train, batchsize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, Y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        if (epoch + 1) % 20 == 0:
            # decreasing learning rate after each 20 epochs
            learning_rate /= 2.0
            updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)
            train_fn = theano.function([input_var, target_var], loss, updates=updates)

    test_err = 0
    test_acc = 0
    test_batches = 0

    for batch in iterate_minibatches(X_test, Y_test, batchsize, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


    # saving the results to a .txt file

    results_file = open("results.txt", "w")
    results = []
    labels = []

    for batch in iterate_minibatches(X_test, Y_test, batchsize, shuffle=False):
        inputs, targets = batch
        labels += list(targets)
        outputs = lasagne.layers.get_output(network, inputs=inputs, deterministic=True)
        predictions = T.argmax(outputs, axis=1)
        results += list(predictions.eval())

    for result, label in zip(results, labels):
        results_file.write(str(result) + " " + str(label) + "\n")


if __name__ == "__main__":
    main()

