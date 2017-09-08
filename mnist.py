#!/usr/bin/python3

import numpy as np
from models import BasicNet, Softmax

def norm(v):
    return np.sqrt(np.sum(np.square(v)))

def train(model, next_batch):
    params = np.random.randn(model.n_params)
    model.param_load(params)
    for _ in range(200):
        inputs, labels = next_batch(128)
        _, outputs = model.step([None] * len(inputs), inputs)
        trajs = [
            [(i, o, l - o)]
            for i, o, l in zip(inputs, outputs, labels)
        ]
        update = model.param_gradient(trajs) * 300.0
        print("Update norm: %12.6f" % norm(update))
        params += update
        model.param_load(params)
    return params

def score(model, next_batch):
    inputs, labels = next_batch(5000)
    _, outputs = model.step([None] * len(inputs), inputs)
    correct = (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1))
    return np.mean(correct.astype(np.float32))

def run():
    import tensorflow.examples.tutorials.mnist as tf_mnist
    mnist = tf_mnist.input_data.read_data_sets(
        "/tmp/mnist-download",
        validation_size=0,
        one_hot=True
    )

    model = Softmax(BasicNet([28*28, "relu", 128, "relu", 10]))
    train(model, mnist.train.next_batch)

    s = score(model, mnist.test.next_batch)
    print("Accuracy on test: %.2f%%" % (100.0 * s))
    assert s > 0.9
    assert s < 1.0

if __name__ == "__main__":
    run()
