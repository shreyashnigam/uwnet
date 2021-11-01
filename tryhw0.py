# HW 0 
# Answer to Questions in the Spec

# To get 97% accuracy on MNIST data: 
#     1. Change iters (number of iterations) from 5000 to 8000
#     2. Change rate (learning rate) from 0.01 to 0.05
#     3. Change m (the model) from a softmax_model() to a neural_net()

# This gives:
#     - Training Accuracy: 98.31833243370056%
#     - Test Accuracy: 96.78000211715698%

# Compared to MNIST, the CIFAR dataset had significantly worse accuracy. 
# Using the default hyperparameters, I saw the following accuracies
#     - Training Accuracy: 38.14600110054016%
#     - Test Accuracy: 35.429999232292175

# Making the changes I made in my model for MNIST, and then benchmarking CIFAR, we get
#     - Training Accuracy: 27.616000175476074
#     - Test Accuracy: 26.30000114440918

# Thus, making the same changes I made for MNIST, affect the results for CIFAR negatively
# as the model performance gets significantly worse.
#--------------------------------------------------------------------------------------------

from uwnet import *

mnist = 1

inputs = 784 if mnist else 3072

def softmax_model():
    l = [make_connected_layer(inputs, 10),
        make_activation_layer(SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(inputs, 32),
            make_activation_layer(RELU),
            make_connected_layer(32, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
if mnist:
    train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels")
    test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels")
else:
    train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
    test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 8000
rate = .05
momentum = .9
decay = .0

m = neural_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))
