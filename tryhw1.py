from uwnet import *

# Number of operations for the covnet architecture:

# For any convolutional layer, the number of operations will be
# equal to : (height * width * channels * size * size * filters)/stride

# Thus, for the 4 convolutional layers, we have:
#     1st Convolution: (32 * 32 * 3 * 3 * 3 * 8)/1 = 221184
#     2nd Convolution: (16 * 16 * 8 * 16 * 3 * 3)/1 = 294912
#     3rd Convolution: (8 * 8 * 16 * 32 * 3 * 3)/1 = 294912
#     4th Convolution: (4 * 4 * 32 * 64 * 3 * 3)/1 = 294912

# For the fully connected layer, we have:
#     1st Fully Connected: 256 * 10 = 2560

# Thus, total number of operations = 221184 + (294912 * 3) + 2560 
#                                  = 1108480
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

# 1104960 operations
def fully_connected():
    l = [
        make_connected_layer(3072, 342),
        make_activation_layer(RELU),
        make_connected_layer(342, 128),
        make_activation_layer(RELU),
        make_connected_layer(128, 64), 
        make_activation_layer(RELU),
        make_connected_layer(64, 32),
        make_activation_layer(RELU),
        make_connected_layer(32, 10),
        make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = fully_connected() #conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#
# Covnet benchmarks
# training accuracy: 0.6923800110816956
# test accuracy:     0.6413999795913696
#
# Fully connected benchmarks
# training accuracy: 0.5455800294876099
# test accuracy:     0.5060999989509583
#
# From the above, we can see that convolutions works better than fully connected layers
# I think this is becasue in convolutional layers, we are able to extract important features
# and then pass them forward for learning. 
# On the other hand, in a fully connected model, there is no prioritization of certain features
# Since the model has to consider all weights and parameters, learning becomes harder
# resulting in a worse performing model, as seen above. 
