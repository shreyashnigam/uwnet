from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? 
# How does it affect convergence? 
# How does it affect what magnitude of learning rate you can use? 
# Write down any observations from your experiments:

# Without Batch Normalization:
#     Learning Rate: 0.01
#         training accuracy: %f 0.4027400016784668
#         test accuracy:     %f 0.4052000045776367

# With Batch Normalization:
#     Learning Rate: 0.1
#         training accuracy: %f 0.451339989900589
#         test accuracy:     %f 0.4514999985694885
    
#     Learning Rate: 0.05
#         training accuracy: %f 0.5642799735069275
#         test accuracy:     %f 0.5485000014305115

#     Learning Rate: 0.03
#         training accuracy: %f 0.5504999756813049
#         test accuracy:     %f 0.5414000153541565
    
#     Learning Rate: 0.01
#         training accuracy: %f 0.5567799806594849
#         test accuracy:     %f 0.5461000204086304

# Training the covnet with Batch Normalization results in a better performing 
# model. As we can see above, with batch normalization, the training and test accuracy
# is always higher. With regards to convergance, I obeserved that covnet with batch normalization
# tended to converge faster. Finally, with Batch Normalization, I was able to use larger learning rates.
# This was thanks to normalization making training more stable.  
