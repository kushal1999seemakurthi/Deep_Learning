# Cifar10 data classifer for 6 classes

Here is the [Reposity](https://github.com/kushal1999seemakurthi/Deep_Learning/tree/main/projects/cifar10_data_classifer) of this project.

## We here used Keras tuner to tune hyperperameters such as :

* #### Number of filters of conv2D layers
* #### Rate of Dropout
* #### Regularization Type and parameter
* #### Number of hidden units of Dense layer
* #### Learning Rate

And then trained A TF model to Classify the images from Cifar10 dataset. Dataset for training available from keras.datasets. It has roughly 60,000 image dataset containing 10 types of classes.
But, here considered only 6 of then have been considered.

Saved the trained model as [**final_model_1_reg.h5**](https://github.com/kushal1999seemakurthi/Deep_Learning/blob/main/projects/cifar10_data_classifer/final_model_1_reg.h5), and got 75% accuracy for the test data set it contains roughly 6000 images.
