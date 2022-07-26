# Cifar10 data classifer for 6 classes

Here is the [Repository](https://github.com/kushal1999seemakurthi/Deep_Learning/tree/main/projects/cifar10_data_classifer) of this project.

## [Keras Tuner](https://keras.io/keras_tuner/)
 
 > KerasTuner is an easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search. Easily configure your search space with a define-by-run syntax, then leverage one of the available search algorithms to find the best hyperparameter values for your models. KerasTuner comes with Bayesian Optimization, Hyperband, and Random Search algorithms built-in, and is also designed to be easy for researchers to extend in order to experiment with new search algorithms. (content from the documentation)

## Aim:
 
 Using Keras tuner one can fibe tune hyper parameters as mentioned above. Here we are fine tuning prameters like:
 - Number of filters of conv2D layers
 - Rate of Dropout
 - Regularization Type and parameter
 - Number of hidden units of Dense layer
 - Learning Rate

## Description
 
Trained A TF model to Classify the images from Cifar10 dataset. Dataset for training available from keras.datasets. It has roughly 60,000 image dataset containing 10 types of classes.
 
But, here considered only 6 of the classes for the classification use case.

Saved the trained model as [**final_model_1_reg.h5**](https://github.com/kushal1999seemakurthi/Deep_Learning/blob/main/projects/cifar10_data_classifer/final_model_1_reg.h5), and got 75% accuracy for the test data set it contains roughly 6000 images.
