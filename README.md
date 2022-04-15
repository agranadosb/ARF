# ARF Project

X-ray Chest Classification Task using ResNet, Inception and Transformed based.

## Author

Alejandro Granados Bañuls <algrabau@inf.upv.es>

## Description
In this project we are going to build three different deep learning
architectures for the X-Ray Chest task of kaggle ([X-Ray Chest Task](
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)). This
task consists on make a classification of chest x-ray images on healthy and
with Pneumonia. The dataset has 5856 samples. We  are going to create the
architectures from scratch to test how good is PyTorch  for building deep
neural networks and for datasets managements. The three  architectures we are
going to build are ResNet, Inception and Transformer based.

## Objectives

The objectives that will be accomplished for the first review will be marked as
bold.

 - **1: Collect the data from kaggle repository.**
 - **2: Create a dataset manager using PyTorch functionalities for data
management. This manager will take care of providing the images without the
need to load them all in memory. There will be three managers, one for each
split (train, validation and test).**
 - **3: Create ResNet model.**
 - 4: Create Inception and Transformer models.
 - 5: Train and test the three architectures using the dataset management created on
point 1.
 - 6: Get the results and analyze them, comparing the three architectures.
 - 7: Recap all results and analyze how good is PyTorch for implementing deep
neural networks and how well is each architecture for Chest X-Ray task.
