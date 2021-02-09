# Semi-supervised Learning

## Introduction
Image labeling is one of the most expensive tasks in computer vision. Since deep neural networks require lots of data from which they learn, there was a need for utilizying unlabeled data in the training proccess. This problem gave birth to a new branch of deep learning called semi-supervised learning. There are many semi-supervised methods in existance and in this project we explored \pi - model as described in this paper [1].

## Dataset 
CIFAR-10

## Technologies
### Python
  - PyTorch
  - Matplotlib
  - NumPy
### Google Colab

## Project

### Validation of hyperparameters
The first step of our project was validation of hyperparameters, and my task, which I showcase in my repository, was finding the best learning rate. We were using an Adam optimizer and an exponential scheduler. I determined the best learning rate through the method of trial and error and came to a conclusion that the best learning rate is 0.001.
The main criteria for selection was accuracy on the test set as shown in the graphs below.
![Learning rate 0.0001](/main/hiperparams_validation/figures/lr_0.0001.png)
![Learning rate 0.001](/main/hiperparams_validation/figures/lr_0.001.png)
![Learning rate 0.01](/main/hiperparams_validation/figures/lr_0.01.png)



## Literature
[1]: https://arxiv.org/pdf/1610.02242.pdf
