# Semi-supervised Learning

## Introduction
Image labeling is one of the most expensive tasks in computer vision. Since deep neural networks require lots of data from which they learn, there was a need for utilizying unlabeled data in the training proccess. This problem gave birth to a new branch of deep learning called semi-supervised learning. There are many semi-supervised methods in existance and in this project we explored &Pi;-model as described in this paper [1].

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

<details>
<summary>Graphs: Test set accuracy</summary>
<br>
  
![Learning rate 0.0001](/hiperparams_validation/figures/lr_0.0001.png "Learning rate 0.0001")

<figcaption>Learning rate 0.0001</figcaption>

![Learning rate 0.001](/hiperparams_validation/figures/lr_0.001.png "Learning rate 0.001")

<figcaption>Learning rate 0.001</figcaption>

![Learning rate 0.01](/hiperparams_validation/figures/lr_0.01.png "Learning rate 0.01")

<figcaption>Learning rate 0.01</figcaption>
</details>

### &Pi;-model
The main part of our project was the implementation of &Pi;-model in PyTorch. The first major discovery of our experiments was the importance of mixing labeled and unlabeled data before training. Mixing data rather than training on labeled data first and then on unlabeled data resulted in a much smooter loss function and consequntly much steadier rise of accuracy as showcased.
<details>
<summary>Graphs: Mixed and separated data</summary>
<br>
 Something
</details>

After this discovery we procceeded with our implementation of &Pi;-model as is described in paper [1]. We trained and tested our model on 250, 1000, and 4000 labeled data and , also, trained and tested our model on labeled data only to serve as a benchmark. We obtained the following results: 

Table here

Although, our model showed only slight improvement with 4000 labeled data compared to labeled data only, the graphs reveal something interesting.

<details>
<summary>Graph: 4000 labeled images accuracy</summary>
<br>
 Something
</details>




## Literature
[1]: Laine, S. and Aila, T., “Temporal Ensembling for Semi-Supervised Learning”,arXiv e-prints, 2016
