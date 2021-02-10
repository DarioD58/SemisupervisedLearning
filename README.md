# Semi-supervised Learning

## Introduction
Image labeling is one of the most expensive tasks in computer vision. Since deep neural networks require lots of data from which they learn, there was a need for utilizing unlabeled data in the training process. This problem gave birth to a new branch of deep learning called semi-supervised learning. There are many semi-supervised methods in existence and in this project we explored &Pi;-model as described in this paper [1].

## Dataset 
CIFAR-10

## Technologies
### Python
  - PyTorch
  - Matplotlib
  - NumPy
### Google Colab

## Project

**Architecture** [2]
  - Convolutional layer
  - Max pooling layer
  - Convolutional layer
  - Max pooling layer
  - Dropout layer
  - Fully connected layer
  - Fully connected layer
  - Fully connected output layer

### Validation of hyperparameters
The first step of our project was the validation of hyperparameters, and my task, which I showcase in my repository, was finding the best learning rate. We were using an Adam optimizer and an exponential scheduler. I determined the best learning rate through the method of trial and error and concluded that the best learning rate is 0.001.
The main criteria for selection was accuracy on the test set as shown in the graphs below.

<details>
<summary>Graphs: Test set accuracy</summary>
<br>
  
<figcaption>Learning rate 0.0001</figcaption>
![Learning rate 0.0001](/hiperparams_validation/figures/lr_0.0001.png "Learning rate 0.0001")

<figcaption>Learning rate 0.001</figcaption>
![Learning rate 0.001](/hiperparams_validation/figures/lr_0.001.png "Learning rate 0.001")

<figcaption>Learning rate 0.01</figcaption>
![Learning rate 0.01](/hiperparams_validation/figures/lr_0.01.png "Learning rate 0.01")
</details>

### &Pi;-model
The main part of our project was the implementation of &Pi;-model in PyTorch. The first major discovery of our experiments was the importance of mixing labeled and unlabeled data before training. Mixing data rather than training on labeled data first and then on unlabeled data resulted in a much smoother loss function and consequently a much steadier rise of accuracy as showcased.
<details>
<summary>Graphs: Mixed and separated data</summary>
<br>

<figcaption>Separated labeled and unlabeled data</figcaption>
![Separated data](/plots/training_plot_separated_data_4000.png "Separated labeled and unlabeled data")

<figcaption>Mixed labeled and unlabeled data</figcaption>  
![Mixed data](/plots/training_plot_connected_data_4000.png "Mixed labeled and unlabeled data")
</details>

After this discovery, we proceeded with our implementation of the &Pi;-model as is described in the paper [1]. An important part of our training process was using a ramp-up scheduler for the first 50 epochs of learning and a ramp-down scheduler for the final 30 epochs. 

**Hyperparameters**
  - Learning rate: 0.001
  - Number of epochs: 80
  - Batch size: 50
  
**Stochastic augmentation**
  - Random horizontal flip
  - Random rotation

We trained and tested our model on 250, 1000, and 4000 labeled data and, also, trained and tested our model on labeled data only to serve as a benchmark. We obtained the following results: 

<figcaption>Results</figcaption>
![Results](/plots/accuracy_pi_model.png "Results")

Although our model showed only a slight improvement with 4000 labeled data compared to labeled data only, the graphs reveal something interesting.

<details>
<summary>Graph: 4000 labeled images accuracy</summary>
<br>

<figcaption>Accuracy and loss on 4000 labeled data only</figcaption>
![4000 only](/plots/training_plot_4000_only.png "Accuracy and loss on 4000 labeled data only")

<figcaption>Accuracy and loss on 4000 labeled data and 46000 unlabeled data</figcaption>
![4000 only](/plots/training_plot4000.png "Accuracy and loss on 4000 labeled data and 46000 unlabeled data")
</details>

As we can see there is a jump in accuracy after the 50<sup>th</sup> epoch which corresponds with the biggest value of the unsupervised component in the loss function. This is a good indicator that our model was indeed learning from unlabeled data and that was the goal all along. Further improvements can be made by increasing the number of epochs thus allowing the model to learn even more from the unlabeled data. Moreover, a more powerful model architecture would yield better results.


## Literature
> [1]: Laine, S. and Aila, T., “Temporal Ensembling for Semi-Supervised Learning”,arXiv e-prints, 2016
> [2]: Architecture is derived from this https://dlunizg.github.io/lab2/ laboratory exercise on Faculty of Electrical Engineering and Computing.
