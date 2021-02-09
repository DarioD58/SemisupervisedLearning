import matplotlib.pyplot as plt
import numpy as np
import data_load
import torch
import torchvision
from conv import ConvolutionalModel
import data_load
from torch import nn


def evaluate(net, type):
    """
    Performs the evaluation of the current performance of a
    given convolutional network. It can perform the evaluation on 
    both training and testing sets. Standard evaluation metrics are
    calcualted such as, accuracy and confusion matrix.
    Parameters
    ----------
    net: ConvolutionalModel
        ConvNet whose performance needs to be evaluated.
    type: bool
        True if eval is made on testing set, false otherwise
    Return
    ------
    loss
        Current loss on the chosen set
    accuracy
        Current acc on the chosen set
    """
    device = torch.device('cuda')
    f = open("C:/Users/Dario/Desktop/Projekt/Hiperparams/ProjektR/Kod/results/rs_data5.txt", "a+")
    net.eval()
    total = 0
    correct = 0
    confMatrix = np.zeros((10, 10), int)
    lossFunc = nn.CrossEntropyLoss()
    accLoss = 0
    if type:
        with torch.no_grad():
            for data in data_load.testloader:
                images, labels = data
                images = images.to(device=device)
                labels = labels.to(device=device)

                output = net.forward(images)
                loss = lossFunc(output, labels)
                _, predictions = torch.max(output.data, 1)
                total += labels.size(0)
                accLoss += loss.item()
                correct += (predictions == labels).sum().item()
                for j in range(labels.size(0)):
                    confMatrix[predictions[j], labels[j]] += 1
    else:
        with torch.no_grad():
            for data in data_load.trainloader:
                images, labels = data
                images = images.to(device=device)
                labels = labels.to(device=device)

                output = net.forward(images)
                loss = lossFunc(output, labels)
                _, predictions = torch.max(output.data, 1)
                total += labels.size(0)
                accLoss += loss.item()
                correct += (predictions == labels).sum().item()
                for j in range(labels.size(0)):
                    confMatrix[predictions[j], labels[j]] += 1

    print("Accuracy of the neural network on CIFAR_10 is: %.2f %%" %((correct/total)*100))
    #print(data_load.classes)
    f.write("Accuracy: " + str(((correct/total)*100)) + '\n')
    f.write(str(data_load.classes) + '\n')
    f.write(str(confMatrix) + '\n')
    #print(confMatrix)
    prec, recall = specificMetrics(confMatrix)
    f.write(str(prec) + '\n')
    f.write(str(recall) + '\n')
    f.close()
    return (accLoss/(total/data_load.trainloader.batch_size)), (correct/total)

def specificMetrics(confMatrix):
    """
    Calculates precision and recall from a given confusion
    matrix and returns calculated metrics.
    Parameters
    ----------
    confMatrix: n x n numpy array
        Made from the predictions and true labels of a
        given set of data
    Return
    ------
    precc
        Precision on all classes
    recal 
        Recall on all classes
    """
    precc = np.zeros(np.size(confMatrix, 0))
    recal = np.zeros(np.size(confMatrix, 0))
    for i in range(np.size(confMatrix, 0)):
        tp = 0
        fp = 0
        fn = 0
        for j in range(np.size(confMatrix, 0)):
            if i == j:
                tp += confMatrix[i, j]
            else:
                fn += confMatrix[j, i]
                fp += confMatrix[i, j]
            
        precc[i] += tp/(tp + fp)
        recal[i] += tp/(tp + fn)

    return precc, recal