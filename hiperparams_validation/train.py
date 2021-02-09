from conv import ConvolutionalModel
import torch.optim as optim
from torch import nn
import data_load
import torch
import evaluate
import plot
import torchvision
import torchvision.transforms as transforms
import math
from PIL import Image
from torchvision.datasets import CIFAR10
import numpy as np

#Simple data loading for usage in
#the rest of the training.
#Before using for the first time set download to
#true

trainTransform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))])

validTransform = transforms.Compose([    
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))])

class CifarDataset(CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img1 = img
        img2 = img

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target



trainSet = CifarDataset(root='./data', train=True,
                    download=False, transform=trainTransform)
testSet = CIFAR10(root='./data', train=False,
                    download=False, transform=validTransform)

classLabels = np.zeros(10)
counter = 0
for i in range(len(trainSet)):
    if counter == 4000:
        trainSet.targets[i] = -1
    else:
        target = trainSet.targets[i]
        if classLabels[target] < 400:
            classLabels[target] += 1
            counter += 1

print(classLabels)

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=50,
                                          shuffle=True, num_workers=0)


testLoader = torch.utils.data.DataLoader(testSet, batch_size=50,
                                         shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def update_w_t(epoch, num_epochs):
    if epoch < num_epochs:
        p = max(0.0, float(epoch) / float(num_epochs))
        p = 1.0 - p
        return math.exp(-p * p * 5.0)
    else:
        return 1.0

def trainNetwork():
    """Performs a standard procedure for training a neural network.
    Training progress after each learning epoch is evaluated in order to
    gain insigth into ConvNets continuous performance.
    Important notes
    ---------------
    Loss function: Cross entropy loss

    Optimizer: Adam
    
    Scheduler: ExponentialLR
    """
    SAVE_DIR = 'C:/Users/Dario/Desktop/Projekt/Hiperparams/ProjektR/Kod/figures'
    device = torch.device('cuda')
    



    epoch = 80
    epoch_rampup = 80
    epoch_rampdown = 50
    
    plot_data = {}
    plot_data['train_loss'] = []
    plot_data['valid_loss'] = []
    plot_data['train_acc'] = []
    plot_data['valid_acc'] = []
    plot_data['lr'] = []
    net = ConvolutionalModel(3, 16, 128, 10)
    net.train()
    net.to(device=device)
    f = open("C:/Users/Dario/Desktop/Projekt/Hiperparams/ProjektR/Kod/results/rs_data5.txt", "a+")
    f.close()
    lossFunc = nn.CrossEntropyLoss(ignore_index=-1)
    lossFuncUnsupervised = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler_rampup = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.05)
    scheduler_rampdown = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


    for e in range(epoch):

        accLoss = 0.0

        w = update_w_t(e, epoch_rampup)
        
        for i, data in enumerate(trainLoader, 0):
            input, input1, labels = data
            input = input.to(device=device)
            input1 = input1.to(device=device)
            labels = labels.to(device=device)

            optimizer.zero_grad()

            outputs_supervised = net.forward(input)
            outputs_unsupervised = net.forward(input1)

            loss = lossFunc(outputs_supervised, labels) + (w * lossFuncUnsupervised(outputs_supervised, outputs_unsupervised) / 10)
            loss.backward()
            optimizer.step()

            accLoss += loss.item()

            if i % 10 == 0:
                print("Epoch: %d, Iteration: %5d, Loss: %.3f" % ((e + 1), (i * 100), loss.item()))
                
    
        train_loss, train_acc = evaluate.evaluate(net, False)
        val_loss, val_acc = evaluate.evaluate(net, True)

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [val_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [val_acc]

        print("w(T) = %.3f" % w)

        #if e < epoch_rampup:
            #plot_data['lr'] += [scheduler_rampdown.get_last_lr()]
            #scheduler_rampup.step()
        #else:
        plot_data['lr'] += [scheduler_rampdown.get_last_lr()]
        scheduler_rampdown.step()


    plot.plot_training_progress(SAVE_DIR, plot_data)
    PATH = 'C:/Users/Dario/Desktop/Projekt/Hiperparams/ProjektR/Kod/CIFAR_10/cifar_netAdam1.pth'
    torch.save(net.state_dict(), PATH)


trainNetwork()


