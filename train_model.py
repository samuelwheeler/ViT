import ViT_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import os
import time
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import MultiStepLR


# set hyperparameters and initial conditions
batch_size = 512
image_size = (32,32)
patch_size = (8,8)
channels = 3
dim = patch_size[0]*patch_size[1]*channels
numblocks = 16
hidden_dim = dim
heads = 12
#dropout = 0.1
state_path = 'ViT_model_state'
epochs = 100





# device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define model:
model = ViT_model.ViT_Model(image_size = image_size, patch_size = patch_size, dim = dim, hidden_dim = hidden_dim, numblocks = numblocks)
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
starting_epoch = 0

try:
   state = torch.load(state_path, map_location = device)
   model.load_state_dict(state['model_state_dict'])
   optimizer.load_state_dict(state['optimizer_state_dict'])
   starting_epoch = state['epoch']
   model.to(device)
   optimizer.to(device)
except:
    print('No state found')





transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def train(model, device, epochs, trainloader, testloader, optimizer, start_epoch, verbose = False):

    start_time = time.time()
    #model = model.to(device)    
    criterion = nn.CrossEntropyLoss()
    #lambda1 = lambda epoch: 0.89**(2*epoch)
    scheduler = MultiStepLR(optimizer, milestones=[20*n for n in range(1,10)],gamma =0.5)
    train_accs = np.zeros(epochs)
    test_accs = np.zeros(epochs)
    learning_rates = np.zeros(epochs)
   
    for epoch in range(epochs):
        
        lr = optimizer.param_groups[0]["lr"]
        print(f'Learning Rate: {lr}')
        learning_rates[epoch] = lr
        train_correct = 0
        train_total = 0    
        for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
            if torch.cuda.is_available():
                data, target = data.to(device), target.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)
            if batch_idx%100 == 0 and verbose:
                print(f'Loss: {loss.item()}')
        scheduler.step()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                # calculate outputs by running images through the network
                model.eval()
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        train_acc, test_acc = train_correct/train_total, test_correct/test_total
        train_accs[epoch + start_epoch] = train_acc
        test_accs[epoch + start_epoch] = test_acc
        '''if epoch >= 2 and False:
            if test_accs[epoch] - test_accs[epoch-1] < 0.01:
                lr = lr * 0.75
                for g in optimizer.param_groups:
                    g['lr'] = lr'''
        print(f'Epoch: {epoch + 1 + start_epoch}, Train Acc: {train_acc}, Test Acc: {test_acc}')
    total_time = time.time() - start_time
    return train_accs, test_accs, learning_rates


training_history = None
try:
    training_history = pd.read_csv('ViT_training_results')
except:
   training_history = None
   print('No training history found')



train_accs, test_accs, info = train(model = model, device = device, epochs = epochs, trainloader = trainloader, testloader = testloader, optimizer = optimizer, start_epoch = starting_epoch)
df = pd.DataFrame({'train_accs':train_accs, 'test_accs':test_accs})

if training_history is not None:
    training_history = training_history.append(df).reset_index(drop = True)
    training_history.to_csv('ViT_training_results')
else:
    df.to_csv('ViT_training_results')

state = {'epoch': starting_epoch + 100, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
torch.save(state, state_path)
