#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import time
import random
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[5]:


config = dict(
    saved_path="saved_models/child.pt",
    best_saved_path = "saved_models/random_best.pt",
    lr=0.001, 
    EPOCHS = 3,
    BATCH_SIZE = 32,
    IMAGE_SIZE = 132,
    TRAIN_VALID_SPLIT = 0.2,
    device=device,
    SEED = 42,
    pin_memory=True,
    num_workers=3,
    USE_AMP = True,
    channels_last=False)


# In[6]:


random.seed(config['SEED'])
# If you or any of the libraries you are using rely on NumPy, you can seed the global NumPy RNG 
np.random.seed(config['SEED'])
# Prevent RNG for CPU and GPU using torch
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed(config['SEED'])
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


# In[7]:


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


train_data = torchvision.datasets.CIFAR10(root='../Images', train=True, download=True, transform=data_transforms['test'])
test_data = torchvision.datasets.CIFAR10(root='../Images', train=False, download=True, transform=data_transforms['test'])
valid_data = test_data

train_dl = torch.utils.data.DataLoader(train_data, batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])


test_dl = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])
valid_dl = torch.utils.data.DataLoader(valid_data, batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])

def train_model(model,criterion,optimizer,num_epochs=10):

    since = time.time()                                            
    batch_ct = 0
    example_ct = 0
    best_acc = 0.3
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        run_corrects = 0
        #Training
        model.train()
        for x,y in train_dl: #BS=32 ([BS,3,224,224], [BS,4])            
            if config['channels_last']:
                x = x.to(config['device'], memory_format=torch.channels_last) #CHW --> #HWC
            else:
                x = x.to(config['device'])
            y = y.to(config['device']) #CHW --> #HWC
            
            
            
            optimizer.zero_grad()
            #optimizer.zero_grad(set_to_none=True)
            ######################################################################
            
            train_logits = model(x) #Input = [BS,3,224,224] (Image) -- Model --> [BS,4] (Output Scores)
            
            _, train_preds = torch.max(train_logits, 1)
            train_loss = criterion(train_logits,y)
            train_loss = criterion(train_logits,y)
            run_corrects += torch.sum(train_preds == y.data)
            
            train_loss.backward() # Backpropagation this is where your W_gradient
            loss=train_loss

            optimizer.step() # W_new = W_old - LR * W_gradient 
            example_ct += len(x) 
            batch_ct += 1
            if ((batch_ct + 1) % 400) == 0:
                train_log(loss, example_ct, epoch)
            ########################################################################
        
        #validation
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        # Disable gradient calculation for validation or inference using torch.no_rad()
        with torch.no_grad():
            for x,y in valid_dl:
                if config['channels_last']:
                    x = x.to(config['device'], memory_format=torch.channels_last) #CHW --> #HWC
                else:
                    x = x.to(config['device'])
                y = y.to(config['device'])
                valid_logits = model(x)
                _, valid_preds = torch.max(valid_logits, 1)
                valid_loss = criterion(valid_logits,y)
                running_loss += valid_loss.item() * x.size(0)
                running_corrects += torch.sum(valid_preds == y.data)
                total += y.size(0)
            
        epoch_loss = running_loss / len(valid_data)
        epoch_acc = running_corrects.double() / len(valid_data)
        train_acc = run_corrects.double() / len(train_data)
        print("Train Accuracy",train_acc.cpu())
        print("Validation Loss is {}".format(epoch_loss))
        print("Validation Accuracy is {}\n".format(epoch_acc.cpu()))
        if epoch_acc.cpu()>best_acc:
            print('One of the best validation accuracy found.\n')
            torch.save(model.state_dict(), config['best_saved_path'])
            best_acc = epoch_acc.cpu()

            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    torch.save(model.state_dict(), config['saved_path'])

    
def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
print('Training Child Model:')
efficientnet = models.efficientnet_b0(pretrained = True)
efficientnet.classifier[1] = nn.Linear(in_features = 1280, out_features = 10, bias = True)
model = efficientnet
criterion = nn.CrossEntropyLoss()
model = model.to(config['device'])
optimizer = optim.Adam(model.parameters(),lr=config['lr'])

train_model(model,criterion,optimizer,num_epochs=8)

def evaluation(model,test_dl):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    preds = []
    pred_labels = []
    labels = []

    with torch.no_grad():
                for x,y in test_dl:
                    x = x.to(config['device'])
                    y = y.to(config['device']) #CHW --> #HWC
                    valid_logits = model(x)
                    predict_prob = F.softmax(valid_logits)
                    _,predictions = predict_prob.max(1)
                    predictions = predictions.to('cpu')

                    _, valid_preds = torch.max(valid_logits, 1)
                    valid_loss = criterion(valid_logits,y)
                    running_loss += valid_loss.item() * x.size(0)
                    running_corrects += torch.sum(valid_preds == y.data)
                    total += y.size(0)
                    predict_prob = predict_prob.to('cpu')

                    pred_labels.extend(list(predictions.numpy()))
                    preds.extend(list(predict_prob.numpy()))
                    y = y.to('cpu')
                    labels.extend(list(y.numpy()))

    epoch_loss = running_loss / len(test_data)
    epoch_acc = running_corrects.double() / len(test_data)
    print("Test Loss is {}".format(epoch_loss))
    print("Test Accuracy is {}".format(epoch_acc.cpu()))
    return np.array(labels),np.array(pred_labels),np.array(preds)
    

labels, pred_labels,preds = evaluation(model, test_dl)
#print(metrics.precision_recall_fscore_support(np.array(labels), np.array(pred_labels)))
print('\nAUROC:')
print(metrics.roc_auc_score(np.array(labels), np.array(preds), multi_class='ovr'))
print()
print(metrics.classification_report(labels,pred_labels))
print('\n\n')

config = dict(
    saved_path="saved_models/parent.pt",
    best_saved_path = "saved_models/parent_best.pt",
    lr=0.001, 
    EPOCHS = 3,
    BATCH_SIZE = 32,
    IMAGE_SIZE = 132,
    TRAIN_VALID_SPLIT = 0.2,
    device=device,
    SEED = 42,
    pin_memory=True,
    num_workers=3,
    USE_AMP = True,
    channels_last=False)

print('Training Parent Model:')

efficientnet = models.efficientnet_b4(pretrained = True)
efficientnet.classifier[1] = nn.Linear(in_features = 1792, out_features = 10, bias = True)
model = efficientnet
criterion = nn.CrossEntropyLoss()
model = model.to(config['device'])
optimizer = optim.Adam(model.parameters(),lr=config['lr'])

train_model(model,criterion,optimizer,num_epochs=8)