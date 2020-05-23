from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import random
#import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# Convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        # for layer one, separate convolution and relu step from maxpool and batch normalization
        # to extract convolutional filters
        self.layer1_conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=300,
                      kernel_size=(4, 19),
                      stride=1,
                      padding=0),  # padding is done in forward method along 1 dimension only
            nn.ReLU())

        self.layer1_process = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=(0,1)),
            nn.BatchNorm2d(300))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=300,
                      out_channels=200,
                      kernel_size=(1, 11),
                      stride=1,
                      padding=0),  # padding is done in forward method along 1 dimension only
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4), padding=(0,1)),
            nn.BatchNorm2d(200))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=200,
                      out_channels=200,
                      kernel_size=(1, 7),
                      stride=1,
                      padding=0),  # padding is done in forward method along 1 dimension only
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1,4), padding=(0,1)),
            nn.BatchNorm2d(200))

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.03))

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.03))

        self.layer6 = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=num_classes))#, #)
            #nn.Sigmoid()) # Output sigmoid + scalar in forward function


    def forward(self, input):
        # run all layers on input data
        # add dummy dimension to input (for num channels=1)
        input = torch.unsqueeze(input, 1)

        # Run convolutional layers
        input = F.pad(input, (9, 9), mode='constant', value=0) # padding - last dimension goes first
        out = self.layer1_conv(input)
        out = self.layer1_process(out)
        
        out = F.pad(out, (5, 5), mode='constant', value=0)
        out = self.layer2(out)

        out = F.pad(out, (3, 3), mode='constant', value=0)
        out = self.layer3(out)
        
        # Flatten output of convolutional layers
        out = out.view(out.size()[0], -1)
        
        # run fully connected layers
        out = self.layer4(out)
        out = self.layer5(out)
        predictions = self.layer6(out)
        #predictions = predictions * 8 + 1 # scalar function when combined with Sigmoid function
                
        return predictions
      
# define model for extracting motifs from first convolutional layer
# and determining importance of each filter on prediction
class motifCNN(nn.Module):
    def __init__(self, original_model):
        super(motifCNN, self).__init__()
        self.layer1_conv = nn.Sequential(*list(original_model.children())[0])
        self.layer1_process = nn.Sequential(*list(original_model.children())[1])
        self.layer2 = nn.Sequential(*list(original_model.children())[2])
        self.layer3 = nn.Sequential(*list(original_model.children())[3])
        
        self.layer4 = nn.Sequential(*list(original_model.children())[4])
        self.layer5 = nn.Sequential(*list(original_model.children())[5])
        self.layer6 = nn.Sequential(*list(original_model.children())[6])
        

    def forward(self, input):
        # add dummy dimension to input (for num channels=1)
        input = torch.unsqueeze(input, 1)
        
        # Run convolutional layers
        input = F.pad(input, (9, 9), mode='constant', value=0) # padding - last dimension goes first
        out= self.layer1_conv(input)
        layer1_activations = torch.squeeze(out)
        
        #do maxpooling and batch normalization for layer 1
        layer1_out = self.layer1_process(out)
        layer1_out = F.pad(layer1_out, (5, 5), mode='constant', value=0)
        
        #calculate average activation by filter for the whole batch
        filter_means_batch = layer1_activations.mean(0).mean(1)
    
        # run all other layers with 1 layer left out at a time
        batch_size = layer1_out.shape[0]
        predictions = torch.zeros(batch_size, 300,  self.layer6[0].out_features)
        #filt_indx = shuffled_list = random.sample(range(300), k=300)
        #filter_matches = np.load("../outputs/motifs2/run2_motif_matches.npy")

        for i in range(300):
            #modify filter i of first layer output
            filter_input = layer1_out.clone()

            filter_input[:,i,:,:] = filter_input.new_full((batch_size, 1, 94), fill_value=filter_means_batch[i])
           
            #match = filter_matches[filter_matches[:,0]==i,][:,1]
            #for j in match:
            #    filter_input[:,j,:,:] = filter_input.new_full((batch_size, 1, 94), fill_value=filter_means_batch[j])
            
            out = self.layer2(filter_input)
            out = F.pad(out, (3, 3), mode='constant', value=0)
            out = self.layer3(out)
            
            # Flatten output of convolutional layers
            out = out.view(out.size()[0], -1)
            # run fully connected layers
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)
            
            predictions[:,i,:] = out

        return layer1_activations, predictions  
       
    
#define the model loss
def pearson_loss(obs, pred):
    mean_obs = torch.mean(obs, dim=1, keepdim=True)
    mean_pred = torch.mean(pred, dim=1, keepdim=True)
    obs_mean, pred_mean = obs - mean_obs, pred - mean_pred

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = torch.sum(1-cos(obs_mean, pred_mean))
    # Normalize by the batch_size
    #loss = loss / obs.shape[0]
    return loss 

    
#define pearson loss with social regularization
def pearson_reg_loss(obs, pred, neighbors):
    mean_obs = torch.mean(obs, dim=1, keepdim=True)
    mean_pred = torch.mean(pred, dim=1, keepdim=True)
    obs_mean, pred_mean = obs - mean_obs, pred - mean_pred
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = torch.sum(1-cos(obs_mean, pred_mean))

    for edge in neighbors:
        neighbor1 = edge[0]
        neighbor2 = edge[1]
        loss += torch.abs(obs[neighbor1] - obs[neighbor2])

    return loss 
     
    
def train_model(train_loader, valid_loader, model, device, criterion, optimizer, num_epochs):
    # Save the best model based on the minimun valid loss
    # TODO: early stopping using the valid loss 
    # (criteria: e.g. no improvement of eval loss for 3 epochs)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_valid = float('inf')
    best_epoch = 1
    
    total_step_train = len(train_loader) 
    #total_step_valid = len(valid_loader) 
    training_loss = []
    valid_loss = []
    for epoch in range(num_epochs):
        # Train the model
        model.train() # Set model to training mode
        epoch_loss_train=0
        for i, (seqs, labels) in enumerate(train_loader):
            seqs = seqs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            epoch_loss_train += loss.item()
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step_train, loss.item() / len(labels)))
        epoch_loss_train = epoch_loss_train / len(train_loader.dataset)
        training_loss.append(epoch_loss_train) 
        
        # Validate the model
        model.eval() # Set model to evaluate mode
        epoch_loss_valid=0
        with torch.no_grad():
            for i, (seqs, labels) in enumerate(valid_loader):
                seqs = seqs.to(device)
                labels = labels.to(device)
                outputs = model(seqs) # original
#                outputs = model.eval(seqs)
                loss_valid = criterion(outputs, labels)
                epoch_loss_valid += loss_valid.item()            
            epoch_loss_valid = epoch_loss_valid / len(valid_loader.dataset)
            valid_loss.append(epoch_loss_valid)
            print ('Epoch [{}/{}], Valid Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, epoch_loss_valid))
            
        # deep copy the model # Is indentation needed to use no grad?
        if epoch_loss_valid < best_loss_valid:
            best_loss_valid = epoch_loss_valid
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            print ('Saving the best model weights at Epoch [{}], Best Valid Loss: {:.4f}' 
                       .format(epoch+1, best_loss_valid))
    # load best model weights
    model.load_state_dict(best_model_wts)    
    return model, training_loss, valid_loss, best_epoch


def test_model(test_loader, model, device, num_classes):
    predictions =  torch.zeros(0, num_classes)
    model.eval() # Set model to evaluate mode
    with torch.no_grad():        
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            pred = model(seqs)
            predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)

    predictions = predictions.numpy()
    return predictions


def get_motifs(data_loader, model, device, num_classes):
    activations = torch.zeros(0, 300, 251)
    predictions = torch.zeros(0, 300, num_classes)
    model.eval() # Set model to evaluate mode
    with torch.no_grad():        
        for seqs, labels in data_loader:
            seqs = seqs.to(device)
            act, pred = model(seqs)
            
            activations = torch.cat((activations, act.type(torch.FloatTensor)), 0)
            predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)
            
    predictions = predictions.numpy()
    activations = activations.numpy()
    return activations, predictions
