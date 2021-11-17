import torch
from torchvision import models
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

#############################################
# Class: Autoencoder
#############################################
class Autoencoder(nn.Module): 
    def __init__(self, input_size, encode_size, model=None):
        # init super nn.Module
        super(Autoencoder, self).__init__()

        self.input_size = input_size
        self.encode_size = encode_size

        # create encode & decoder
        self.encoder = nn.Sequential(   nn.Linear(self.input_size, 512),
                                        nn.ReLU(inplace = True),
                                        nn.Linear(512, self.encode_size) )

        self.decoder = nn.Sequential(   nn.Linear(self.encode_size, 512),
                                        nn.ReLU(inplace = True),
                                        nn.Linear(512, self.input_size) )

        if(model):
            torch.load(model, map_location={'cuda:1':'cuda:0'})


    #############################################
    # Model Forward Pass
    #############################################
    def forward(self, x, train=False):
        if(train):
            x = self.encoder(x)
            output = self.decoder(x)
        else:
            output = self.encoder(x)

        return output


    #############################################
    # Train Autoencoder
    #############################################
    def train(self, data, learning_rate, epochs, path_save):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        train_loss = []
        min_loss = 5.0
        best_model = None

        # For each epoch
        for epoch in tqdm(range(epochs), desc='Training AE (Epochs)'):
            epoch_loss = []

            # For each sample in batch
            for batch, sample in enumerate(data):
                # Put on GPU
                sample = torch.Tensor(sample).type('torch.FloatTensor').cuda()
                # sample = torch.Tensor(sample)

                # Forward pass 
                out_features = self.forward(sample, train=True)

                # Calculate loss
                loss = self.mse_loss(out_features, sample)
                epoch_loss.append(loss.item())

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = np.sum(epoch_loss)/len(epoch_loss)

            train_loss.append(avg_loss)
            
            if(avg_loss < min_loss):
                min_loss = avg_loss
                best_model = self.state_dict()

        path_save = os.path.join(path_save, str(epochs)+'_epochs_'+str(self.encode_size)+'_dims')
        torch.save(best_model, path_save+'.pt')

        return train_loss


    #############################################
    # MSE Loss function - pytorch
    #############################################
    def mse_loss(self, pred, truth):
        cost = torch.nn.MSELoss()

        return cost(pred, truth)
