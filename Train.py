#!/opt/anaconda3/bin/python

"""
This script is used to train and test the model. 

Usage: type "from Train import <class>" to use one of its class.
       type "from Train import <function>" to use one of its function.

Contributors: Ambroise Odonnat.
"""

import torch

import numpy as np

from torch.autograd import Variable

from data import Data
from dataloader import Loader
from Model import Transformer
from utils import check_balance

import logging
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

class Trans():
    
    def __init__(self, folder, channel_fname, wanted_event_label,list_channel_type, binary_classification, selected_rows,\
                 train_size,batch_size, num_workers, random_state, shuffle, balanced, \
                 normalized_shape = 201, linear_size = 28, vector_size = 201,\
                 attention_dropout = 0.3, attention_negative_slope = 1e-2, attention_kernel_size = 30, attention_stride = 30,\
                 spatial_dropout = 0.5,\
                 out_channels = 2, position_kernel_size = 51, position_stride = 1, emb_negative_slope = 0.2,\
                 channel_kernel_size = 28, time_kernel_size = 5, time_stride = 5, slice_size = 10,\
                 depth = 3, num_heads = 5, transformer_dropout = 0.5,forward_expansion = 4, forward_dropout = 0.5,\
                 n_classes = 7, n_epochs = 2, lr = 1e-3, b1 = 0.5, b2 = 0.9, BETA = 0.4, display_balance = True):
        
        super().__init__()
        self.normalized_shape = normalized_shape
        self.linear_size = linear_size
        self.vector_size = vector_size
        self.attention_dropout = attention_dropout
        self.attention_negative_slope = attention_negative_slope
        self.attention_kernel_size = attention_kernel_size 
        self.attention_stride = attention_stride
        self.spatial_dropout = spatial_dropout
        self.out_channels = out_channels 
        self.position_kernel_size = position_kernel_size 
        self.position_stride = position_stride
        self.emb_negative_slope = emb_negative_slope
        self.channel_kernel_size = channel_kernel_size
        self.time_kernel_size = time_kernel_size
        self.time_stride = time_stride
        self.slice_size = slice_size
        self.depth = depth
        self.num_heads = num_heads
        self.transformer_dropout = transformer_dropout
        self.forward_expansion = forward_expansion
        self.forward_dropout = forward_dropout
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.BETA = BETA # parameter of Beta law for mix-up
        
        # Data format
        self.Tensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor
        
        # Recover dataset
        log.info(' Get Dataset')
        self.dataset = Data(folder,channel_fname,wanted_event_label, list_channel_type,binary_classification, selected_rows)
        self.allData, self.allLabels, self.allSpikeTimePoints, self.allTimes = self.dataset.csp_data()
        
        # Recover dataloader
        log.info(' Get Dataloader')
        self.dataloader = Loader(self.allData, self.allLabels, train_size, batch_size, num_workers, random_state, shuffle, balanced)
        self.train_loader, self.test_loader = self.dataloader.train_test_dataloader()

        # Check that train_loader is balanced
        if display_balance:
            log.info(' Check balance')
            check_balance(self.train_loader, n_classes, n_epochs, True)
        

    def train(self, mix_up = True):

        # Define model
        self.model = Transformer(self.normalized_shape, self.linear_size, self.vector_size,\
                 self.attention_dropout, self.attention_negative_slope, self.attention_kernel_size, self.attention_stride,\
                 self.spatial_dropout,\
                 self.out_channels, self.position_kernel_size, self.position_stride, self.emb_negative_slope,\
                 self.channel_kernel_size, self.time_kernel_size, self.time_stride, self.slice_size,\
                 self.depth, self.num_heads, self.transformer_dropout, self.forward_expansion, self.forward_dropout,\
                 self.n_classes)

        # Define loss
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        
        bestAcc = 0
        averAcc = 0
        best_epochs = 0
        num = 0
        Y_true = []
        Y_pred = []    
      
        for e in range(self.n_epochs):
            
            # Train the model
            self.model.train()
            correct, total = 0,0
            for i, (data, labels) in enumerate(self.train_loader):
                                
                if mix_up:
                    
                    # Apply a mix-up strategy for data augmentation as adviced here '<https://forums.fast.ai/t/mixup-data-augmentation/22764>'

                    # Roll a copy of the batch
                    roll_factor =  torch.randint(0, data.shape[0], (1,)).item()
                    rolled_data = torch.roll(data, roll_factor, dims=0)        
                    rolled_labels = torch.roll(labels, roll_factor, dims=0)  

                    # Create a tensor of lambdas sampled from the beta distribution
                    lambdas = np.random.beta(self.BETA, self.BETA, data.shape[0])

                    # trick from here https://forums.fast.ai/t/mixup-data-augmentation/22764
                    lambdas = torch.reshape(torch.tensor(np.maximum(lambdas, 1-lambdas)), (-1,1,1,1))

                    # Mix samples
                    mix_data = lambdas*data + (1-lambdas)*rolled_data

                    # Recover data, labels
                    mix_data = Variable(mix_data.type(self.Tensor))
                    data = Variable(data.type(self.Tensor))
                    labels = Variable(labels.type(self.LongTensor))
                    rolled_labels = Variable(rolled_labels.type(self.LongTensor))

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward
                    x, mix_outputs = self.model(mix_data)
                    loss = lambdas.squeeze()*self.criterion_cls(mix_outputs, labels) + (1-lambdas.squeeze())*self.criterion_cls(mix_outputs, rolled_labels)
                    loss = loss.sum()
                    loss.backward()
                    
                    # Optimize
                    self.optimizer.step()
                    
                    # Count accurate prediction
                    y_pred = torch.max(mix_outputs.data, 1)[1]
                    total += labels.size(0)
                    correct += (y_pred == labels).sum().item()
                    
                else:
                    data = Variable(data.type(self.Tensor))
                    labels = Variable(labels.type(self.LongTensor))
                
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward
                    x, outputs = self.model(data)
                    loss = self.criterion_cls(outputs, labels)
                    loss.backward()
                    
                    # Optimize
                    self.optimizer.step()
                    
                    # Count accurate prediction
                    y_pred = torch.max(outputs.data, 1)[1]
                    total += labels.size(0)
                    correct += (y_pred == labels).sum().item()
                                
            if (e + 1) % 1 == 0:
                
                # Evaluate the model
                self.model.eval()
                test_correct, test_total = 0,0
                Predictions = []
                Labels = []
                for j, (test_data, test_labels) in enumerate(self.test_loader):
                    
                    # Recover data, labels
                    test_data = Variable(test_data.type(self.Tensor))
                    test_labels = Variable(test_labels.type(self.LongTensor))
                    
                    # Recover outputs
                    test_x, test_outputs = self.model(test_data)
                    test_loss = self.criterion_cls(test_outputs, test_labels)
                    
                    # Count accurate prediction
                    test_y_pred = torch.max(test_outputs, 1)[1]
                    test_total += test_labels.size(0)
                    test_correct += (test_y_pred == test_labels).sum().item()
                
                    # Recover labels and prediction
                    Predictions.append(test_y_pred.detach().numpy())
                    Labels.append(test_labels.detach().numpy())
                    
                train_acc = 100 * correct // total
                test_acc = 100 * test_correct // test_total
                print('Epoch:', e,
                      '  Train loss:', loss.detach().numpy(),
                      '  Test loss:', test_loss.detach().numpy(),
                      '  Train accuracy:', train_acc,
                      '  Test accuracy is:', test_acc)
                num+=1
                averAcc = averAcc + test_acc
                if test_acc > bestAcc:
                    bestAcc = test_acc
                    best_epochs = e
                    Y_true = Predictions
                    Y_pred = Labels

        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)

        return bestAcc, averAcc, Y_true, Y_pred