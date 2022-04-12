#!/usr/bin/env python

"""
This script is used to implement custom losses. 

Usage: type "from custom_losses import <class>" to use one of its classes.
       
Contributors: Ambroise Odonnat.
"""

import torch

import numpy as np

from torch import nn
from torchmetrics.functional import  f1_score, precision_recall

from utils import *


class CostSensitiveLoss(nn.Module):
    
    """
    Implement a cost-sensitive loss inspired by:
    "Cost-Sensitive Convolution based Neural Networks for Imbalanced Time-Series Classification"
    `<https://arxiv.org/pdf/2010.00291.pdf>`_.
    """
    
    def __init__(self, criterion, n_classes, lambd):
        
        """
        Args:
            criterion (Loss): Criterion,
            n_classes (int): Number of classes,
            lambd (float): Modulate influence of the cost-sensitive weight.
        """
        
        super().__init__()
        
        self.criterion = criterion
        self.n_classes = n_classes
        self.lambd = lambd
        
        # Compute cost-sensitive matrix
        M = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(i+1,n_classes):
                M[i,j] = (j - i) ** 2
                
        self.M = torch.from_numpy(M)

    def forward(self, outputs, labels):
        
        """
        Args:
            outputs (tensor): Array of logits of dimension (batch_size x n_classes),
            labels (tensor): Ground truth of dimension batch_size.
        
        Return:
            loss (float): Mean loss value on the batch.
        """
        
        loss = self.criterion(outputs, labels)
        prediction = torch.max(outputs.data, 1)[1]
        CS = self.M[prediction,labels]
        
        # Compute coeff
        F1_score = f1_score(prediction, labels, average = 'macro', num_classes = self.n_classes)
        precision, recall = precision_recall(prediction, labels, average = 'macro', num_classes = self.n_classes)

        coeff = self.lambd
        if F1_score:
            #coeff = - self.lambd * np.log(F1_score)
            #coeff = - self.lambd * np.log(recall)
            coeff = - self.lambd * np.log(np.sqrt(recall*precision)) #best one yet
            
        loss += coeff * CS.mean()
        return loss
        
        
        
class DetectionLoss(nn.Module):
    
    """
    Implement cost-sensitive loss based on `<https://arxiv.org/pdf/2010.00291.pdf>`.
    """
    
    def __init__(self, criterion, n_classes, lambd):
        
        """
        Args:
            criterion (Loss): Criterion,
            n_classes (int): Number of classes,
            lambd (float): Modulate influence of the cost-sensitive weight.
        """
        
        super().__init__()
        
        self.criterion = criterion
        self.n_classes = n_classes
        self.lambd = lambd
        
        # Compute cost-sensitive matrix
        M = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(i+1,n_classes):
                M[i,j] = (j - i) ** 2
                
        self.M = torch.from_numpy(M)
        
    def forward(self, outputs, labels):
        
        """
        Args:
            outputs (tensor): Array of logits of dimension (batch_size x n_classes),
            labels (tensor): Ground truth of dimension batch_size.
        
        Return:
            loss (float): Mean loss value on the batch.
        """
        
        loss = self.criterion(outputs, labels)
        prediction = torch.max(outputs.data, 1)[1]
        CS = self.M[prediction,labels]

        # Compute coeff
        confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        for t, p in zip(labels.reshape(-1), prediction.reshape(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        TP = confusion_matrix[1][1]
        TN = confusion_matrix[0][0] 
        FP = confusion_matrix[0][1] 
        FN = confusion_matrix[1][0] 
        coeff = self.lambd
        if FN * TP * FP * TN:
            #coeff = - self.lambd * np.log(TP / (TP + FN)) 
            coeff = - self.lambd * np.log(np.sqrt((TP/(TP + FN))*(FP / (FP + TN)))) #best one yet
        loss += coeff * CS.mean()
        return loss
    
    
def get_training_loss(labels, n_classes, cost_sensitive, lambd, weight_method, beta):
    
    """
    Build a custom cross-entropy.
    
    Args:
        labels (array): Labels in the training set,
        cost_sensitive (bool): Build cost-sensitive cross entropy loss,
        lambd (float): Modulate the influence of the cost-sensitive weight,
        weight_method (str): Use weights in the cross-entropy loss if weight_method is provided,
        beta (float): Beta value for ENS method.
        
    Return:
        criterion (Loss): CrossEntropyLoss.
    """
    
    # Apply weighting method
    if weight_method == None:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        if weight_method == 'INS':
            weight = inverse_number_samples(labels)
        elif weight_method == 'ISNS':
            weight = inverse_square_root_number_samples(labels)
        elif weight_method == 'ENS':
            weight = effective_number_samples(labels, beta)
        criterion =  torch.nn.CrossEntropyLoss(weight = weight)
    
    # Apply cost-sensitive method
    if cost_sensitive:
        return CostSensitiveLoss(criterion, n_classes, lambd)
    else:
        return criterion

        
