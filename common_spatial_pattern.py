#!/opt/anaconda3/bin/python

"""
This script is used to compute the Common Spatial Pattern projection W.

Usage: type "from common_spatial_filter import <function>" to use one of ist functions.

Contributors: Ambroise Odonnat.
"""

import numpy as np
from numpy.linalg import eig


def csp(data, labels, selected_rows):
    
    """
    Compute Common Spatial Pattern matrix W based on 
    `"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding " <https://arxiv.org/pdf/2106.11170.pdf>`_
    
    Args:
        data (array): Array of trials, size (n_trials)x(n_channels)x(n_sample_points).
        labels (array): Array of corresponding labels, size n_trials).
        selected_rows (int): Number of first rows and number of last rows selected on each sub-filter.
        
    Returns:
        W (array): CSP projection matrix (Nr)x(n_channels) where N = number of classes; r = selected_rows
    """
    
    # Recover trials' index for all classes
    unique_labels = np.unique(labels)
    N = unique_labels.shape[0] 
    idx = [n_class for n_class in range(N)]
    for n_class in range(N):
        idx[n_class] = np.where(labels == unique_labels[n_class])[0]
        
    # Recover number of electrode channels in the trials
    n_channels = data.shape[1]

    W = []
    
    # Apply One-vs-Rest strategy (N bi-classification)
    for n_class in range(N):
        
        # Recover index of trials of class "One" and class "Rest"
        idx_One = idx[n_class]
        idx_Rest = np.sort(np.concatenate([idx[i] for i in range(len(idx)) if i != n_class]))
        
        # Compute covariance matrices for all trials
        Cov_One = np.zeros([n_channels, n_channels, len(idx_One)])
        Cov_Rest = np.zeros([n_channels, n_channels, len(idx_Rest)])

        for nOne in range(len(idx_One)):
            E = data[idx_One[nOne], :, :]
            EE = np.dot(E,E.transpose())
            Cov_One[:, :, nOne] = EE/np.trace(EE)
            
        for nRest in range(len(idx_Rest)):
            E = data[idx_Rest[nRest], :, :]
            EE = np.dot(E,E.transpose())
            Cov_Rest[:, :, nRest] = EE/np.trace(EE)
            
        # Mean covariance matrices
        Cov_One = np.mean(Cov_One, axis=2)
        Cov_Rest = np.mean(Cov_Rest, axis=2)
        CovTotal = Cov_One + Cov_Rest  
        
        # Eigendecomposition of CovTotal
        lam, V = eig(CovTotal)
        eigorder = np.argsort(lam)
        eigorder = eigorder[::-1]
        eigValues = lam[eigorder]
        eigVectors = V[:, eigorder]
        
        # Compute P and B      
        """
        CovTotal can have very small negative ones due to computational error.
        To overcome this issue, we can add a small positive constant to the the diagonal. 
        Compute the pseudo-inverse can also be a solution.
        We choose to add epsilon = 1e-15.
        """
        
        epsilon = 1e-16
        Ptmp = np.sqrt(np.diag(np.power(eigValues + epsilon, -1)))
        P = np.dot(Ptmp, eigVectors.transpose())
        
        SOne = np.dot(P, Cov_One)
        SOO = np.dot(SOne, P.transpose())
        SRest = np.dot(P, Cov_Rest)
        SRR = np.dot(SRest, P.transpose())

        lam_Rest, VRest = eig(SRR)
        eigorderRest = np.argsort(lam_Rest)
        B = VRest[:, eigorderRest]

        # Compute sub-filter for class n_class
        w = np.dot(B.transpose(), P)
        
        # Select r first and last rows of sub-filter w (r = selected_rows)
        nrows = w.shape[0]
        W.append(np.concatenate((w[:selected_rows], w[nrows-selected_rows:]), axis = 0))

    # Concatenate all N subfilters to obtain the spatial projection wanted of size (Nr)x(number of channels)
    W = np.concatenate(W, axis=0)

    return W