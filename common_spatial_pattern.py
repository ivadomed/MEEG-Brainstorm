#!/usr/bin/env python

"""
This script is used to compute the Common Spatial Pattern projection based on:
`"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
<https://arxiv.org/pdf/2106.11170.pdf>`_.

Usage: type "from common_spatial_filter import <function>" to use function.

Contributors: Ambroise Odonnat.
"""


import numpy as np

from numpy.linalg import eig


def whitening(data, r):

    """ Compute whitening matrix to apply
        Principal Component Analysis whitening.
        
    Args:
        data (array): Set of trials of dimension
                      [n_trials x n_channels x n_time_points].
        r (int): Number of first and last rows to select.
 
    Returns:
        whitening_matrix (array): Whitening matrix of dimension
                                  [(2xselected_rows) x n_channels]
    """

    # Compute covariance matrices for all the trials
    n_trials, n_channels, _ = data.shape
    cov = np.zeros([n_trials, n_channels, n_channels])

    for n in range(n_trials):
        E = data[n, :, :]
        EE = np.dot(E, E.transpose())
        cov[n, :, :] = EE/np.trace(EE)

    # Mean covariance matrix
    cov = np.mean(cov, axis=0)

    # Eigendecomposition of cov
    lambd, V = eig(cov)
    eig_order = np.argsort(lambd)
    eig_order = eig_order[::-1]
    eig_values = lambd[eig_order]
    eig_vectors = V[:, eig_order]

    """
    Cov_total can have very small negative values due
    to computational error. To overcome this issue, we can add
    a small positive constant to the the diagonal.
    We choose to add epsilon = 1e-10.
    """
    epsilon = 1e-10

    # Compute whitening matrix
    Ptmp = np.sqrt(np.diag(np.power(eig_values + epsilon, -1)))
    P = np.dot(Ptmp, eig_vectors.transpose())

    # Select r first and last rows of w (r = selected_rows)
    n_rows = P.shape[0]
    whitening_matrix = np.concatenate((P[:r],
                                       P[n_rows-r:]), axis=0)

    return whitening_matrix


def common_spatial_pattern(data, labels, r):

    """
    Compute Common Spatial Pattern matrix based on:
    `"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
    <https://arxiv.org/pdf/2106.11170.pdf>`_.

    Args:
        data (array): Array of trials of dimension
                      [n_trials x n_channels x n_time_sample_points].
        labels (array): Array of corresponding labels of dimension [n_trials].
        r (int): Number of first and last rows to select on each subfilter.
  
    Returns:
        CSP_projection (array): CSP projection array of dimension
                                [(2r) x n_channels] if binary classification.
                                [(2Nr) x n_channels] else.
                                 N = n_classes: r = selected_rows.
    """

    # Recover subsets of trials corresponding to each class
    unique_labels = np.unique(labels)
    N = len(unique_labels)
    idx = [n_class for n_class in range(N)]
    for n_class in range(N):
        idx[n_class] = np.where(labels == unique_labels[n_class])[0]
 
    # Recover number of electrode channels in the trials
    n_channels = data.shape[1]
    
    # Binary classification
    if N <= 2:
        idx_one = idx[0]
        idx_rest = idx[1]
          
        # Compute covariance matrices for all trials
        cov_one = np.zeros([n_channels, n_channels, len(idx_one)])
        cov_rest = np.zeros([n_channels, n_channels, len(idx_rest)])

        for n_one in range(len(idx_one)):
            E = data[idx_one[n_one], :, :]
            EE = np.dot(E, E.transpose())
            cov_one[:, :, n_one] = EE/np.trace(EE)

        for n_rest in range(len(idx_rest)):
            E = data[idx_rest[n_rest], :, :]
            EE = np.dot(E, E.transpose())
            cov_rest[:, :, n_rest] = EE/np.trace(EE)

        # Mean covariance matrix
        cov_one = np.mean(cov_one, axis=2)
        cov_rest = np.mean(cov_rest, axis=2)
        cov_total = cov_one + cov_rest

        # Eigendecomposition of cov_total
        lambd, V = eig(cov_total)
        eig_order = np.argsort(lambd)
        eig_order = eig_order[::-1]
        eig_values = lambd[eig_order]
        eig_vectors = V[:, eig_order]

        """
        Cov_total can have very small negative values due
        to computational error. To overcome this issue, we can add
        a small positive constant to the the diagonal.
        We choose to add epsilon = 1e-10.
        """
        epsilon = 1e-10

        # Compute P and B   
        Ptmp = np.sqrt(np.diag(np.power(eig_values + epsilon, -1)))
        P = np.dot(Ptmp, eig_vectors.transpose())

        s_one = np.dot(P, cov_one)
        s_one = np.dot(s_one, P.transpose())
        s_rest = np.dot(P, cov_rest)
        s_rest = np.dot(s_rest, P.transpose())

        lambd_rest, V_rest = eig(s_rest)
        eig_order_rest = np.argsort(lambd_rest)
        B = V_rest[:, eig_order_rest]

        # Compute sub-filter for class n_class
        sub_filter = np.dot(B.transpose(), P)

        # Select r first and last rows of w (r = selected_rows)
        n_rows = sub_filter.shape[0]
        CSP_projection = np.concatenate((sub_filter[:r],
                                         sub_filter[n_rows-r:]),
                                        axis=0)

    # Multi-classification with N classes
    else:
        CSP_projection = []
    
        # Apply One-vs-Rest strategy (N binary classifications)
        for n_class in range(N):

            # Recover index of trials of class "One" and class "Rest"
            idx_one = idx[n_class]
            idx_rest = np.sort(np.concatenate([idx[i] for i in range(len(idx))
                                               if i != n_class]))

            # Compute covariance matrices for all trials
            cov_one = np.zeros([n_channels, n_channels, len(idx_one)])
            cov_rest = np.zeros([n_channels, n_channels, len(idx_rest)])

            for n_one in range(len(idx_one)):
                E = data[idx_one[n_one], :, :]
                EE = np.dot(E, E.transpose())
                cov_one[:, :, n_one] = EE/np.trace(EE)

            for n_rest in range(len(idx_rest)):
                E = data[idx_rest[n_rest], :, :]
                EE = np.dot(E, E.transpose())
                cov_rest[:, :, n_rest] = EE/np.trace(EE)

            # Mean covariance matrices
            cov_one = np.mean(cov_one, axis=2)
            cov_rest = np.mean(cov_rest, axis=2)
            cov_total = cov_one + cov_rest

            # Eigendecomposition of CovTotal
            lambd, V = eig(cov_total)
            eig_order = np.argsort(lambd)
            eig_order = eig_order[::-1]
            eig_values = lambd[eig_order]
            eig_vectors = V[:, eig_order]

            """
            Cov_total can have very small negative values due
            to computational error. To overcome this issue, we can add
            a small positive constant to the the diagonal.
            We choose to add epsilon = 1e-10.
            """
            epsilon = 1e-10

            # Compute P and B
            Ptmp = np.sqrt(np.diag(np.power(eig_values + epsilon, -1)))
            P = np.dot(Ptmp, eig_vectors.transpose())

            s_one = np.dot(P, cov_one)
            s_one = np.dot(s_one, P.transpose())
            s_rest = np.dot(P, cov_rest)
            s_rest = np.dot(s_rest, P.transpose())

            lambd_rest, V_rest = eig(s_rest)
            eig_order_rest = np.argsort(lambd_rest)
            B = V_rest[:, eig_order_rest]

            # Compute sub-filter for class n_class
            sub_filter = np.dot(B.transpose(), P)

            # Select r first and last rows of sub-filter w (r = selected_rows)
            nrows = sub_filter.shape[0]
            CSP_projection.append(np.concatenate((sub_filter[:r],
                                                  sub_filter[nrows-r:]),
                                                 axis=0))

        # Concatenate all N subfilters
        CSP_projection = np.concatenate(CSP_projection, axis=0)

    return CSP_projection
