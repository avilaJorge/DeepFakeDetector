import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykeops.torch import LazyTensor

use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

def save_model(decoder, name, date, model_path, encoder=None, specify_fn=False, fn=''):
    # Save the model checkpoints
    if specify_fn:
        torch.save(decoder.state_dict(), os.path.join(
            model_path, 'decoder-{}-{}.ckpt'.format(name, date)))
        if encoder is not None:
            torch.save(encoder.state_dict(), os.path.join(
                model_path, 'encoder-{}-{}.ckpt'.format(name, date)))
    else:
        torch.save(decoder.state_dict(), os.path.join(
            model_path, 'decoder-{}-{}.ckpt'.format(name, date)))
        if encoder is not None:
            torch.save(encoder.state_dict(), os.path.join(
                model_path, 'encoder-{}-{}.ckpt'.format(name, date)))    

class LinearRegression(nn.Module):
    
    def __init__(self, in_dims):
        super(LinearRegression, self).__init__()
        self.linear_in  = nn.Linear(in_dims, 1)
        self.sigmoid    = nn.Sigmoid()
        self.batch_norm = torch.nn.BatchNorm1d(1)
        
    def forward(self, x):
        y = self.linear_in(self.batch_norm(x)).squeeze(-1)
        return y



class SVM(nn.Module):

    def __init__(self, in_dims):
        super(SVM, self).__init__()
        self.linear = nn.Linear(in_dims, 1)
        self.batch_norm = torch.nn.BatchNorm1d(1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = self.linear(self.batch_norm(x)).squeeze(-1)
        return y
        
# From: https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
class KMeans:

    def __init__(self):
        self.i = 0

    def run(self, x, K=10, Niter=10, verbose=True):
        N, D = x.shape  # Number of samples, dimension of the ambient space

        # K-means loop:
        # - x  is the point cloud,
        # - cl is the vector of class labels
        # - c  is the cloud of cluster centroids
        start = time.time()
        c = x[:K, :].clone()  # Simplistic random initialization
        x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

        for i in range(Niter):

            c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
            cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

            Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
            for d in range(D):  # Compute the cluster centroids with torch.bincount:
                c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

        end = time.time()

        if verbose:
            print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
            print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                    Niter, end - start, Niter, (end-start) / Niter))

        return cl, c