import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.linear_in  = nn.Linear(725, 1)
        self.sigmoid    = nn.Sigmoid()
        self.batch_norm = torch.nn.BatchNorm1d(1)
        
    def forward(self, x):
        y = self.linear_in(self.batch_norm(x)).squeeze(-1)
        return y
