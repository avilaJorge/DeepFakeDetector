# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

# numpy imports
import numpy as np
# Image imports (PIL)
from PIL import Image
# Pretty Print
import pprint
pp = pprint.PrettyPrinter(width=20)


# Configuration variables
img_root    = '/home/jupyter/CSE253_FinalProject/Frequency/Faces-HQ'
_batch_size = 128
_shuffle    = True
_num_wrks   = 4

data_transforms = transforms.Compose([
    transforms.ToTensor(),
])


# Radial Profile of Shifted Data (Azimuthally Averaged)
def np_radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    y, x = torch.from_numpy(y), torch.from_numpy(x)

    r = torch.stack((torch.sub(x, center[0]), torch.sub(y, center[1])), dim=2).float()
    r = torch.norm(r, dim=2).long()
    tbin = torch.bincount(r.view(r.numel()), data.view(data.numel()))
    nr   = torch.bincount(r.view(r.numel()))

    radialprofile = torch.div(tbin, nr)
    return radialprofile

# FFT is applied and the amplitude is computed
def np_magnitude_spectrum(img):
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    return 20*np.log(np.abs(f))

def magnitude_spectrum(img):
    t_img =  torch.rfft(img, signal_ndim=2, onesided=False) * 255
    shifts = (int(t_img.size()[0]/2), int(t_img.size()[1]/2))
    t_img = torch.roll(t_img, shifts=shifts, dims=(0, 1))
    return 20*torch.log(torch.norm(t_img, dim=2))

# Image loader 
def pil_grey_loader(path):
    """
    
    Args:
        
    Returns:
        
    Ensures:
        
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

class DeepFakeDataset(ImageFolder):
    
    def __init__(self, root, transforms):
        super(DeepFakeDataset, self).__init__(root=root, 
                                              loader=pil_grey_loader,
                                              transform=transforms)
        
        self.images = self.samples
        pp.pprint("Classes: %s" % self.classes)
        pp.pprint("Indices: %s" % self.class_to_idx)
        """
        Target folders where 1 is fake and 0 is real
        """
        self.switcher = {
            'celebA-HQ_10K': 0,
            'Flickr-Faces-HQ_10K': 0,
            'thispersondoesntexists_10K': 1,
            '100KFake_10K': 1
        }
        
    def __getitem__(self, index):
        img, t = super(DeepFakeDataset, self).__getitem__(index)
        
        ms_img = magnitude_spectrum(img.squeeze(0).float())
        rad_p = radial_profile(ms_img, center=(ms_img.shape[0]/2, ms_img.shape[1]/2))
        
        return rad_p, self.switcher[self.classes[t]]
        

def get_dataloaders(image_root=img_root, 
                   transforms=data_transforms, 
                   batch_size=_batch_size, 
                   shuffle=_shuffle, 
                   num_workers=_num_wrks):
    
    ds = DeepFakeDataset(root=img_root, transforms=transforms)
    
    # Compute train, val, test splits
    trn_len, val_len, tst_len = int(len(ds)*0.6), int(len(ds)*0.2), int(len(ds)*0.2)
    trn_len += (len(ds) - (trn_len + val_len + tst_len))
    
    trn, val, tst = random_split(ds, [trn_len, val_len, tst_len]) 
    trn_dl = DataLoader(trn, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers)
    val_dl = DataLoader(val, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers)
    tst_dl = DataLoader(tst, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers)
    return trn_dl, val_dl, tst_dl 