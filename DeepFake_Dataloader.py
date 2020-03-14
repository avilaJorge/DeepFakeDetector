# PyTorch imports
import torch
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split


# numpy imports
import numpy as np
# Image imports (PIL)
from PIL import Image
# hdf5 import
import h5py
# Pretty Print
import pprint
pp = pprint.PrettyPrinter(width=20)

# data_origin = {
#     'celebA-HQ_10K': 0,
#     'Flickr-Faces-HQ_10K': 1,
#     'thispersondoesntexists_10K': 2,
#     '100KFake_10K': 3,
#     0: 'celebA-HQ_10K',
#     1: 'Flickr-Faces-HQ_10K',
#     2: 'thispersondoesntexists_10K',
#     3: '100KFake_10K'
# }

data_origin = {
    'lsun_bedrooms': 0,
    'lsun_cats': 1,
    'lsun_churches': 2,
    'stylegan1_bedrooms': 3,
    'stylegan2_cats': 4,
    'stylegan2_churches': 5,
    0: 'lsun_bedrooms',
    1: 'lsun_cats',
    2: 'lsun_churches',
    3: 'stylegan1_bedrooms',
    4: 'stylegan2_cats',
    5: 'stylegan2_churches',
}

rad_size = {
    'stylegan2_cats': 182,
    'stylegan2_cars': 363,
    'stylegan2_churches': 182,
    'lsun_cats': 182,
    'lsun_cars': 363,
    'lsun_churches': 182,
    'stylegan1_bedrooms': 182,
    'lsun_bedrooms': 182
}

data_img_size = {
    'stylegan2_cats': 256,
    'stylegan2_cars': 512,
    'stylegan2_churches': 256,
    'lsun_cats': 256,
    'lsun_cars': 512,
    'lsun_churches': 256,
    'stylegan1_bedrooms': 256,
    'lsun_bedrooms': 256
}

# Configuration variables
# img_root    = '/home/jupyter/image_folder'
# img_root    = '/home/jupyter/image_folder_cars'
# img_root    = '/home/jupyter/image_folder_256'
img_root    = '/home/jupyter/image_folder_bcc_256'
# fhq_hdf5_pt = '/home/jupyter/CSE253_FinalProject/Faces_HQ.hdf5'
fhq_hdf5_pt = '~/CSE253_FinalProject/LSUN.hdf5'
_batch_size = 128
_shuffle    = True
_num_wrks   = 16
epsilon     = 1e-10

"""
  Transformation applied to grayscale images
"""
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# Radial Profile of Shifted Data (Azimuthally Averaged)
def np_radial_profile(data, center, pad=True, length=363):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())

    radialprofile = tbin / nr
    
#     if pad:
#         radialprofile = np.pad(radialprofile, (0, length - tbin.shape[0]), 'constant', constant_values=(0,0))
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
    return 20*np.log(np.abs(f) + epsilon)

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
    
class DeepFakePreProcessor(ImageFolder):
    def __init__(self, root, transforms):
        super(DeepFakePreProcessor, self).__init__(root=root, 
                                              loader=pil_grey_loader,
                                              transform=None)
        
        self.images = self.samples
        pp.pprint("Classes: %s" % self.classes)
        pp.pprint("Indices: %s" % self.class_to_idx)
        """
        Target folders where 1 is fake and 0 is real
        """
        self.switcher = {
            'stylegan2_cats': 1,
            'stylegan2_cars': 1,
            'stylegan2_churches': 1,
            'lsun_cats': 0,
            'lsun_cars': 0,
            'lsun_bedrooms': 0,
            'stylegan1_bedrooms': 1,
            'lsun_churches': 0,
            'celebA-HQ_10K': 0,
            'Flickr-Faces-HQ_10K': 0,
            'thispersondoesntexists_10K': 1,
            '100KFake_10K': 1
        }
        
    def __getitem__(self, index):
        img, t = super(DeepFakePreProcessor, self).__getitem__(index)

        ms_img = np_magnitude_spectrum(img)
        rad_p = np_radial_profile(ms_img, center=(ms_img.shape[0]/2, ms_img.shape[1]/2))
        return rad_p, self.switcher[self.classes[t]], self.classes[t], np.asarray(img), ms_img
    

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
            'stylegan2_cats': 1,
            'stylegan2_cars': 1,
            'stylegan2_churches': 1,
            'lsun_cats': 0,
            'lsun_cars': 0,
            'lsun_bedrooms': 0,
            'stylegan1_bedrooms': 1,
            'lsun_churches': 0,
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
    
class DeepFakeHDF5Dataset(Dataset):
    
    def __init__(self, hdf5_path=fhq_hdf5_pt):
        super(DeepFakeHDF5Dataset, self).__init__()
        self.data = None
        self.lbls = None
        self.og_d = None
        with h5py.File(fhq_hdf5_pt, 'r') as hdf5_file: 
            self.data = hdf5_file['fft_data'][:]
            self.lbls = hdf5_file['lbl_data'][:]
            self.og_d = hdf5_file['orgn_data'][:]
        
    def __getitem__(self, index):

        sample = torch.from_numpy(self.data[index,:]).float()
        label  = ((torch.FloatTensor([1]) * self.lbls[index]))
        return sample, label
                
    
    def __len__(self):
        return self.data.shape[0]

class DeepFakeHDF5Dataset_SVM(DeepFakeHDF5Dataset):
    
    def __init__(self, hdf5_path=fhq_hdf5_pt):
        super(DeepFakeHDF5Dataset_SVM, self).__init__()
        
    def __getitem__(self, index):

        sample = torch.from_numpy(self.data[index,:]).float()
        label  = 1  if self.lbls[index] == 1 else -1
        label  = (torch.FloatTensor([1]) * label)
        return sample, label


"""
    get_preprocessor
    Helper method for creating and returning the preprocessor
"""
def get_preprocessors(image_root=img_root, 
                   transforms=data_transforms, 
                   batch_size=_batch_size, 
                   shuffle=_shuffle, 
                   num_workers=_num_wrks):
    
    ds = DeepFakePreProcessor(root=img_root, transforms=transforms)
    
    return DataLoader(ds,
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      num_workers=num_workers)
        
"""
    get_dataloaders
    Helper method for creating and returning the all three dataloaders
"""
def get_dataloaders(image_root=img_root,
                   dataset=DeepFakeHDF5Dataset,
                   transforms=data_transforms, 
                   batch_size=_batch_size, 
                   shuffle=_shuffle,
                   full_dataset=False,
                   num_workers=_num_wrks):
    
    ds = dataset()
    
    # Compute train, val, test splits
    if not full_dataset:
        trn_len, val_len, tst_len = int(len(ds)*0.6), int(len(ds)*0.2), int(len(ds)*0.2)
        trn_len += (len(ds) - (trn_len + val_len + tst_len))
        trn, val, tst = random_split(ds, [trn_len, val_len, tst_len]) 
    else:
        trn_len = len(ds)
        trn = ds
    

    trn_dl = DataLoader(trn,
                        batch_size=batch_size if not full_dataset else trn_len, 
                        shuffle=shuffle, 
                        num_workers=num_workers)
    if full_dataset:
        return trn_dl
    val_dl = DataLoader(val, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers)
    tst_dl = DataLoader(tst, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers)
    return trn_dl, val_dl, tst_dl 