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

"""
For np.log
"""
epsilon     = 1e-10
    """
    
    Args:
        
    Returns:
        
    Ensures:
        
    """

"""
  Transformation applied to grayscale images (Not used)
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

"""
Tranformations applied in Dataset class.  Passed in as parameter to get_preprocessor
"""

# Image loader 
def pil_grey_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L'), img
    
def pil_grey_converter(img, img_class):
    return img.convert('L'), img

stgn2_cars_transforms = transforms.Compose([
    transforms.CenterCrop(256)
])

lsun_cars_transforms  = transforms.Compose([
    transforms.Resize(256)
])

def convert_to_256(img, img_class):
    new_img, col_img = pil_grey_converter(img, img_class)
    if (img_class == 'lsun_cars'):
        new_img = lsun_cars_transforms(new_img)
        col_img = lsun_cars_transforms(col_img)
    if (img_class == 'stylegan2_cars'):
        new_img = stgn2_cars_transforms(new_img)
        col_img = stgn2_cars_transforms(col_img)
    return np.asarray(new_img), col_img

"""
Dataset pre-processor for Durall deepfake detection model.
"""
    
class DeepFakePreProcessor(ImageFolder):
    def __init__(self, root, img_transforms, output_size):
        print(root)
        super(DeepFakePreProcessor, self).__init__(root=root,
                                              transform=None)
        self.img_transforms = img_transforms
        self.images = self.samples
        self.output_sz = output_size
        
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
        
        img_class = self.classes[t]
        gray_img, img = self.img_transforms(img, img_class)
        
        ms_img = np_magnitude_spectrum(gray_img)
        rad_p = np_radial_profile(
            ms_img, 
            center=(ms_img.shape[0]/2, ms_img.shape[1]/2), 
            pad=True, 
            length=self.output_sz)
            
        return rad_p, self.switcher[img_class], img_class, np.asarray(img), ms_img
    
"""
    get_preprocessor
    Helper method for creating and returning the preprocessor
"""
def get_preprocessors(image_root=None, 
                   transforms=pil_grey_converter, 
                   batch_size=32, 
                   shuffle=True,
                   destf_hdf5=None,
                   output_size=182,
                   num_workers=0):
    if image_root is None or destf_hdf5 is None:
        raise ValueError("Must specify image_root and destf_hdf5 (Destination HDF5 file)")
            
    
    ds = DeepFakePreProcessor(root=image_root, img_transforms=transforms, output_size=output_size)
    
    return DataLoader(ds,
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      num_workers=num_workers)
    
    
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
    
    def __init__(self, hdf5_path=None, queries=[]):
        super(DeepFakeHDF5Dataset, self).__init__()
        self.data = None
        self.lbls = None
        self.og_d = None
        with h5py.File(hdf5_path, 'r') as hdf5_file: 
            self.og_d = hdf5_file['orgn_data'][:]
            if len(queries) > 0:
                ids = list(set().union(*[[data_origin[data] for data in data_origin.keys() if (query in str(data))] for query in queries]))
                indices = [i for i,og_d in enumerate(self.og_d) if (og_d in ids)]
                self.data = hdf5_file['fft_data'][indices]
                self.lbls = hdf5_file['lbl_data'][indices]
            else:
                self.data = hdf5_file['fft_data'][:]
                self.lbls = hdf5_file['lbl_data'][:]
        
    def __getitem__(self, index):

        sample = torch.from_numpy(self.data[index,:]).float()
        label  = ((torch.FloatTensor([1]) * self.lbls[index]))
        return sample, label
                
    
    def __len__(self):
        return self.data.shape[0]

class DeepFakeHDF5Dataset_SVM(DeepFakeHDF5Dataset):
    
    def __init__(self, hdf5_path=None):
        super(DeepFakeHDF5Dataset_SVM, self).__init__()
        
    def __getitem__(self, index):

        sample = torch.from_numpy(self.data[index,:]).float()
        label  = 1  if self.lbls[index] == 1 else -1
        label  = (torch.FloatTensor([1]) * label)
        return sample, label

        
"""
    get_dataloaders
    Helper method for creating and returning the all three dataloaders
"""
def get_dataloaders(image_root=None,
                   dataset=DeepFakeHDF5Dataset,
                   transforms=data_transforms, 
                   batch_size=_batch_size, 
                   shuffle=_shuffle,
                   full_dataset=False,
                   num_workers=_num_wrks):
    
    ds = dataset
    
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