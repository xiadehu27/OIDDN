import scipy.io as sio
import os
from torch.utils.data import Dataset
import h5py

# h5py based dataset loader
class DeephomographyDataset(Dataset):
    def __init__(self,hdf5file,labels_key='data',
                transform=None,loadAllToMemory=False):
    
        self.hdf5file=hdf5file
        self.labels_key=labels_key
        self.transform=transform
        self.loadAllToMemory = loadAllToMemory

        if loadAllToMemory :
            with h5py.File(self.hdf5file,'r') as db:                
                self.samples = list(db[self.labels_key])

    def __len__(self):
        
        with h5py.File(self.hdf5file, 'r') as db:
            lens=len(db[self.labels_key])
        return lens

    def __getitem__(self, idx):

        if self.loadAllToMemory :
            sample=self.samples[idx]
            if self.transform:
                sample=self.transform(sample)
            return sample

        with h5py.File(self.hdf5file,'r') as db:
            label=db[self.labels_key][idx]
        sample=label
        if self.transform:
            sample=self.transform(sample)
        return sample  