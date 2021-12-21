import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader



class Data(object):
    
    def __init__(self, dataset, batch_size=64, validation_size=0.2):
        self.dataset = dataset
        self.validation_size = validation_size
        self.batch_size = batch_size
        
        self.transformations = transforms.Compose([transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(20), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            ])
        self.select_dataset()
    
    def select_dataset(self):
        if self.dataset == 'CIFAR10':
            self.train_data = datasets.CIFAR10('CIFAR10', 
                    train=True, 
                    download=True,
                    transform=self.transformations)
            
            self.test_data = datasets.CIFAR10('CIFAR10', 
                    train=False, 
                    download=True,
                    transform=self.transformations)
            
            print('Length of train_data=',
                    len(self.train_data), 
                    '\nLength of test_data=',
                    len(self.test_data))
        
        
        elif self.dataset == 'MNIST':
            self.train_data = datasets.MNIST('MNIST', 
                    train=True, 
                    download=True,
                    transform=self.transformations)
            
            self.test_data = datasets.MNIST('MNIST', 
                    train=False, 
                    download=True,
                    transform=self.transformations)
            
            print('Length of train_data=',
                    len(self.train_data), 
                    '\nLength of test_data=',
                    len(self.test_data))
            
            
    def download_data(self):
        training_size = len(self.train_data)
        indices = list(range(training_size))
        np.random.shuffle(indices)
        
        index_split = int(np.floor(training_size * self.validation_size))
        validation_indices, training_indices = indices[:index_split], indices[index_split:]
        
        
        training_sample = SubsetRandomSampler(training_indices) 
        validation_sample = SubsetRandomSampler(validation_indices)
        
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, sampler=training_sample)
        valid_loader = DataLoader(self.train_data, batch_size=self.batch_size, sampler=validation_sample)
        test_loader = DataLoader(self.train_data, batch_size=self.batch_size)
        
        return train_loader, valid_loader, test_loader
