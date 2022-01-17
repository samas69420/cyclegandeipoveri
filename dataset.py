import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC
from PIL import Image
import random
import os
import torch
from config import *

# DATASET E PREPROCESSING

class BaseDataset(data.Dataset, ABC): # solo per farlo come l'originale
    def __init__(self):
        pass

class UnalignedDataset(BaseDataset):
    def __init__(self):
        BaseDataset.__init__(self)
        #self.dir_A = "datasets/monet2photo/trainA"
        #self.dir_B = "datasets/monet2photo/trainB" 
        self.dir_A = f"{datasetpath}/trainA"
        self.dir_B = f"{datasetpath}/trainB" 
        self.A_paths = sorted(make_dataset(self.dir_A))   
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)  
        self.B_size = len(self.B_paths)  
        input_nc = 3
        output_nc = 3
        self.transform = get_transform()
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size] 
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform(A_img)
        B = self.transform(B_img) # a e b vengono trasformati in tensori da transform
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
    def __len__(self):
        return max(self.A_size, self.B_size)

class CustomLoader():
    def __init__(self):
        self.dataset = UnalignedDataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = batchsize,
            shuffle = False, # con batch size 1 dovrebbe non servire a niente questo
            num_workers = 1 # num threads
        )
    def __iter__(self): # quindi iterare su customloader dovrebbe essere come iterare su dataloader?
        for data in self.dataloader:
            yield data
    def __len__(self):
        return len(self.dataset)

def make_dataset(dir):
    paths = []
    for root,subdirs,elements in os.walk(dir):
        for element in elements:
            if element.endswith(".jpg"):
                paths.append(root+'/'+element) 
    return paths

def get_transform(): # no_flip = false params = None convert = True grayscale = False
    method=Image.BICUBIC #originale ma mi da warning
    #method = transforms.InterpolationMode.BICUBIC # dovrebbe fare la stessa cosa di image.bicubic ma senza warning
    transform_list = []
    transform_list.append(transforms.Resize([256,256], method))
    transform_list.append(transforms.RandomCrop(256))
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
