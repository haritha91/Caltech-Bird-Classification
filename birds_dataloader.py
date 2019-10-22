import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageOps

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class CaltechBirds(Dataset):
    """Caltech-UCSD 200 Birds dataset."""

    def __init__(self, data_split):
        """
        Args:
            data_split (string): desired data split
        """
        self.train_image_list_path = './lists/train.txt'
        self.test_image_list_path = './lists/test.txt' 
        self.data_split = data_split

        #Image transformation
        self.transform = transforms.Compose([transforms.Resize((224,224)), 
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        #Image resize with envelope mode
        def imageResizeAndPadding(im,desired_size):
            old_size = im.size  # old_size[0] is in (width, height) format
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])
            im = im.resize(new_size, Image.ANTIALIAS)
            delta_w = desired_size - new_size[0]
            delta_h = desired_size - new_size[1]
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            new_im = ImageOps.expand(im, padding)
            return new_im

        #Read train-test image lists
        self.train_images = open(self.train_image_list_path).readlines()
        self.test_images = open(self.test_image_list_path).readlines()

        #create data lists
        if self.data_split =='train':
            self.image_list = self.train_images
        if self.data_split == 'test':
            self.image_list = self.test_images

        #create train images array
        self.data_list = list()
        for i in range(len(self.image_list)):
            _train_set = self.image_list[i].split("\n")[0]
            species = self.image_list[0].split('/')[0]
            id, name = int(species.split('.')[0])-1, species.split('.')[1]
            _data_list = ['./images/' + _train_set, id, name]
            self.data_list.append(_data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        


        return sample