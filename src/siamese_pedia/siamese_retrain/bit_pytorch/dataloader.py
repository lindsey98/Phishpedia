import torch.utils.data as data
from PIL import Image, ImageOps
import pickle
import numpy as np
import os

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, label_dict, transform=None, grayscale=False):
        
        self.transform = transform
        self.data_root = data_root
        self.grayscale = grayscale
        data_list = [x.strip('\n') for x in open(data_list).readlines()]

        with open(label_dict, 'rb') as handle:
            self.label_dict = pickle.load(handle)

        self.classes = list(self.label_dict.keys())

        self.n_data = len(data_list)

        self.img_paths = []
        self.labels = []

        for data in data_list:
            image_path = data
            label = image_path.split('/')[0]
            self.img_paths.append(image_path)
            self.labels.append(label)

    def __getitem__(self, item):

        img_path, label= self.img_paths[item], self.labels[item]
        img_path_full = os.path.join(self.data_root, img_path)
        if self.grayscale:
            img = Image.open(img_path_full).convert('L').convert('RGB')
        else:
            img = Image.open(img_path_full).convert('RGB')

        img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

        # label = np.array(label,dtype='float32')
        label = self.label_dict[label]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.n_data