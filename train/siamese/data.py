import torch.utils.data as data
from PIL import Image, ImageOps
import pickle
import numpy as np
import os
import torchvision as tv


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, label_dict, transform=None, grayscale=False):

        self.transform = transform
        self.data_root = data_root
        self.grayscale = grayscale
        data_list = [x.strip('\n') for x in open(data_list).readlines()]

        with open(label_dict, 'rb') as handle:
            self.label_dict = pickle.load(handle)

        self.classes = list(self.label_dict.keys())

        self.img_paths = []
        self.labels = []

        for data in data_list:
            image_path = data
            label = os.path.basename(os.path.dirname(image_path))
            if not os.path.exists(os.path.join(self.data_root, image_path)):
                continue
            self.img_paths.append(image_path)
            self.labels.append(label)

        self.n_data = len(self.img_paths)


    def __getitem__(self, item):

        img_path, label = self.img_paths[item], self.labels[item]
        img_path_full = os.path.join(self.data_root, img_path)
        img = Image.open(img_path_full).convert('RGB')

        img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2),
                              fill=(255, 255, 255))

        label = self.label_dict[label]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.n_data


if __name__ == '__main__':
    # prepare data
    ## convert label2id
    label2id_dict = {}
    for it, line in enumerate(open('./datasets/siamese_training/List/Logo-2K+classes.txt').readlines()):
        label2id_dict[line.strip()] = it

    import pickle
    with open('./datasets/siamese_training/List/logo2k_labeldict.pkl', 'wb') as f:
        pickle.dump(label2id_dict, f)

    # convert label2id
    label2id_dict = {}
    for it, brand in enumerate(os.listdir('./models/expand_targetlist')):
        label2id_dict[brand] = it

    import pickle
    with open('./datasets/siamese_training/target_dict.pkl', 'wb') as f:
        pickle.dump(label2id_dict, f)



