"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

modified by Yihao Zhao
"""
import torch.utils.data as data
import os.path

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
import glob
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root_A, root_B, transform=None, return_paths=False,
                 loader=default_loader):
        # print(root_A)
        # imgs_A = sorted(make_dataset(root_A))
        # imgs_B = sorted(make_dataset(root_B))
        # if len(root_A) == 0:
        #     raise(RuntimeError("Found 0 images in: " + root_A + "\n"
        #                        "Supported image extensions are: " +
        #                        ",".join(IMG_EXTENSIONS)))

        imgs_A = glob.glob(f'{root_A}/*')
        imgs_B = glob.glob(f'{root_B}/*')

        self.root = root_A
        self.imgs = imgs_A
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

        self.path_A = []
        self.path_B = []

        """
        inner class image 셔플
        """
        # print(imgs_A)
        for i in range(len(imgs_A)):
            A = glob.glob(f'{imgs_A[i]}/*.png')
            B = glob.glob(f'{imgs_B[i]}/*.png')
            B = self.shuffle_image(A, B)

            self.path_A = self.path_A + A
            self.path_B = self.path_B + B

    def shuffle_image(self, A, B):
        random.shuffle(B)
        for i in range(len(A)):
            if A[i] == B[i]:
                return self.shuffle_image(A, B)
        return B

    def __getitem__(self, index):
        path_A = self.path_A[index]
        img_A = self.loader(path_A)

        path_B = self.path_B[index]
        img_B = self.loader(path_B)

        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        if self.return_paths:
            return path_A
        else:
            return img_A, img_B

    def __len__(self):
        return len(self.imgs)
