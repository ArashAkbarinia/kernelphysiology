"""
Berkeley Adobe Perceptual Patch Similarity (BAPPS) dataset.
Download the dataset from https://github.com/richzhang/PerceptualSimilarity .
"""

import os
import ntpath
import numpy as np
from scipy.io import loadmat

import cv2
from torchvision import datasets as tdatasets

from kernelphysiology.utils import path_utils


class BAPPS2afc(tdatasets.VisionDataset):
    def __init__(self, split, distortion, **kwargs):
        super(BAPPS2afc, self).__init__(**kwargs)
        self.split = split
        self.distortion = distortion
        self.loader = tdatasets.folder.pil_loader
        self.root = os.path.join(self.root, '2afc', split, distortion)
        self.ref_imgs = path_utils.image_in_folder(self.root + '/ref/')
        print('Read %d images.' % len(self.ref_imgs))

    def __getitem__(self, index):
        path_ref = self.ref_imgs[index]
        img_ref = self.loader(path_ref)
        img_ref = np.asarray(img_ref).copy()
        base_name = ntpath.basename(path_ref)[:-4]

        path_p0 = '%s/p0/%s.png' % (self.root, base_name)
        img_p0 = self.loader(path_p0)
        img_p0 = np.asarray(img_p0).copy()
        path_p1 = '%s/p1/%s.png' % (self.root, base_name)
        img_p1 = self.loader(path_p1)
        img_p1 = np.asarray(img_p1).copy()
        # a few images are of size 252, so we convert themt o 256
        if img_ref.shape[0] != 256:
            img_ref = cv2.resize(img_ref, (256, 256))
            img_p0 = cv2.resize(img_p0, (256, 256))
            img_p1 = cv2.resize(img_p1, (256, 256))

        path_judge = '%s/judge/%s.npy' % (self.root, base_name)
        gt = np.load(path_judge)

        if self.transform is not None:
            img_ref, img_p0, img_p1 = self.transform([img_ref, img_p0, img_p1])

        return img_ref, img_p0, img_p1, gt

    def __len__(self):
        return len(self.ref_imgs)


class BAPPSjnd(tdatasets.VisionDataset):
    def __init__(self, split, distortion, **kwargs):
        super(BAPPSjnd, self).__init__(**kwargs)
        self.split = split
        self.distortion = distortion
        self.loader = tdatasets.folder.pil_loader
        self.root = os.path.join(self.root, 'jnd', split, distortion)
        self.img0_paths = path_utils.image_in_folder(self.root + '/p0/')
        print('Read %d images.' % len(self.img0_paths))

    def __getitem__(self, index):
        path0 = self.img0_paths[index]
        img0 = self.loader(path0)
        img0 = np.asarray(img0).copy()
        base_name = ntpath.basename(path0)[:-4]

        path1 = '%s/p1/%s.png' % (self.root, base_name)
        img1 = self.loader(path1)
        img1 = np.asarray(img1).copy()
        # a few images are of size 252, so we convert themt o 256
        if img0.shape[0] != 256:
            img0 = cv2.resize(img0, (256, 256))
            img1 = cv2.resize(img1, (256, 256))

        path_same = '%s/same/%s.npy' % (self.root, base_name)
        gt = np.load(path_same)

        if self.transform is not None:
            img0, img1 = self.transform([img0, img1])

        return img0, img1, gt

    def __len__(self):
        return len(self.img0_paths)


class LIVE(tdatasets.VisionDataset):
    def __init__(self, part, **kwargs):
        super(LIVE, self).__init__(**kwargs)
        self.part = part
        self.loader = tdatasets.folder.pil_loader
        self.root = os.path.join(self.root, part)
        self.img_dir = os.path.join(self.root, 'imgs')
        self.image_list = loadmat(self.root + '/Imagelists.mat')
        self.scores = loadmat(self.root + '/Scores.mat')
        print('Number of images: %d' % len(self.image_list['distimgs']))

    def __getitem__(self, index):
        img_name = self.image_list['distimgs'][index][0][0]
        img_path = os.path.join(self.img_dir, img_name)
        img = self.loader(img_path)
        img = np.asarray(img).copy()

        ref_name = self.image_list['refimgs'][index][0][0]
        ref_path = os.path.join(self.img_dir, ref_name)
        ref = self.loader(ref_path)
        ref = np.asarray(ref).copy()

        gt = [
            self.scores['DMOSscores'][0][index],
            self.scores['Zscores'][0][index]
        ]

        if self.transform is not None:
            ref, img = self.transform([ref, img])

        return ref, img, gt

    def __len__(self):
        return len(self.image_list['distimgs'])