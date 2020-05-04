"""
Various types of dataloader.
"""

import os
import numpy as np
import random

from scipy.io import loadmat
from PIL import Image

from torchvision import datasets as tdatasets

from kernelphysiology.utils import path_utils


class ImageFolder(tdatasets.ImageFolder):
    def __init__(self, intransform=None, outtransform=None, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.imgs = self.samples
        self.intransform = intransform
        self.outtransform = outtransform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (imgin, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        path, class_target = self.samples[index]
        imgin = self.loader(path)
        imgin = np.asarray(imgin).copy()
        imgout = imgin.copy()
        if self.intransform is not None:
            imgin = self.intransform(imgin)
        if self.outtransform is not None:
            imgout = self.outtransform(imgout)

        if self.transform is not None:
            imgin, imgout = self.transform([imgin, imgout])

        # right now we're not using the class target, but perhaps in the future
        if self.target_transform is not None:
            class_target = self.target_transform(class_target)

        return imgin, imgout, path


class OneFolder(tdatasets.VisionDataset):
    def __init__(self, intransform=None, outtransform=None, **kwargs):
        super(OneFolder, self).__init__(**kwargs)
        self.samples = path_utils.image_in_folder(self.root)
        print('Read %d images.' % len(self.samples))
        self.loader = tdatasets.folder.pil_loader
        self.intransform = intransform
        self.outtransform = outtransform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (imgin, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        path = self.samples[index]
        imgin = self.loader(path)
        imgin = np.asarray(imgin).copy()
        imgout = imgin.copy()
        if self.intransform is not None:
            imgin = self.intransform(imgin)
        if self.outtransform is not None:
            imgout = self.outtransform(imgout)

        if self.transform is not None:
            imgin, imgout = self.transform([imgin, imgout])

        return imgin, imgout, path

    def __len__(self):
        return len(self.samples)


class CategoryImages(OneFolder):
    def __init__(self, root, category, **kwargs):
        kwargs['root'] = os.path.join(root, category)
        super(CategoryImages, self).__init__(**kwargs)


class CocoDetection(tdatasets.CocoDetection):

    def __init__(self, intransform=None, outtransform=None, **kwargs):
        super(CocoDetection, self).__init__(**kwargs)
        self.loader = tdatasets.folder.pil_loader
        self.intransform = intransform
        self.outtransform = outtransform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        imgout = coco.loadAnns(ann_ids)

        path = os.path.join(self.root, coco.loadImgs(img_id)[0]['file_name'])

        imgin = self.loader(path)
        imgin = np.asarray(imgin).copy()

        if self.intransform is not None:
            imgin = self.intransform(imgin)
        if self.outtransform is not None:
            imgout = self.outtransform(imgout)

        if self.transform is not None:
            imgin, imgout = self.transform([imgin, imgout])

        return imgin, imgout, path

    def __len__(self):
        return len(self.ids)


class VOCSegmentation(tdatasets.VOCSegmentation):
    def __init__(self, intransform=None, outtransform=None, **kwargs):
        super(VOCSegmentation, self).__init__(**kwargs)
        self.loader = tdatasets.folder.pil_loader
        self.intransform = intransform
        self.outtransform = outtransform

    def __getitem__(self, index):
        path = self.images[index]
        imgin = self.loader(path)
        imgin = np.asarray(imgin).copy()
        imgout = Image.open(self.masks[index])
        imgout = np.asarray(imgout).copy()

        if self.intransform is not None:
            imgin = self.intransform(imgin)
        if self.outtransform is not None:
            imgout = self.outtransform(imgout)

        if self.transform is not None:
            imgin, imgout = self.transform([imgin, imgout])

        return imgin, imgout, path

    def __len__(self):
        return len(self.images)


class BSDSEdges(tdatasets.VisionDataset):
    def __init__(self, img_list='all_imgs.txt', intransform=None,
                 outtransform=None, **kwargs):
        super(BSDSEdges, self).__init__(**kwargs)
        self.samples = np.loadtxt(os.path.join(self.root, img_list), dtype=str)
        print('Read %d images.' % len(self.samples))
        self.loader = tdatasets.folder.pil_loader
        self.intransform = intransform
        self.outtransform = outtransform
        self.imgs_root = os.path.join(self.root, 'images')
        self.gts_root = os.path.join(self.root, 'groundTruth')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (imgin, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        path = os.path.join(self.imgs_root, self.samples[index] + '.jpg')
        imgin = self.loader(path)
        imgin = np.asarray(imgin).copy()
        edge_path = os.path.join(self.gts_root, self.samples[index] + '.mat')
        imgout = loadmat(edge_path)
        gt_ind = random.randint(0, imgout['groundTruth'].shape[1] - 1)
        imgout = imgout['groundTruth'][0, gt_ind][0][0][1]
        imgout = np.expand_dims(imgout, axis=2)
        if self.intransform is not None:
            imgin = self.intransform(imgin)
        if self.outtransform is not None:
            imgout = self.outtransform(imgout)

        if self.transform is not None:
            imgin, imgout = self.transform([imgin, imgout])

        return imgin, imgout, path

    def __len__(self):
        return len(self.samples)
