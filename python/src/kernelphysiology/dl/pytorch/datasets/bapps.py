"""
Berkeley Adobe Perceptual Patch Similarity (BAPPS) dataset.
Download the dataset from https://github.com/richzhang/PerceptualSimilarity .
"""

import os
import ntpath
import numpy as np

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

        path_judge = '%s/judge/%s.npy' % (self.root, base_name)
        gt = np.load(path_judge)

        if self.transform is not None:
            img_ref, img_p0, img_p1 = self.transform([img_ref, img_p0, img_p1])

        return img_ref, img_p0, img_p1, gt

    def __len__(self):
        return len(self.ref_imgs)
