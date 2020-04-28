import numpy as np
import os
import sys

import torch
from torchvision import transforms

from skimage import io

from kernelphysiology.dl.pytorch.vaes import model
from kernelphysiology.dl.pytorch.utils.preprocessing import inv_normalise_tensor
from kernelphysiology.transformations import colour_spaces

main_dir = '/home/arash/Desktop/nips/'

batch_size = int(sys.argv[2])
colour_space = sys.argv[1]
model_path = main_dir + 'models/' + colour_space + '/checkpoints/model_40.pth'
weights_rgb = torch.load(model_path, map_location='cpu')

net_rgb = model.VQ_CVAE(128, k=512, in_chns=3)
net_rgb.load_state_dict(weights_rgb)

from kernelphysiology.dl.pytorch.utils import cv2_transforms
from kernelphysiology.dl.pytorch.utils import preprocessing

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transform_funcs = transforms.Compose(
    [cv2_transforms.Resize(256), cv2_transforms.CenterCrop(224),
     cv2_transforms.ToTensor(),
     cv2_transforms.Normalize(mean, std)
     ])

intransform_funs = []
mosaic_pattern = None
if mosaic_pattern is not None:
    intransform_funs.append(
        preprocessing.MosaicTransformation(mosaic_pattern)
    )
intransform = transforms.Compose(intransform_funs)
outtransform_funs = []
inv_func = None
colour_space_lab = 'lab'
if colour_space_lab is not None:
    outtransform_funs.append(
        preprocessing.ColourTransformation(None, colour_space_lab)
    )
outtransform_lab = transforms.Compose(outtransform_funs)

outtransform_funs_hsv = []
colour_space_hsv = 'hsv'
if colour_space_hsv is not None:
    outtransform_funs_hsv.append(
        preprocessing.ColourTransformation(None, colour_space_hsv)
    )
outtransform_hsv = transforms.Compose(outtransform_funs_hsv)

import cv2

out_dir = main_dir + '/outs/'

imagenet_path = '/home/arash/Software/imagenet/'
validation_set = imagenet_path + 'raw-data/validation/'

# net_rgb.cuda()
net_rgb.eval()

org_img = []
rec_rgb = []
diff_rgb = []

import data_loaders

test_loader = torch.utils.data.DataLoader(
    data_loaders.ImageFolder(
        root=validation_set,
        intransform=None,
        outtransform=None,
        transform=transform_funcs
    ),
    batch_size=batch_size, shuffle=False
)
with torch.no_grad():
    for i, (img_readies, img_target, img_paths) in enumerate(test_loader):
        img_readies = img_readies  # .cuda()
        out_rgb = net_rgb(img_readies)
        out_rgb = out_rgb[0].detach().cpu()
        img_readies = img_readies.detach().cpu()

        for img_ind in range(out_rgb.shape[0]):
            img_path = img_paths[img_ind]
            img_ready = img_readies[img_ind].unsqueeze(0)

            cat_in_dir = os.path.dirname(img_path) + '/'

            cat_out_dir = cat_in_dir.replace(validation_set, out_dir)
            if not os.path.exists(cat_out_dir):
                os.mkdir(cat_out_dir)

            org_img_tmp = inv_normalise_tensor(img_ready, mean, std)
            org_img_tmp = org_img_tmp.numpy().squeeze().transpose(1, 2, 0)
            org_img_tmp = org_img_tmp * 255
            org_img_tmp = org_img_tmp.astype('uint8')
            # org_img.append(org_img_tmp)

            org_dir = cat_out_dir + '/org/'
            if not os.path.exists(org_dir):
                os.mkdir(org_dir)
            rgb_dir = cat_out_dir + '/' + colour_space[4:] + '/'
            if not os.path.exists(rgb_dir):
                os.mkdir(rgb_dir)

            if os.path.exists(img_path.replace(cat_in_dir, rgb_dir)):
                rec_rgb_tmp = cv2.imread(img_path.replace(cat_in_dir, rgb_dir))
                rec_rgb_tmp = cv2.cvtColor(rec_rgb_tmp, cv2.COLOR_BGR2RGB)
            else:
                rec_rgb_tmp = inv_normalise_tensor(
                    out_rgb[img_ind].unsqueeze(0), mean, std)
                rec_rgb_tmp = rec_rgb_tmp.numpy().squeeze().transpose(1, 2, 0)
                if colour_space == 'rgb2lab':
                    rec_rgb_tmp = rec_rgb_tmp * 255
                    rec_rgb_tmp = rec_rgb_tmp.astype('uint8')
                    rec_rgb_tmp = cv2.cvtColor(rec_rgb_tmp, cv2.COLOR_LAB2RGB)
                elif colour_space == 'rgb2hsv':
                    rec_rgb_tmp = rec_rgb_tmp * 255
                    rec_rgb_tmp = rec_rgb_tmp.astype('uint8')
                    rec_rgb_tmp = cv2.cvtColor(rec_rgb_tmp, cv2.COLOR_HSV2RGB)
                elif colour_space == 'rgb2dkl':
                    rec_rgb_tmp = colour_spaces.dkl012rgb(rec_rgb_tmp)
            ## rec_rgb.append(rec_rgb_tmp)

            tmp_org = org_img_tmp.astype('float') / 255
            tmp_rgb = rec_rgb_tmp.astype('float') / 255
            diff_rgb.append(np.array((tmp_org - tmp_rgb) ** 2).mean())

            if not os.path.exists(img_path.replace(cat_in_dir, org_dir)):
                io.imsave(img_path.replace(cat_in_dir, org_dir), org_img_tmp)
            if not os.path.exists(img_path.replace(cat_in_dir, rgb_dir)):
                io.imsave(img_path.replace(cat_in_dir, rgb_dir), rec_rgb_tmp)
        np.savetxt('' + colour_space[4:] + '.txt', np.array(diff_rgb))
