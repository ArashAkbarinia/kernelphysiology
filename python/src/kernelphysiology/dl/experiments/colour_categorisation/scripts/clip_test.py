import numpy as np
import glob
import argparse
import ntpath
import sys
import os

from skimage import io
from PIL import Image

import torch
import clip


def main(argv):
    parser = argparse.ArgumentParser(description='Testing CLIP.')
    parser.add_argument('--val_dir', required=True, type=str)
    parser.add_argument('--munsell_path', required=True, type=str)
    parser.add_argument('--text_path', required=True, type=str)
    parser.add_argument('--out_dir', default='outputs', type=str)
    parser.add_argument('--clip_arch', default='ViT-B/32', type=str)

    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    _main_worker(args)


def _main_worker(args):
    model, preprocess = clip.load(args.clip_arch)
    model.cuda().eval()

    munsell_img = io.imread(args.munsell_path)[1:-1, 1:]

    labels = np.loadtxt(args.text_path, dtype=str)

    out_file = '%s/text_probs.npy' % args.out_dir
    old_results = np.load(out_file, allow_pickle=True)[0] if os.path.exists(out_file) else dict()

    all_text_probls = dict()
    for img_path in sorted(glob.glob(args.val_dir + '/*.gif')):
        image_name = ntpath.basename(img_path)[:-4]
        if image_name in old_results.keys():
            all_text_probls[image_name] = old_results[image_name]
            continue
        else:
            print(img_path)
            text_probs = _one_image(img_path, munsell_img, labels, model, preprocess)
            all_text_probls[image_name] = text_probs

        np.save(out_file, [all_text_probls])


def _one_image(img_path, munsell_img, labels, model, preprocess):
    image = io.imread(img_path)
    image_mask = image == 255

    grey_img = np.zeros((*image.shape, 3), dtype='uint8')
    grey_img[:, :] = 128

    images = []
    for i in range(munsell_img.shape[0]):
        for j in range(munsell_img.shape[1]):
            image_vis = grey_img.copy()
            image_vis[image_mask] = munsell_img[i, j]
            images.append(preprocess(Image.fromarray(image_vis)))

    text_descriptions = [f"This is a {label} object" for label in labels]
    text_tokens = clip.tokenize(text_descriptions).cuda()
    image_input = torch.tensor(np.stack(images)).cuda()

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs_raw = image_features @ text_features.T

    return text_probs_raw.cpu().numpy()


if __name__ == '__main__':
    main(sys.argv[1:])
