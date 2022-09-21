import numpy as np
import sys
import os
import argparse
import shutil

import torch
from torch.nn import functional as torch_f
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv
import kornia as K

biggan_path = '/home/arash/Software/repositories/others/gans/pytorch-pretrained-BigGAN/'
sys.path.append(biggan_path)

from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, convert_to_images,
                                       one_hot_from_int)

import warnings

warnings.filterwarnings('ignore')

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

logging.basicConfig(level=logging.CRITICAL)

import io
from PIL import Image
from matplotlib import pyplot as plt

import clip

text_templates = [
    'itap of a {} {}.',
    'a bad photo of the {} {}.',
    'a origami {} {}.',
    'a photo of the large {} {}.',
    'a {} {} in a video game.',
    'art of the {} {}.',
    'a photo of the small {} {}.',
]

colour_labels = [
    'pink', 'red', 'orange', 'brown', 'yellow', 'green', 'blue', 'purple',
    'white', 'grey', 'black'
]


def _load_clip_model():
    clip_model, clip_preprocess = clip.load("ViT-B/32")  # ViT-B/32 L/14
    clip_model.cuda().eval()

    for param in clip_model.parameters():
        param.requires_grad = False

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    return clip_model, mean, std


def zeroshot_classifier(classnames, object_name, templates, clip_model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname, object_name) for template in templates]
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def clip_response(templates, colours, object_name, images, clip_model):
    text_features = zeroshot_classifier(colours, object_name, templates, clip_model).float().T

    image_features = clip_model.encode_image(images).float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    text_probs_raw = 100.0 * image_features @ text_features.T

    return text_probs_raw


def plot_results(similarity, labels, original_images, row_wise, figmag=1):
    count = len(labels)

    fig = plt.figure(figsize=(8 * figmag, 8 * figmag))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(similarity, vmin=similarity.min(), vmax=similarity.max(), cmap='PiYG')
    ax.set_yticks(range(count))
    ax.set_yticklabels(labels, fontsize=18)
    ax.set_xticks([])
    if row_wise:
        ax.set_yticks(np.arange(-.5, similarity.shape[0], 1), minor=True)
    else:
        ax.set_xticks(np.arange(-.5, similarity.shape[1], 1), minor=True)

    for i, image in enumerate(original_images):
        ax.imshow(image.numpy().transpose(1, 2, 0),
                  extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            ax.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        ax.spines[side].set_visible(False)

    ax.set_xlim([-0.5, len(original_images) - 0.5])
    ax.set_ylim([count - 0.5, -1.5])

    ax.grid(which='minor', color='orange', linestyle='-', linewidth=2)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf


def inv_normalise(img, mean, std):
    img_inv = img.clone()
    for i in range(3):
        img_inv[i] = (img_inv[i] * std[i]) + mean[i]
    return img_inv


def main(argv):
    parser = argparse.ArgumentParser(description='Testing CLIP.')
    # parser.add_argument('--val_dir', required=True, type=str)
    # parser.add_argument('--munsell_path', required=True, type=str)
    # parser.add_argument('--text_path', required=True, type=str)
    parser.add_argument('--colour_path', type=str)
    parser.add_argument('--clip_gt', type=int)
    parser.add_argument('--out_dir', default='outputs', type=str)
    parser.add_argument('--clip_arch', default='ViT-B/32', type=str)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--num_iters', default=10000, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int)

    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copy('ill_gen.py', args.out_dir)

    # CLIP-Model
    clip_model, clip_mean, clip_std = _load_clip_model()

    # Load pre-trained model tokenizer (vocabulary)
    gan_model = BigGAN.from_pretrained('biggan-deep-256')
    gan_model.to('cuda')
    gan_model.train()

    # optimiser
    params_to_optimize = [{'params': [p for p in gan_model.parameters()]}]
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr)

    # which colour to make the illusion, NOW JUST BLUE
    # other_colours_ind = np.delete(np.arange(len(colour_labels)), [args.clip_gt])
    # min_hue, max_hue = (np.array([173, 253]) / 360) * 2 * np.pi
    focal_colours = np.loadtxt(args.colour_path, delimiter=',')
    if focal_colours.shape[0] == 3:
        focal_colours = focal_colours.T
    colour_illusion = focal_colours[args.clip_gt]

    batch_size = args.batch_size
    num_iterations = int(args.num_iters / args.batch_size)

    tb_writer = SummaryWriter(os.path.join(args.out_dir, 'tb_logger'))

    losses = []
    for iter_ind in range(num_iterations):
        # Prepare a input
        class_vector = one_hot_from_names(['dress'] * batch_size, batch_size=batch_size)
        truncation = 1.0
        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)

        # All in tensors
        noise_vector = torch.from_numpy(noise_vector)
        class_vector = torch.from_numpy(class_vector)

        # If you have a GPU, put everything on cuda
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')

        # Generate an image
        output = gan_model(noise_vector, class_vector, truncation)

        # normalising the image to 0-1
        output = (output + 1) / 2.0
        # loss colour to perform before clip
        # output_hsv = K.color.rgb_to_hsv(output)
        # hue_mask = (
        #         (output_hsv[:, 0] >= min_hue) & (output_hsv[:, 0] <= max_hue) &
        #         (output_hsv[:, 1] >= 0.50) & (output_hsv[:, 2] >= 0.50)
        # )
        # hue_mask = hue_mask.unsqueeze(dim=1).repeat(1, 3, 1, 1)
        # loss_colour = torch.sum(hue_mask)
        target = torch.ones(output.shape)
        for i in range(3):
            target[:, i] = colour_illusion[i]
        target = target.cuda()
        loss_colour = torch_f.mse_loss(output, target)

        # preparing the input for CLIP
        target = torch.tensor([args.clip_gt] * output.shape[0]).cuda()
        output = torch_f.interpolate(output, (224, 224))
        for i in range(3):
            output[:, i] = (output[:, i] - clip_mean[i]) / clip_std[i]

        text_probs_raw = clip_response(text_templates, colour_labels, 'object', output, clip_model)
        # colour_probs = (1 * text_probs_raw).softmax(dim=-1).T
        # colour_probs = torch.mean(colour_probs, dim=1)
        # loss_illusion = torch.mean(colour_probs[other_colours_ind]) / colour_probs[args.clip_gt]
        loss_illusion = torch_f.cross_entropy(text_probs_raw, target)
        loss = loss_illusion * 0.5 + loss_colour * 0.5

        losses.append(loss.detach().item())
        tb_writer.add_scalar("{}".format('loss'), loss.detach().item(), iter_ind)
        tb_writer.add_scalar("{}".format('loss_ill'), loss_illusion.detach().item(), iter_ind)
        tb_writer.add_scalar("{}".format('loss_colour'), loss_colour.detach().item(), iter_ind)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if np.mod(iter_ind, 100) == 0:
            print('[%d]/[%d] %.2f' % (iter_ind, num_iterations, np.mean(losses)))
            img_inv = [inv_normalise(img.cpu(), clip_mean, clip_std) for img in output.detach()]
            for j in range(min(16, len(img_inv))):
                img_name = 'img%03d' % j
                tb_writer.add_image('{}'.format(img_name), img_inv[j], iter_ind)

            probs = (1 * text_probs_raw.detach()).softmax(dim=-1).T
            clip_res_buf = plot_results(probs.cpu(), colour_labels, img_inv, False, figmag=1)
            tv_image = tv.transforms.ToTensor()(Image.open(clip_res_buf))
            tb_writer.add_image('{}'.format('clip_pred'), tv_image, iter_ind)

        if np.mod(iter_ind, 1000) == 0:
            gan_model_path = '%s/gan_model.pth' % (args.out_dir)
            torch.save(gan_model.state_dict(), gan_model_path)

    gan_model_path = '%s/gan_model.pth' % (args.out_dir)
    torch.save(gan_model.state_dict(), gan_model_path)
    return np.array(losses)


if __name__ == '__main__':
    main(sys.argv[1:])
