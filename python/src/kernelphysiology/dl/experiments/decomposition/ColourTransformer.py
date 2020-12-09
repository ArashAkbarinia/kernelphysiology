"""
A model to learn parameters of a LAB-like transformation.
"""

import torch
from torch import nn
import torch.utils.data

from kernelphysiology.dl.pytorch.optimisations import losses


class LabTransformer(nn.Module):
    def __init__(self, trans_mat=None, ref_white=None, distortion=None,
                 trans_mat_gr=True, ref_white_gr=True, distortion_gr=True,
                 linear=True):
        super(LabTransformer, self).__init__()

        if trans_mat is None:
            trans_mat = torch.rand((3, 3))
        else:
            trans_mat = torch.tensor(trans_mat)
        self.trans_mat = nn.Parameter(trans_mat, requires_grad=trans_mat_gr)

        if ref_white is None:
            ref_white = torch.rand(3)
        else:
            ref_white = torch.tensor(ref_white)
        self.ref_white = nn.Parameter(ref_white, requires_grad=ref_white_gr)

        self.linear = linear
        if not self.linear:
            if distortion is None:
                distortion = torch.rand(5)
            else:
                distortion = torch.tensor(distortion)
            self.distortion = nn.Parameter(
                distortion, requires_grad=distortion_gr
            )

        self.rec_mse = 0
        self.org_mse = 0
        # vals = [116.0, 16.0, 500.0, 200.0, 0.2068966]
        # vals = [e / 500 for e in vals]

    def forward(self, x):
        x = self.rgb2rnd(x)
        x = torch.tanh(x)
        return x

    def rgb2rnd(self, rgb):
        lin_arr = rgb.clone()
        for i in range(3):
            x_r = rgb[:, 0:1, ] * self.trans_mat[i, 0]
            y_g = rgb[:, 1:2, ] * self.trans_mat[i, 1]
            z_b = rgb[:, 2:3, ] * self.trans_mat[i, 2]
            lin_arr[:, i:i + 1, ] = x_r + y_g + z_b

        # scale by tristimulus values of the reference white point
        for i in range(3):
            lin_arr[:, i:i + 1, ] /= (self.ref_white[i] + 1e-4)

        if not self.linear:
            vals = self.distortion + 1e-4
            eta = vals[4]
            m = (1 / 3) * (eta ** -2)
            t0 = eta ** 3

            # Nonlinear distortion and linear transformation
            mask = lin_arr > t0
            lin_arr[mask] = lin_arr[mask].pow(1 / 3)
            lin_arr[~mask] = m * lin_arr[~mask] + (vals[1] / vals[0])

            x = lin_arr[:, 0:1, ]
            y = lin_arr[:, 1:2, ]
            z = lin_arr[:, 2:3, ]

            # Vector scaling
            L = (vals[0] * y) - vals[1]
            a = vals[2] * (x - y)
            b = vals[3] * (y - z)

            nonlin_arr = lin_arr.clone()
            nonlin_arr[:, 0:1, ] = L
            nonlin_arr[:, 1:2, ] = a
            nonlin_arr[:, 2:3, ] = b

            return nonlin_arr
        return lin_arr

    def rnd2rgb(self, rnd, clip=False):
        rnd = torch.atanh(rnd)
        lin_arr = rnd.detach().clone()

        if not self.linear:
            vals = self.distortion + 1e-4
            eta = vals[4]

            L = lin_arr[:, 0:1, ]
            a = lin_arr[:, 1:2, ]
            b = lin_arr[:, 2:3, ]
            y = (L + vals[1]) / vals[0]
            x = (a / vals[2]) + y
            z = y - (b / vals[3])

            lin_arr[:, 0:1, ] = x
            lin_arr[:, 1:2, ] = y
            lin_arr[:, 2:3, ] = z

            mask = lin_arr > eta
            lin_arr[mask] = lin_arr[mask].pow(3.)
            lin_arr[~mask] = (lin_arr[~mask] - (vals[1] / vals[0])) * 3 * (
                    eta ** 2)

        # rescale to the reference white (illuminant)
        for i in range(3):
            lin_arr[:, i:i + 1, ] *= (self.ref_white[i] + 1e-4)

        rgb = lin_arr.clone()
        rgb_transform = torch.inverse(self.trans_mat.detach())
        for i in range(3):
            x_r = lin_arr[:, 0:1, ] * rgb_transform[i, 0]
            y_g = lin_arr[:, 1:2, ] * rgb_transform[i, 1]
            z_b = lin_arr[:, 2:3, ] * rgb_transform[i, 2]
            rgb[:, i:i + 1, ] = x_r + y_g + z_b

        if clip:
            rgb[torch.isnan(rgb)] = 0
            rgb = (rgb * 0.5) + 0.5
            rgb[rgb < 0] = 0
            rgb[rgb > 1] = 1
        return rgb

    def loss_function(self, out_space, in_space, model_rec):
        self.rec_mse = losses.decomposition_loss(model_rec, out_space)

        target_rgb = self.rnd2rgb(out_space.detach().clone(), clip=True)
        target_mse = losses.decomposition_loss(target_rgb, in_space)

        model_rgb = self.rnd2rgb(model_rec.detach().clone(), clip=True)
        model_mse = losses.decomposition_loss(model_rgb, in_space)

        self.org_mse = target_mse + model_mse

        return self.rec_mse + self.org_mse

    def latest_losses(self):
        return {'rec_mse': self.rec_mse, 'org_mse': self.org_mse}
