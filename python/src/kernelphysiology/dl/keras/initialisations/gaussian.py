'''
Initialising the weights with Gaussian related functoins.
'''

import numpy as np

import keras

from kernelphysiology.filterfactory.gaussian import gaussian_kernel2, gaussian2_gradient1


def initialise_with_gaussian(model, sigmax, sigmay=None, meanx=0, meany=0, theta=0,
                             which_layers=[keras.layers.convolutional.Conv2D, keras.layers.DepthwiseConv2D]):
    for i, layer in enumerate(model.layers):
        if type(layer) in which_layers:
            weights = layer.get_weights()
            (rows, cols, chns, dpts) = weights[0].shape
            # FIXME: with all type of convolution
            if rows > 1:
                print('initialising with Gaussian', layer.name)
                for d in range(dpts):
                    for c in range(chns):
                        sigmax_dc = np.random.uniform(0, sigmax)
                        if sigmay is not None:
                            sigmay_dc = np.random.uniform(0, sigmay)
                        else:
                            sigmay_dc = None
                        meanx_dc = np.random.uniform(-meanx, meanx)
                        meany_dc = np.random.uniform(-meany, meany)
                        theta_dc = np.random.uniform(-theta, theta)
                        g_kernel = gaussian_kernel2(sigmax=sigmax_dc, sigmay=sigmay_dc, meanx=meanx_dc,
                                                    meany=meany_dc, theta=theta_dc, width=rows, threshold=1e-4)
                        weights[0][:, :, c, d] = g_kernel
                model.layers[i].set_weights(weights)
    return model


def initialise_with_gaussian_gradient1(model, sigma, theta, seta,
                                      which_layers=[keras.layers.convolutional.Conv2D, keras.layers.DepthwiseConv2D]):
    for i, layer in enumerate(model.layers):
        if type(layer) in which_layers:
            weights = layer.get_weights()
            (rows, cols, chns, dpts) = weights[0].shape
            # FIXME: with all type of convolution
            if rows > 1:
                print('initialising with Gaussian gradient 1', layer.name)
                for d in range(dpts):
                    for c in range(chns):
                        weights_dc = random_g1(sigma=sigma, theta=theta, seta=seta, width=rows)
                        weights[0][:, :, c, d] = weights_dc
                model.layers[i].set_weights(weights)
    return model


def initialise_with_tog(model, tog_sigma, tog_surround, op,
                        which_layers=[keras.layers.convolutional.Conv2D, keras.layers.DepthwiseConv2D]):
    for i, layer in enumerate(model.layers):
        if type(layer) in which_layers:
            weights = layer.get_weights()
            (rows, cols, chns, dpts) = weights[0].shape
            # FIXME: with all type of convolution
            if rows > 1:
                print('initialising with ToG', layer.name)
                for d in range(dpts):
                    for c in range(chns):
                        weights_dc = random_tog(tog_sigma, tog_surround, op=op, width=rows)
                        weights[0][:, :, c, d] = weights_dc
                model.layers[i].set_weights(weights)
    return model


def random_tog(tog_sigma, tog_surround, op, width):
    sigmax1 = np.random.uniform(0, tog_sigma)
    g1 = gaussian_kernel2(sigmax=sigmax1, sigmay=None, meanx=0,
                          meany=0, theta=0, width=width, threshold=1e-4)
    sigmax2 = np.random.uniform(0, tog_sigma * tog_surround)
    g2 = gaussian_kernel2(sigmax=sigmax2, sigmay=None, meanx=0,
                          meany=0, theta=0, width=width, threshold=1e-4)
    if type(op) is tuple:
        wg2 = np.random.uniform(*op)
    else:
        wg2 = op
    return g1 + wg2 * g2


def random_g1(sigma, theta, seta, width):
    sigma_dc = np.random.uniform(0, sigma)
    theta_dc = np.random.uniform(-theta, theta)
    seta_dc = np.random.uniform(0, seta)
    g1_kernel = gaussian2_gradient1 (sigma=sigma_dc, theta=theta_dc, seta=seta_dc, width=width, threshold=1e-4)
    return g1_kernel


def initialise_with_g1g2(model, args,
                         which_layers=[keras.layers.convolutional.Conv2D, keras.layers.DepthwiseConv2D]):
    for i, layer in enumerate(model.layers):
        if type(layer) in which_layers:
            weights = layer.get_weights()
            (rows, cols, chns, dpts) = weights[0].shape
            # FIXME: with all type of convolution
            if rows > 1:
                print('initialising with G1G2', layer.name)
                for d in range(dpts):
                    for c in range(chns):
                        g1g2 = np.random.randint(2)
                        if g1g2 == 0:
                            weights_dc = random_tog(args.tog_sigma, args.tog_surround, op=args.tog_op, width=rows)
                        elif g1g2 == 1:
                            weights_dc = random_g1(args.gg_sigma, args.gg_theta, args.gg_seta, width=rows)
                        weights[0][:, :, c, d] = weights_dc
                model.layers[i].set_weights(weights)
    return model


# FIXME: better pass of arguments, right now, it's only g1, g2 and nothing
def initialise_with_all(model, args,
                        which_layers=[keras.layers.convolutional.Conv2D, keras.layers.DepthwiseConv2D]):
    for i, layer in enumerate(model.layers):
        if type(layer) in which_layers:
            weights = layer.get_weights()
            (rows, cols, chns, dpts) = weights[0].shape
            # FIXME: with all type of convolution
            if rows > 1:
                print('initialising with G1G2', layer.name)
                for d in range(dpts):
                    for c in range(chns):
                        g1g2 = np.random.randint(3)
                        if g1g2 == 0:
                            weights_dc = random_tog(args.tog_sigma, args.tog_surround, op=args.tog_op, width=rows)
                        elif g1g2 == 1:
                            weights_dc = random_g1(args.gg_sigma, args.gg_theta, args.gg_seta, width=rows)
                        elif g1g2 == 2:
                            continue
                        weights[0][:, :, c, d] = weights_dc
                model.layers[i].set_weights(weights)
    return model