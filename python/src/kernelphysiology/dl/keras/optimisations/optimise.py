'''
Helper functions to set optimisation of a network.
'''


import keras
import numpy as np


def get_default_decays(optimiser_name):
    # TODO: add more optimisers and parametrise from argument line
    if optimiser_name == 'adam':
        decay = 0
    elif optimiser_name == 'sgd':
        decay = 0
    elif optimiser_name == 'rmsprop':
        decay = 0
    elif optimiser_name == 'adagrad':
        decay = 0
    return decay


def get_default_lrs(optimiser_name):
    # TODO: add more optimisers and parametrise from argument line
    if optimiser_name == 'adam':
        lr = 1e-3
    elif optimiser_name == 'sgd':
        lr = 1e-1
    elif optimiser_name == 'rmsprop':
        lr = 0.045
    elif optimiser_name == 'adagrad':
        lr = 1e-2
    return lr


def set_optimisation(args):
    # TODO: add more optimisers and parametrise from argument line
    lr = args.lr
    decay = args.decay
    if args.optimiser.lower() == 'adam':
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = None
        amsgrad = False
        opt = keras.optimizers.Adam(lr=lr, decay=decay, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
    elif args.optimiser.lower() == 'sgd':
        momentum = 0.9
        nesterov = False
        opt = keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
    elif args.optimiser.lower() == 'rmsprop':
        rho = 0.9
        epsilon = 1.0
        opt = keras.optimizers.RMSprop(lr=lr, decay=decay, rho=rho, epsilon=epsilon)
    elif args.optimiser.lower() == 'adagrad':
        opt = keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=decay)
    return opt


def exp_decay(epoch, lr, exp_decay):
   new_lr = lr * np.exp(-exp_decay * epoch)
   return new_lr


def lr_schedule(epoch, lr):
    '''Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    '''
    if epoch < 81:
        # FXIME: better handling ot this
        new_lr = lr
        return new_lr
    new_lr = lr
    if epoch > 180:
        new_lr *= 0.5e-3
    elif epoch > 160:
        new_lr *= 1e-3
    elif epoch > 120:
        new_lr *= 1e-2
    elif epoch > 80:
        new_lr *= 1e-1
    return new_lr