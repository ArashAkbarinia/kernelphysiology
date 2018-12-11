'''
Analysis generator
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np

from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from keras.utils.generic_utils import to_list

from scipy.stats import pearsonr


#FIXME: this is copied from version 2.2.0
def multiple_models_generator(model1, model2, generator,
                      steps=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      verbose=0):
    """See docstring for `Model.predict_generator`."""
#    model._make_predict_function()

    steps_done = 0
    wait_time = 0.01
    all_outs = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps is None:
        if is_sequence:
            steps = len(generator)
        else:
            raise ValueError('`steps=None` is only valid for a generator'
                             ' based on the `keras.utils.Sequence` class.'
                             ' Please specify `steps` or use the'
                             ' `keras.utils.Sequence` class.')
    enqueuer = None

    try:
        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if is_sequence:
                output_generator = iter(generator)
            else:
                output_generator = generator

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            if isinstance(generator_output, tuple):
                # Compatibility with the generators
                # used for training.
                if len(generator_output) == 2:
                    x, _ = generator_output
                elif len(generator_output) == 3:
                    x, _, _ = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
            else:
                # Assumes a generator that only
                # yields inputs (not targets and sample weights).
                x = generator_output

            outs1 = model1.predict_on_batch(x)
            outs2 = model2.predict_on_batch(x)

            nimages = outs1.shape[0]
#            outs1 = np.reshape(outs1, (nimages, -1))
#            outs2 = np.reshape(outs2, (nimages, -1))

            kernels = [27, 129, 138, 155, 195, 260, 301, 368, 406, 462, 482, 511]
            outs = np.zeros((nimages, len(kernels)))
            for i in range(nimages):
                for k_ind, k in enumerate(kernels):
#                    (outs[i, k_ind], _) = pearsonr(outs1[i, :, :, k].flatten(), outs2[i, :, :, k].flatten())
                    outs[i, k_ind] = np.mean(outs1[i, :, :, k].flatten() - outs2[i, :, :, k].flatten())
#            import pdb
#            pdb.set_trace()

            if not isinstance(outs, list):
                outs = [outs]

            if not all_outs:
                for out in outs:
                    all_outs.append([])

            for i, out in enumerate(outs):
                all_outs[i].append(out)
            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    if len(all_outs) == 1:
        if steps_done == 1:
            return all_outs[0][0]
        else:
            return np.concatenate(all_outs[0])
    if steps_done == 1:
        return [out[0] for out in all_outs]
    else:
        return [np.concatenate(out) for out in all_outs]


# TODO: move me to correct place
def top_k_accuracy(y_true, y_pred, k=1):
    '''
    From: https://github.com/chainer/chainer/issues/606
    Expects both y_true and y_pred to be one-hot encoded.
    '''
    argsorted_y = np.argsort(y_pred)[:,-k:]
    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0)


def predict_generator(model, generator,
                      steps=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      verbose=0):
    """See docstring for `Model.predict_generator`."""
#    model._make_predict_function()

    steps_done = 0
    wait_time = 0.01
    all_outs = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps is None:
        if is_sequence:
            steps = len(generator)
        else:
            raise ValueError('`steps=None` is only valid for a generator'
                             ' based on the `keras.utils.Sequence` class.'
                             ' Please specify `steps` or use the'
                             ' `keras.utils.Sequence` class.')
    enqueuer = None

    try:
        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if is_sequence:
                output_generator = iter(generator)
            else:
                output_generator = generator

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            if isinstance(generator_output, tuple):
                # Compatibility with the generators
                # used for training.
                if len(generator_output) == 2:
                    x, y = generator_output
                elif len(generator_output) == 3:
                    x, y, _ = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
            else:
                # Assumes a generator that only
                # yields inputs (not targets and sample weights).
                x = generator_output

            y_pred = model.predict_on_batch(x)
            # TODO: for now only top 1 and top 5
            outs = np.zeros((y_pred.shape[0], 2))
            outs[:, 0] = top_k_accuracy(y, y_pred)
            outs[:, 1] = top_k_accuracy(y, y_pred, 5)

            if not isinstance(outs, list):
                outs = [outs]

            if not all_outs:
                for out in outs:
                    all_outs.append([])

            for i, out in enumerate(outs):
                all_outs[i].append(out)
            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    if len(all_outs) == 1:
        if steps_done == 1:
            return all_outs[0][0]
        else:
            return np.concatenate(all_outs[0])
    if steps_done == 1:
        return [out[0] for out in all_outs]
    else:
        return [np.concatenate(out) for out in all_outs]


def analysis_generator(model, generator,
                      steps=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      verbose=0):
    """See docstring for `Model.predict_generator`."""
    model._make_predict_function()

    steps_done = 0
    wait_time = 0.01
    all_outs = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps is None:
        if is_sequence:
            steps = len(generator)
        else:
            raise ValueError('`steps=None` is only valid for a generator'
                             ' based on the `keras.utils.Sequence` class.'
                             ' Please specify `steps` or use the'
                             ' `keras.utils.Sequence` class.')
    enqueuer = None

    try:
        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if is_sequence:
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            if isinstance(generator_output, tuple):
                # Compatibility with the generators
                # used for training.
                if len(generator_output) == 2:
                    x, _ = generator_output
                elif len(generator_output) == 3:
                    x, _, _ = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
            else:
                # Assumes a generator that only
                # yields inputs (not targets and sample weights).
                x = generator_output

            outs = model.predict_on_batch(x)
            outs = to_list(outs)

            if not all_outs:
                for out in outs:
                    all_outs.append([])

            for i, out in enumerate(outs):
                all_outs[i].append(out)
            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    if len(all_outs) == 1:
        if steps_done == 1:
            return all_outs[0][0]
        else:
            return np.concatenate(all_outs[0])
    if steps_done == 1:
        return [out[0] for out in all_outs]
    else:
        return [np.concatenate(out) for out in all_outs]
