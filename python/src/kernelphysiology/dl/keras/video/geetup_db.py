"""
Reading the GEETUP dataset and creating train and validation sets.
"""

import numpy as np
import glob
import logging

import keras
import keras.backend as K
from keras.preprocessing import image

from kernelphysiology.filterfactory import gaussian


def last_valid_frame(video_dir, frames_gap=10, sequence_length=9):
    """
    Given the length of sequence and gap between frames, it computes which is
    the last valid frame to read from a directory.
    """
    num_frames = len(glob.glob((video_dir + '/*.jpg')))
    last_frame = num_frames - (frames_gap * sequence_length)
    return last_frame


def subject_frame_limits(subject_dir, frames_gap=10, sequence_length=9):
    subject_data = []
    for segment in glob.glob(subject_dir + '/segments/*/'):
        for video_dir in glob.glob(segment + '/CutVid_*/'):
            current_num_frames = len(glob.glob((video_dir + '/*.jpg')))
            video_ind = video_dir.split('/')[-2].split('_')[-1]
            fixation_array = np.loadtxt(
                segment + '/SUBSAMP_EYETR_' + video_ind + '.txt')
            if fixation_array.shape[0] != current_num_frames:
                logging.info('%s contains %d frames but %d fixation points' %
                             (video_dir,
                              current_num_frames,
                              fixation_array.shape[0]))
                continue
            current_video_limit = last_valid_frame(video_dir, frames_gap,
                                                   sequence_length)
            if current_video_limit > 0:
                subject_data.append([segment, video_ind, current_video_limit])
    return subject_data


def dataset_frame_list(dataset_dir, frames_gap=10, sequence_length=9):
    dataset_data = []
    for subject_dir in glob.glob(dataset_dir + '/*/'):
        subject_data = subject_frame_limits(subject_dir, frames_gap=frames_gap,
                                            sequence_length=sequence_length)
        dataset_data.extend(subject_data)
    return dataset_data


def heat_map_from_fixation(fixation_point, target_size, org_size,
                           gaussian_kernel=None, gaussian_sigma=1.5):
    if gaussian_kernel is None:
        gaussian_kernel = gaussian.gaussian_kernel2(gaussian_sigma)
    rows = target_size[1]
    cols = target_size[0]
    org_rows = org_size[1]
    org_cols = org_size[0]
    # print(rows, cols, org_rows, org_cols)
    fixation_map = np.zeros((rows, cols, 1))
    if fixation_point[0] > 0 and fixation_point[1] > 0:
        fpr = int(fixation_point[1] * (rows / org_rows))
        fpc = int(fixation_point[0] * (cols / org_cols))

        sr = fpr - (gaussian_kernel.shape[0] // 2)
        sc = fpc - (gaussian_kernel.shape[1] // 2)
        # making sure they're within the range of image
        gsr = np.maximum(0, -sr)
        gsc = np.maximum(0, -sc)

        er = sr + gaussian_kernel.shape[0]
        ec = sc + gaussian_kernel.shape[1]
        # making sure they're within the range of image
        sr = np.maximum(0, sr)
        sc = np.maximum(0, sc)

        er_diff = er - rows
        ec_diff = ec - cols
        ger = np.minimum(gaussian_kernel.shape[0],
                         gaussian_kernel.shape[0] - er_diff)
        gec = np.minimum(gaussian_kernel.shape[1],
                         gaussian_kernel.shape[1] - ec_diff)

        er = np.minimum(er, rows)
        ec = np.minimum(ec, cols)
        # print(fpr, fpc, gaussian_kernel[gsr:ger, gsc:gec].shape)
        g_max = gaussian_kernel[gsr:ger, gsc:gec].max()
        fixation_map[sr:er, sc:ec, 0] = \
            gaussian_kernel[gsr:ger, gsc:gec] / g_max
    return fixation_map


class GeetupGenerator(keras.utils.Sequence):
    """GEETUP generator for training and validation."""

    def __init__(self, video_list, batch_size=32, target_size=(224, 224),
                 num_chns=3, frames_gap=10, sequence_length=9,
                 gaussian_sigma=2.5, preprocessing_function=None, shuffle=True):
        """Initialisation"""
        self.video_list = video_list
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_chns = num_chns
        self.frames_gap = frames_gap
        self.sequence_length = sequence_length
        self.preprocessing_function = preprocessing_function
        self.shuffle = shuffle
        self.grey_scale = self.num_chns == 1
        self.gaussian_kernel = gaussian.gaussian_kernel2(gaussian_sigma)

        if K.image_data_format() == 'channels_last':
            self.in_shape = (self.sequence_length,
                             *self.target_size,
                             self.num_chns)
            self.out_shape = (self.sequence_length,
                              *self.target_size,
                              1)
        elif K.image_data_format() == 'channels_first':
            self.in_shape = (self.sequence_length,
                             self.num_chns,
                             *self.target_size)
            self.out_shape = (self.sequence_length,
                              1,
                              *self.target_size)

        self.num_sequences = 0
        for f in video_list:
            self.num_sequences += f[2]
        self.check_list = self.initialise_sequence_check_list()
        self.on_epoch_end()

    def initialise_sequence_check_list(self):
        check_list = []
        i = 0
        for f, video_info in enumerate(self.video_list):
            for j in range(video_info[2]):
                check_list.append([f, video_info[1], j])
                i += 1
        return check_list

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.num_sequences / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        current_batch = self.check_list[
                        index * self.batch_size:(index + 1) * self.batch_size]

        # generate data
        (x_batch, y_batch) = self.__data_generation(current_batch)

        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indices after each epoch"""
        if self.shuffle is True:
            np.random.shuffle(self.check_list)

    def __data_generation(self, current_batch):
        """Generates data containing batch_size samples"""
        # initialisation
        current_num_imgs = len(current_batch)
        x_batch = np.empty((current_num_imgs, *self.in_shape), dtype='float32')
        # TODO: could be that only Euclidean distance might be better?
        y_batch = np.empty((current_num_imgs, *self.out_shape), dtype='float32')

        # generate data
        for i, video_id in enumerate(current_batch):
            # get the video and frame number
            segment_dir = self.video_list[video_id[0]][0]
            video_num = self.video_list[video_id[0]][1]
            video_path = segment_dir + '/CutVid_' + video_num + '/frames'
            fixation_path = segment_dir + 'SUBSAMP_EYETR_' + video_num + '.txt'
            frame_num = video_id[2]
            fixation_points = np.loadtxt(fixation_path)
            start_frame = frame_num + 1
            end_frame = start_frame + self.sequence_length * self.frames_gap
            for j, c_f_num in enumerate(
                    range(start_frame, end_frame, self.frames_gap)):
                current_img = image.load_img(
                    video_path + str(c_f_num) + '.jpg',
                    grayscale=self.grey_scale)
                org_size = current_img.size
                current_img = current_img.resize(self.target_size)
                current_img = image.img_to_array(current_img)
                x_batch[i, j,] = current_img
                y_batch[i, j,] = heat_map_from_fixation(
                    fixation_points[j],
                    target_size=self.target_size,
                    org_size=org_size,
                    gaussian_kernel=self.gaussian_kernel)

        if self.preprocessing_function is not None:
            x_batch = self.preprocessing_function(x_batch)

        rows = self.target_size[1]
        cols = self.target_size[0]
        y_batch = np.reshape(y_batch,
                             (-1, self.sequence_length, rows * cols, 1))
        return x_batch, y_batch
