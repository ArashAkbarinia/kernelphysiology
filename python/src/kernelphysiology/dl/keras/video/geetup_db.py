"""
Reading the GEETUP dataset and creating train and validation sets.
"""

import numpy as np

import keras
import keras.backend as K
from keras.preprocessing import image

from kernelphysiology.dl.geetup.geetup_utils import map_point_to_image_size
from kernelphysiology.utils.imutils import heat_map_from_point
from kernelphysiology.filterfactory.gaussian import gaussian_kernel2


class GeetupGenerator(keras.utils.Sequence):
    """GEETUP generator for training and validation."""

    def __init__(self, video_list, batch_size=32, target_size=(224, 224),
                 num_chns=3, frames_gap=10, sequence_length=9, all_frames=False,
                 gaussian_sigma=2.5, preprocessing_function=None, shuffle=True,
                 org_size=(360, 640), only_name_and_gt=False):
        """Initialisation"""
        self.video_list = video_list
        self.batch_size = batch_size
        self.target_size = target_size
        self.org_size = org_size
        self.num_chns = num_chns
        self.frames_gap = frames_gap
        self.sequence_length = sequence_length
        self.preprocessing_function = preprocessing_function
        self.shuffle = shuffle
        self.only_name_and_gt = only_name_and_gt
        self.grey_scale = self.num_chns == 1
        self.gaussian_kernel = gaussian_kernel2(gaussian_sigma)
        self.only_last_frame = not all_frames

        if K.image_data_format() == 'channels_last':
            self.in_shape = (
                self.sequence_length,
                *self.target_size,
                self.num_chns
            )
            self.out_shape = (
                self.sequence_length,
                *self.target_size,
                1
            )
        elif K.image_data_format() == 'channels_first':
            self.in_shape = (
                self.sequence_length,
                self.num_chns,
                *self.target_size
            )
            self.out_shape = (
                self.sequence_length,
                1,
                *self.target_size
            )

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
        if self.only_name_and_gt:
            x_batch = np.empty(
                (current_num_imgs, self.sequence_length), dtype='<U180'
            )
            y_batch = np.empty(
                (current_num_imgs, self.sequence_length, 2), dtype='float32'
            )
        else:
            x_batch = np.empty(
                (current_num_imgs, *self.in_shape), dtype='float32'
            )
            y_batch = np.empty(
                (current_num_imgs, *self.out_shape), dtype='float32'
            )

        # generate data
        for i, video_id in enumerate(current_batch):
            # get the video and frame number
            segment_dir = self.video_list[video_id[0]][0]
            video_num = self.video_list[video_id[0]][1]
            selected_path = segment_dir + '/CutVid_' + video_num + \
                            '/selected_frames/'
            video_path = selected_path + 'frames'
            fixation_path = selected_path + 'gt.txt'
            frame_num = video_id[2]
            fixation_points = np.loadtxt(fixation_path)
            start_frame = frame_num + 1
            end_frame = start_frame + self.sequence_length * self.frames_gap
            for j, c_f_num in enumerate(
                    range(start_frame, end_frame, self.frames_gap)):
                image_name = video_path + str(c_f_num) + '.jpg'
                # if org_size is None, we assume different images are different
                if self.org_size is None or self.only_name_and_gt is False:
                    current_img = image.load_img(image_name,
                                                 grayscale=self.grey_scale)
                    # [::-1] because PIL images have size ay XY, not rows cols
                    org_size = current_img.size[::-1]
                else:
                    org_size = self.org_size

                gt_resized = map_point_to_image_size(
                    fixation_points[c_f_num - 1][::-1],
                    self.target_size,
                    org_size
                )

                if self.only_name_and_gt:
                    x_batch[i, j,] = image_name
                    y_batch[i, j,] = gt_resized
                else:
                    # [::-1] because PIL images have size ay XY, not rows cols
                    current_img = current_img.resize(self.target_size[::-1])
                    current_img = image.img_to_array(current_img)
                    x_batch[i, j,] = current_img
                    y_batch[i, j,] = heat_map_from_point(
                        gt_resized,
                        target_size=self.target_size,
                        g_kernel=self.gaussian_kernel
                    )

        if self.only_name_and_gt is False:
            if self.preprocessing_function is not None:
                x_batch = self.preprocessing_function(x_batch)

            rows = self.target_size[1]
            cols = self.target_size[0]
            # TODO: in case of only last frame, don't read all other GTs
            if self.only_last_frame:
                y_batch = np.reshape(y_batch[:, -1, ], (-1, 1, rows * cols, 1))
            else:
                y_batch = np.reshape(
                    y_batch, (-1, self.sequence_length, rows * cols, 1)
                )
        else:
            if self.only_last_frame:
                x_batch = np.reshape(x_batch[:, -1, ], (-1, 1, 1))
                y_batch = np.reshape(y_batch[:, -1, ], (-1, 1, 2))
        return x_batch, y_batch
