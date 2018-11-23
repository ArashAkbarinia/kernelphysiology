# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
import pickle
import datetime
import pandas as pd
import keras

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.externals import joblib
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image


use_cache = 1
# color type: 1 - grey, 3 - rgb
color_type_global = 3


def get_driver_data():
    dr = dict()
    path = os.path.join('input', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr


def load_train(img_rows, img_cols, color_type=1):
    x_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('input', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = image.load_img(fl, grayscale=color_type==1, target_size=(img_rows, img_cols))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            features = model.predict(img)
            
            x_train.append(features)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return x_train, y_train, driver_id, unique_drivers


def load_test(img_rows, img_cols, color_type=1, start=0, end=-1):
    print('Read test images')
    path = os.path.join('input', 'test', '*.jpg')
    files = glob.glob(path)
    nfiles = len(files)
    if end == -1:
        end = nfiles
    
    x_test = []
    x_test_id = []
    
    # initialising the batch size
    imgs_inds = 0
    left_batch_size = min(batch_size, nfiles - start - 1)
    imgs = np.zeros((left_batch_size, img_rows, img_cols, color_type))
    for j in range(start, end):
        flbase = os.path.basename(files[j])
        img = image.load_img(files[j], grayscale=color_type==1, target_size=(img_rows, img_cols))
        img = image.img_to_array(img)
        
        imgs[imgs_inds, :, :, :] = img
        
        if imgs_inds == left_batch_size - 1:
            # adapting the image to the network preprocessing
            img = preprocess_input(imgs)
            # extracting the features from the chosen layer
            features = model.predict(imgs)
            x_test.extend(features)
            # setting the batch size back to zero
            imgs_inds = 0
            left_batch_size = min(batch_size, end - j - 1)
            imgs = np.zeros((left_batch_size, img_rows, img_cols, color_type))
        else:
            imgs_inds += 1
        
        x_test_id.extend({flbase})
    return x_test, x_test_id


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists %s ' % path)


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def create_submission(predictions, test_id, info):
    result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result.to_csv(sub_file, index=False)


def read_and_normalize_train_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)

    train_data = np.array(train_data)
    train_data = np.transpose(train_data, (0, 2, 1))
    train_data = np.squeeze(train_data, axis=2)
    train_target = np.array(train_target)
    
    print('Train shape:', train_data.shape)
    return train_data, train_target, driver_id, unique_drivers


def read_and_normalize_test_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data)
    test_data = np.transpose(test_data, (0, 2, 1))
    test_data = np.squeeze(test_data, axis=2)

    print('Test shape:', test_data.shape)
    return test_data, test_id


def test_on_chunks(clf, img_rows, img_cols, color_type=1, chunk_size=8000):
    test_name = 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type)
 
    print('Tesing on different chunks')
    path = os.path.join('input', 'test', '*.jpg')
    files = glob.glob(path)
    nfiles = len(files)
    nclasses = 10
    test_res = np.zeros((nfiles, nclasses))
    test_id = []
    
    # going through the test images in chunks
    for start in range(0, nfiles, chunk_size):
        end = min(nfiles, start + chunk_size)
        print('Reading chunk: ' + str(start) + '-' + str(end))
        
        cache_path = os.path.join('cache', '%s_%.5d_%.5d.dat' % (test_name, start + 1, end))
        if not os.path.isfile(cache_path) or use_cache == 0:
            test_data, current_test_id = load_test(img_rows, img_cols, color_type, start, end)
            test_data = np.array(test_data)
            cache_data((test_data, current_test_id), cache_path)
        else:
            print('Restore test from cache!')
            (test_data, current_test_id) = restore_data(cache_path)
        
        print('Test shape:', test_data.shape)
        current_res = clf.predict_proba(test_data)
        test_res[start:end, :] = current_res
        test_id.extend(current_test_id)

    return test_res, test_id


def run_single(debug=True):
    # input image dimensions
    img_rows, img_cols = 224, 224

    train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global)

    # using the svm as the classifier
    clf = svm.SVC(kernel='linear', C=1, probability=True, max_iter=100, tol=1e-3, verbose=True)
    # for testing purposes, try it with cross validation
    if debug:
        print('Cross validating')
        cv = ShuffleSplit(n_splits=3, test_size=0.4, random_state=0)
        scores = cross_val_score(clf, train_data, train_target, cv=cv)
        print("Accuracy: %0.2f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    # this would be the final submission
    else:
        print('Testing')
        ModelPath = 'FinalModel.pkl'
        if not os.path.isfile(ModelPath):
            clf.fit(train_data, train_target)
            joblib.dump(clf, ModelPath)
        else:
            clf = joblib.load(ModelPath)
        test_res, test_id = test_on_chunks(clf, img_rows, img_cols, color_type_global, chunk_size=8000)
        info_string = 'vgg16_r_' + str(img_rows) + '_c_' + str(img_cols)
        create_submission(test_res, test_id, info_string)


batch_size = 32
# which model to use as feature extractor
base_model = VGG16(weights='imagenet')
# which layer to use as feature extractor
layer_name = 'flatten'
model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
run_single(int(os.sys.argv[1]))
