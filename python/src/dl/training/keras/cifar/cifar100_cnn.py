'''Train a simple deep CNN on the CIFAR100 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
import os
import sys
import cifar
import cifar100
from keras.callbacks import CSVLogger, ModelCheckpoint

project_root = '/home/arash/Software/repositories/kernelphysiology/python/'

batch_size = 32
num_classes = 100
epochs = 100
log_period = round(epochs / 4)
data_augmentation = False

model_name = 'keras_cifar100_area_'
save_dir = os.path.join(project_root, 'data/nets/cifar/cifar100/')

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data('fine', os.path.join(project_root, 'data/datasets/cifar/cifar100/'))
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

nlayers = sys.argv[1]
print('Processing with %s layers' % nlayers)
model_name += nlayers
log_dir = os.path.join(save_dir, model_name)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
csv_logger = CSVLogger(os.path.join(log_dir, 'log.csv'), append=False, separator=';')
check_points = ModelCheckpoint(os.path.join(log_dir, 'weights.{epoch:05d}.h5'), period=log_period)

nlayers = int(nlayers)

model = cifar.generate_model(train_shape=x_train.shape[1:], num_classes=num_classes, area1_nlayers=nlayers)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

cifar.train_model(x_train, y_train, x_test, y_test, model, 
                  callbacks=[check_points, csv_logger], save_dir=save_dir, model_name=model_name, 
                  data_augmentation=False, batch_size=32, epochs=10)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
