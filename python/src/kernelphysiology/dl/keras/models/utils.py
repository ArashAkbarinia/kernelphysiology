'''
Utility functoins for models.
'''


import keras

from keras import applications as kmodels

from kernelphysiology.dl.keras.models import resnet50
from kernelphysiology.dl.keras.models import inception_v3
from kernelphysiology.dl.keras.models import vgg16, vgg19
from kernelphysiology.dl.keras.models import densenet

from kernelphysiology.dl.keras.utils import get_input_shape


def which_network(args, network_name):
    # if passed by name we assume the original architectures
    # TODO: make the arguments nicer so in this case no preprocessing can be passed
    # TODO: very ugly work around for target size and input shape
    if network_name == 'resnet50':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.resnet50.ResNet50(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'inception_v3':
        args.target_size = (299, 299)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.inception_v3.InceptionV3(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'inception_resnet_v2':
        args.target_size = (299, 299)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.inception_resnet_v2.InceptionResNetV2(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'xception':
        args.target_size = (299, 299)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.xception.Xception(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'vgg16':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.vgg16.VGG16(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'vgg19':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.vgg19.VGG19(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'densenet121':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.densenet.DenseNet121(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'densenet169':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.densenet.DenseNet169(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'densenet201':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.densenet.DenseNet201(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'mobilenet':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.mobilenet.MobileNet(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'mobilenet_v2':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        # FIXME: compatibility with version 2.2.0
        args.model = kmodels.mobilenetv2.MobileNetV2(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'nasnetmobile':
        args.target_size = (224, 224)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.nasnet.NASNetMobile(input_shape=args.input_shape, weights='imagenet')
    elif network_name == 'nasnetlarge':
        args.target_size = (331, 331)
        args.input_shape = get_input_shape(args.target_size)
        args.model = kmodels.nasnet.NASNetLarge(input_shape=args.input_shape, weights='imagenet')
    else:
        args.model = keras.models.load_model(network_name, compile=False)
    return args


def which_architecture(args):
    # TODO: add other architectures of keras
    network_name = args.network_name
    if network_name == 'resnet50':
        if args.dataset == 'cifar10':
            # FIXME: make it more generic
            from kernelphysiology.dl.keras.models import resnet
            model = resnet.resnet_v1(input_shape=args.input_shape, depth=3*6+2)
        else:
            model = resnet50.ResNet50(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'inception_v3':
        model = inception_v3.InceptionV3(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'vgg16':
        model = vgg16.VGG16(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'vgg19':
        model = vgg19.VGG19(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'densenet121':
        model = densenet.DenseNet121(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'densenet169':
        model = densenet.DenseNet169(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    elif network_name == 'densenet201':
        model = densenet.DenseNet201(input_shape=args.input_shape, classes=args.num_classes, area1layers=args.area1layers)
    return model


def get_preprocessing_function(preprocessing):
    # switch case of preprocessing functions
    if preprocessing == 'resnet50':
        preprocessing_function = kmodels.resnet50.preprocess_input
    elif preprocessing == 'inception_v3':
        preprocessing_function = kmodels.inception_v3.preprocess_input
    elif preprocessing == 'inception_resnet_v2':
        preprocessing_function = kmodels.inception_resnet_v2.preprocess_input
    elif preprocessing == 'xception':
        preprocessing_function = kmodels.xception.preprocess_input
    elif preprocessing == 'vgg16':
        preprocessing_function = kmodels.vgg16.preprocess_input
    elif preprocessing == 'vgg19':
        preprocessing_function = kmodels.vgg19.preprocess_input
    elif preprocessing == 'densenet121' or preprocessing == 'densenet169' or preprocessing == 'densenet201':
        preprocessing_function = kmodels.densenet.preprocess_input
    elif preprocessing == 'mobilenet':
        preprocessing_function = kmodels.mobilenet.preprocess_input
    elif preprocessing == 'mobilenet_v2':
        # FIXME: compatibility with version 2.2.0
        preprocessing_function = kmodels.mobilenetv2.preprocess_input
    elif preprocessing == 'nasnetmobile' or preprocessing == 'nasnetlarge':
        preprocessing_function = kmodels.nasnet.preprocess_input
    return preprocessing_function