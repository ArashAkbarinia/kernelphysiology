'''
Utility functoins for models.
'''


import keras

from keras import applications as kmodels

from kernelphysiology.dl.keras.models import resnet50, visual_netex
from kernelphysiology.dl.keras.models import inception_v3
from kernelphysiology.dl.keras.models import vgg16, vgg19
from kernelphysiology.dl.keras.models import densenet
from kernelphysiology.dl.keras.models import mrcnn

from kernelphysiology.dl.keras.datasets.utils import get_default_target_size, get_default_num_classes

from kernelphysiology.dl.keras.utils import get_input_shape


def export_weights_to_model(weights_path, model_path, architecture, dataset, area1layers=None):
    # TODO: instead of passings args, pass them separately
    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]
    args = {}
    target_size = get_default_target_size(dataset)
    args['input_shape'] = get_input_shape(target_size)
    args['network_name'] = architecture
    args['num_classes'] = get_default_num_classes(dataset)
    args['area1layers'] = area1layers
    args = dotdict(args)
    model = which_architecture(args)
    model.load_weights(weights_path)
    model.save(model_path)


def which_network_classification(args, network_name):
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


def which_network_detection(args, network_name):
    log_dir = '/home/arash/Software/repositories/'
    model = mrcnn.MaskRCNN(mode='inference', config=args.config, model_dir=log_dir)
    if network_name == 'mrcnn':
        #FIXME the path
        model_path = '/home/arash/Software/repositories/kernelphysiology/python/data/nets/coco/mask_rcnn_coco.h5'
        model.load_weights(model_path, by_name=True)
    elif network_name == 'retinanet':
        import sys
        sys.path += ['/home/arash/Software/repositories/keras-retinanet/']
        from keras_retinanet.models import load_model as retina_load_model
        model_path = '/home/arash/Software/repositories/kernelphysiology/python/data/nets/coco/retinanet_resnet50_coco_v2.h5'
        model.keras_model = retina_load_model(model_path)
    args.model = model
    return args


def which_network(args, network_name, task_type):
    # FIXME: network should be acosiated to dataset
    if task_type == 'classification':
        args = which_network_classification(args, network_name)
    elif task_type == 'detection':
        args = which_network_detection(args, network_name)
    return args


def which_architecture(args):
    # TODO: add other architectures of keras
    network_name = args.network_name
    if network_name == 'visual_netex':
        model = visual_netex.VisualNetex(input_shape=args.input_shape, classes=args.num_classes)
    elif network_name == 'resnet20':
        # FIXME: make it more generic
        # FIXME: add initialiser to ass architectures args.initialise
        from kernelphysiology.dl.keras.models import resnet
        model = resnet.resnet_v1(input_shape=args.input_shape, depth=3*6+2, kernel_initializer=None, num_classes=args.num_classes)
    if network_name == 'resnet50':
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
    # by default no preprocessing function
    preprocessing_function = None
    # switch case of preprocessing functions
    if preprocessing == 'visual_netex':
        preprocessing_function = visual_netex.preprocess_input
    elif 'resnet' in preprocessing or 'retinanet' in preprocessing:
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
    elif 'densenet' in preprocessing:
        preprocessing_function = kmodels.densenet.preprocess_input
    elif preprocessing == 'mobilenet':
        preprocessing_function = kmodels.mobilenet.preprocess_input
    elif preprocessing == 'mobilenet_v2':
        # FIXME: compatibility with version 2.2.0
        preprocessing_function = kmodels.mobilenetv2.preprocess_input
    elif 'nasnet' in preprocessing:
        preprocessing_function = kmodels.nasnet.preprocess_input
    elif preprocessing == 'mrcnn':
        preprocessing_function = None
    return preprocessing_function