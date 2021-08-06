"""
A collection of argument groups ready to be added to any other argparser.
"""


def get_logging_group(parser):
    logging_group = parser.add_argument_group('logging')
    logging_group.add_argument(
        '--experiment_name',
        type=str,
        default='Ex',
        help='The name of the experiment (default: Ex)'
    )
    logging_group.add_argument(
        '--print_freq',
        type=int,
        default=100,
        help='Frequency of reporting (default: 100)'
    )
    logging_group.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Accuracy of top K elements (default: 5)'
    )
    logging_group.add_argument(
        '--random_images',
        nargs='+',
        type=int,
        default=None,
        help='Number of random images to try (default: None)'
    )
    logging_group.add_argument(
        '--save_all',
        action='store_true',
        default=False,
        help='Saving all check points (default: False)'
    )


def get_segmentation_group(parser):
    segmentation_group = parser.add_argument_group('segmentation')

    segmentation_group.add_argument(
        '--save_pred',
        action='store_true',
        default=False,
        help='Saving the prediction to an image (default: False)'
    )


def get_input_group(parser):
    input_group = parser.add_argument_group('input')

    input_group.add_argument(
        '--colour_space',
        type=str,
        default='rgb',
        choices=[
            'rgb',
            'lab',
            'lms',
            'grey', 'grey3'
        ],
        help='The colour space of network (default: RGB)'
    )

    # TODO: at the moment this is not used, and lab is used
    input_group.add_argument(
        '--opponent_space',
        type=str,
        default='lab',
        choices=[
            'lab',
            'dkl'
        ],
        help='The default colour opponent space (default: lab)'
    )

    input_group.add_argument(
        '--vision_type',
        type=str,
        default='trichromat',
        # TODO: add luminance blindness
        choices=[
            'trichromat',
            'monochromat',
            'dichromat_rg',
            'dichromat_yb',
            'protanopia',
            'deuteranopia',
            'tritanopia'
        ],
        help='The vision type of the network (default: trichromat)'
    )

    input_group.add_argument(
        '--mosaic_pattern',
        type=str,
        default=None,
        choices=[
            'bayer',
            'retina',
            'random'
        ],
        help='Applying a mosaic pattern to input image (default: None)'
    )

    input_group.add_argument(
        '--target_size',
        type=int,
        default=None,
        help='Target size (default: according to dataset)'
    )

    input_group.add_argument(
        '--sf_filter',
        default=None,
        nargs='+',
        type=float,
        help='Filtering images with <high,low> spatial freq (default: None)'
    )

    input_group.add_argument(
        '--sf_filter_chn',
        default=None,
        choices=[
            'dkl_lum',
            'dkl_rg',
            'dkl_yb'
        ],
        type=str,
        help='Filtering specific channels (default: None)'
    )


def get_output_group(parser):
    output_group = parser.add_argument_group('output')

    output_group.add_argument(
        '--num_classes',
        type=int,
        default=None,
        help='Number of classes for unknown datasets (default: None)'
    )
    output_group.add_argument(
        '--task_type',
        type=str,
        choices=[
            'classification',
            'segmentation',
            'detection'
        ],
        default=None,
        help='The task to perform by network (default: None)'
    )


def get_network_group(parser):
    # TODO: better handling the parameters, e.g. pretrained ones are only for
    #  imagenet
    network_group = parser.add_argument_group('network')

    # TODO: add choices
    # FIXME: change to architecture_name
    network_group.add_argument(
        '-aname', '--network_name',
        type=str,
        help='Name of the architecture or network'
    )

    network_group.add_argument(
        '--num_kernels',
        type=int,
        default=64,
        help='The number of convolutional kernels (default: 64)'
    )
    network_group.add_argument(
        '--kernel_size',
        type=int,
        default=7,
        help='The spatial size of kernels (default: 7)'
    )
    network_group.add_argument(
        '-ca', '--custom_arch',
        dest='custom_arch',
        action='store_true',
        help='Custom architectures instead of those defined in libraries'
    )
    network_group.add_argument(
        '--pooling_type',
        type=str,
        default='max',
        choices=[
            'max',
            'avg',
            'mix',
            'contrast_avg',
            'contrast_max',
            'contrast',
            'none'
        ],
        help='The pooling type (default: max)'
    )
    network_group.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='Use pre-trained model'
    )
    network_group.add_argument(
        '--blocks',
        nargs='+',
        type=int,
        default=None,
        help='Number of layers in every block (default: None)'
    )
    # TODO: this only makes sense for segmentation
    network_group.add_argument(
        '--backbone',
        type=str,
        default=None,
        help='The backbone of segmentation (default: None)'
    )

    # TODO: num_classes is just for backward compatibility
    network_group.add_argument(
        '--old_classes',
        default=None,
        type=int,
        help='Number of new classes (default: None)'
    )

    network_group.add_argument(
        '--transfer_weights',
        nargs='+',
        type=str,
        default=None,
        help='Whether transferring weights from a model (default: None)'
    )


def get_augmentation_group(parser):
    augmentation_group = parser.add_argument_group('augmentation')
    augmentation_group.add_argument(
        '-na', '--num_augmentations',
        type=int,
        default=None,
        help='Number of augmentations applied to each image (default: None)'
    )
    augmentation_group.add_argument(
        '-as', '--augmentation_settings',
        nargs='+',
        type=str,
        default=None,
        help='List of augmentations to be conducted (default: None)'
    )


def get_network_manipulation_group(parser):
    network_manipulation_group = parser.add_argument_group('manipulations')

    network_manipulation_group.add_argument(
        '--kill_kernels',
        nargs='+',
        type=str,
        default=None,
        help='First layer name followed by kernel indices (default: None)'
    )
    network_manipulation_group.add_argument(
        '--kill_planes',
        nargs='+',
        type=str,
        default=None,
        help='Axis number followed by plane indices ax_<P1> (default: None)'
    )
    network_manipulation_group.add_argument(
        '--kill_lines',
        nargs='+',
        type=str,
        default=None,
        help='Intersection of two planes, <P1>_<L1>_<P2>_<L2> (default: None)'
    )


def get_parallelisation_group(parser):
    parallelisation_group = parser.add_argument_group('parallelisation')

    parallelisation_group.add_argument(
        '--world-size',
        default=-1,
        type=int,
        help='Number of nodes for distributed training'
    )
    parallelisation_group.add_argument(
        '--rank',
        default=-1,
        type=int,
        help='Node rank for distributed training'
    )
    parallelisation_group.add_argument(
        '--dist-url',
        default='tcp://224.66.41.62:23456',
        type=str,
        help='URL used to set up distributed training'
    )
    parallelisation_group.add_argument(
        '--dist-backend',
        default='nccl',
        type=str,
        help='Distributed backend'
    )
    parallelisation_group.add_argument(
        '--seed',
        default=None,
        type=int,
        help='Seed for initializing training. '
    )
    parallelisation_group.add_argument(
        '--multiprocessing_distributed',
        action='store_true',
        help='Use multi-processing distributed training to launch '
             'N processes per node, which has N GPUs. This is the '
             'fastest way to use PyTorch for either single node or '
             'multi node data parallel training'
    )


def get_optimisation_group(parser):
    optimisation_group = parser.add_argument_group('optimisation')

    optimisation_group.add_argument(
        '--optimiser',
        type=str,
        default='sgd',
        help='The optimiser to be used (default: sgd)'
    )
    optimisation_group.add_argument(
        '--lr', '--learning_rate',
        type=float,
        default=None,
        help='The learning rate parameter (default: None)'
    )
    optimisation_group.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='The momentum for optimisation (default 0.9)'
    )
    optimisation_group.add_argument(
        '-wd', '--weight_decay',
        type=float,
        default=None,
        help='The decay weight parameter (default: None)'
    )
    optimisation_group.add_argument(
        '--exp_decay',
        type=float,
        default=None,
        help='The exponential decay (default: None)'
    )
    optimisation_group.add_argument(
        '--lr_schedule',
        type=str,
        default=None,
        help='The custom learning rate scheduler (default: None)'
    )
    optimisation_group.add_argument(
        '-e', '--epochs',
        type=int,
        default=90,
        help='Number of epochs (default: 90)'
    )
    optimisation_group.add_argument(
        '--initial_epoch',
        type=int,
        default=0,
        help='The initial epoch number (default: 0)'
    )
    # TODO: whether it should belong to optimisation group
    optimisation_group.add_argument(
        '--resume',
        default=None,
        type=str,
        help='Path to latest checkpoint (default: None)'
    )
    optimisation_group.add_argument(
        '--aux_loss',
        action='store_true',
        help='auxiliar loss'
    )


def get_routine_group(parser):
    routine_group = parser.add_argument_group('routine')

    routine_group.add_argument(
        '--gpus',
        nargs='+',
        type=int,
        default=[0],
        help='List of GPUs to be used (default: [0])'
    )

    # TODO: change the default according to training or testing
    routine_group.add_argument(
        '-j', '--workers',
        type=int,
        default=1,
        help='Number of workers for image generator (default: 1)'
    )

    routine_group.add_argument(
        '-b', '--batch_size',
        type=int,
        default=None,
        help='Batch size (default: according to dataset)'
    )


def get_dataset_group(parser):
    dataset_group = parser.add_argument_group('dataset')

    # FIXME: change to dataset_name
    dataset_group.add_argument(
        '-dname', '--dataset',
        type=str,
        help='Name of the dataset'
    )
    dataset_group.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='The path to the data directory (default: None)'
    )
    dataset_group.add_argument(
        '--train_dir',
        type=str,
        default=None,
        help='The path to the train directory (default: None)'
    )
    dataset_group.add_argument(
        '--validation_dir',
        type=str,
        default=None,
        help='The path to the validation directory (default: None)'
    )
    dataset_group.add_argument(
        '--categories',
        type=str,
        choices=[
            'natural_manmade'
        ],
        default=None,
        help='Choosing different categories from imagenet (default: None)'
    )
    dataset_group.add_argument(
        '--cat_dir',
        type=str,
        default=None,
        help='The directory with category files (default: None)'
    )
    dataset_group.add_argument('--train_samples', default=None, type=int)
    dataset_group.add_argument('--validation_samples', default=None, type=int)
