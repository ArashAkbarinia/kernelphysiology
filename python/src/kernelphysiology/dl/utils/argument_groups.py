"""
A collection of argument groups ready to be added to any other argparser.
"""


def get_segmentation_group(parser):
    segmentation_group = parser.add_argument_group('segmentation')

    segmentation_group.add_argument(
        '--save_pred',
        action='store_true',
        default=False,
        help='Saving the prediction to an image (default: False)'
    )


def get_colour_space_group(parser):
    colour_space_group = parser.add_argument_group('colour space')

    colour_space_group.add_argument(
        '--colour_space',
        type=str,
        default='rgb',
        choices=[
            'rgb',
            'lab',
            'lms'
        ],
        help='The colour space of network (default: RGB)'
    )

    # TODO: at the moment this is not used, and lab is used
    colour_space_group.add_argument(
        '--opponent_space',
        type=str,
        default='lab',
        choices=[
            'lab',
            'dkl'
        ],
        help='The default colour opponent space (default: lab)'
    )

    # TODO: Keras part is not implemented
    colour_space_group.add_argument(
        '--colour_transformation',
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
        help='The preprocessing colour transformation (default: trichromat)'
    )

    colour_space_group.add_argument(
        '--mosaic_pattern',
        type=str,
        default=None,
        choices=[
            'bayer',
            'retina'
        ],
        help='Applying a mosaic pattern to input image (default: None)'
    )


def get_architecture_group(parser):
    # TODO: better handling the parameters, e.g. pretrained ones are only for
    #  imagenet
    architecture_group = parser.add_argument_group('architecture')
    architecture_group.add_argument(
        '--num_kernels',
        type=int,
        default=64,
        help='The number of convolutional kernels (default: 64)'
    )
    architecture_group.add_argument(
        '-ca', '--custom_arch',
        dest='custom_arch',
        action='store_true',
        help='Custom architectures instead of those defined in libraries'
    )
    architecture_group.add_argument(
        '--pooling_type',
        type=str,
        default='max',
        choices=[
            'max',
            'avg',
            'mix',
            'contrast_avg',
            'contrast_max',
            'contrast'
        ],
        help='The pooling type (default: max)'
    )
    architecture_group.add_argument(
        '--pretrained',
        dest='pretrained',
        action='store_true',
        help='Use pre-trained model'
    )
    architecture_group.add_argument(
        '--blocks',
        nargs='+',
        type=int,
        default=None,
        help='Number of layers in every block (default: None)'
    )
    # TODO: this only makes sense for segmentation
    architecture_group.add_argument(
        '--backbone',
        type=str,
        default=None,
        help='The backbone of segmentation (default: None)'
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
    network_manipulation_group = parser.add_argument_group()
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
