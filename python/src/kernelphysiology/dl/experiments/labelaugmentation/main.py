# for epoch in range(args.initial_epoch, args.epochs):
#     if args.distributed:
#         train_sampler.set_epoch(epoch)
#     adjust_learning_rate(optimizer, epoch, args)
#
#     # if doing label augmentation, shuffle the labels
#     if args.augment_labels:
#         train_loader.dataset.datasets[1].shuffle_augmented_labels()


# other than doubleing labels?
# implement for CIFAr and others
# parser.add_argument(
#     '--augment_labels',
#     dest='augment_labels',
#     action='store_true',
#     help='Augmenting labels of ground-truth (False)'
# )


# if 'augment_labels' in args and args.augment_labels:
#     args.num_classes *= 2
#     args.custom_arch = True


# parser.add_argument(
#     '--neg_params',
#     nargs='+',
#     type=str,
#     default=None,
#     help='Negative sample parameters (default: None)'
# )