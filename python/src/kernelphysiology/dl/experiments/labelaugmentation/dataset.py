# if augment_labels:
#     target_transform = lambda x: x + get_num_classes(dataset_name)
#     augmented_dataset = get_augmented_dataset(
#         dataset_name, traindir, colour_transformations,
#         other_transformations, chns_transformation,
#         normalize, target_size, train_dataset, target_transform
#     )
#
#     train_dataset = ConcatDataset([train_dataset, augmented_dataset])


# def get_augmented_dataset(dataset_name, traindir, colour_transformations,
#                           other_transformations, chns_transformation,
#                           normalize, target_size, original_train,
#                           target_transform):
#     # for label augmentation, we don't want to perform crazy cropping
#     augmented_transformations = prepare_transformations_test(
#         dataset_name, colour_transformations,
#         other_transformations, chns_transformation,
#         normalize, target_size
#     )
#     if dataset_name == 'imagenet':
#         augmented_dataset = label_augmentation.RandomNegativeLabelFolder(
#             traindir, augmented_transformations, target_transform
#         )
#     elif dataset_name == 'cifar10':
#         augmented_dataset = label_augmentation.RandomNegativeLabelArray(
#             original_train.data, original_train.targets,
#             augmented_transformations, target_transform
#         )
#     elif dataset_name == 'cifar100':
#         augmented_dataset = label_augmentation.RandomNegativeLabelArray(
#             original_train.data, original_train.targets,
#             augmented_transformations, target_transform
#         )
#     else:
#         sys.exit('Augmented dataset %s is not supported.' % dataset_name)
#     return augmented_dataset
