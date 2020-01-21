"""
Handling all shadow related datasetes.
"""

import glob

from PIL import Image

from torchvision.datasets.vision import VisionDataset

IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp'
]


def _read_extension(root, extension):
    img_paths = []
    img_paths.extend(
        sorted(glob.glob(root + '/*' + extension))
    )
    # with upper case
    img_paths.extend(
        sorted(glob.glob(root + '/*' + extension.upper()))
    )
    return img_paths


def _read_paths(root, extensions=None):
    if extensions is None:
        extensions = IMG_EXTENSIONS

    img_paths = []
    target_paths = []
    # reading all extensions
    for extension in extensions:
        img_paths.extend(_read_extension(root + '/imgs/', extension))
        target_paths.extend(_read_extension(root + '/masks/', extension))

    return img_paths, target_paths


class ShadowDetection(VisionDataset):
    """`Shadow Detection Datasets.

    Args:
        root (string): Root directory where images are downloaded to.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version.
            E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input
            sample and its target as entry and returns a transformed version.
    """

    def __init__(self, root, transform=None, target_transform=None,
                 transforms=None):
        super(ShadowDetection, self).__init__(
            root, transforms, transform, target_transform
        )
        self.img_paths, self.target_paths = _read_paths(root)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target).
        """
        img_path = self.img_paths[index]
        target_path = self.target_paths[index]

        img = Image.open(img_path).convert('RGB')
        target = Image.open(target_path).convert('F')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)
