from torchvision import datasets as tdatasets


class ImageFolder(tdatasets.ImageFolder):
    def __init__(self, img_manipulation=None, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.imgs = self.samples
        self.img_manipulation = img_manipulation

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, img_target) where img_target is the same size as
             original image after applied manipulations.
        """
        path, class_target = self.samples[index]
        sample = self.loader(path)
        img_target = sample.copy()
        if self.img_manipulation is not None:
            img_target = self.transform(img_target)

        if self.transform is not None:
            sample, img_target = self.transform([sample, img_target])

        # right now we're not using the class target, but perhaps in the future
        if self.target_transform is not None:
            class_target = self.target_transform(class_target)

        return sample, img_target
