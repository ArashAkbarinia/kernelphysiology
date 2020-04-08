from torchvision import datasets as tdatasets


class ImageFolder(tdatasets.ImageFolder):
    def __init__(self, intransform=None, outtransform=None, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.imgs = self.samples
        self.intransform = intransform
        self.outtransform = outtransform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (imgin, imgout) where imgout is the same size as
             original image after applied manipulations.
        """
        path, class_target = self.samples[index]
        imgin = self.loader(path)
        imgout = imgin.copy()
        if self.intransform is not None:
            imgin = self.intransform(imgin)
        if self.outtransform is not None:
            imgout = self.outtransform(imgout)

        if self.transform is not None:
            imgin, imgout = self.transform([imgin, imgout])

        # right now we're not using the class target, but perhaps in the future
        if self.target_transform is not None:
            class_target = self.target_transform(class_target)

        return imgin, imgout
