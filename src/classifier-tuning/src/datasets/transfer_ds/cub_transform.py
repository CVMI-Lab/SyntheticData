import random
import math
import numbers

import cv2
import numpy as np

import torch


class Compose:
    """Composes several transforms together.

    Args:
        transforms(list of 'Transform' object): list of transforms to compose

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):

        for trans in self.transforms:
            img = trans(img)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToCVImage:
    """Convert an Opencv image to a 3 channel uint8 image
    """

    def __call__(self, image):
        """
        Args:
            image (numpy array): Image to be converted to 32-bit floating point

        Returns:
            image (numpy array): Converted Image
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image = image.astype('uint8')

        return image


class RandomResizedCrop:
    """Randomly crop a rectangle region whose aspect ratio is randomly sampled
    in [3/4, 4/3] and area randomly sampled in [8%, 100%], then resize the cropped
    region into a 224-by-224 square image.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped (w / h)
        interpolation: Default: cv2.INTER_LINEAR:
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='linear'):

        self.methods = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4
        }

        self.size = (size, size)
        self.interpolation = self.methods[interpolation]
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        h, w, _ = img.shape

        area = w * h

        for attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            target_ratio = random.uniform(*self.ratio)

            output_h = int(round(math.sqrt(target_area * target_ratio)))
            output_w = int(round(math.sqrt(target_area / target_ratio)))

            if random.random() < 0.5:
                output_w, output_h = output_h, output_w

            if output_w <= w and output_h <= h:
                topleft_x = random.randint(0, w - output_w)
                topleft_y = random.randint(0, h - output_h)
                break

        if output_w > w or output_h > h:
            output_w = min(w, h)
            output_h = output_w
            topleft_x = random.randint(0, w - output_w)
            topleft_y = random.randint(0, h - output_w)

        cropped = img[topleft_y: topleft_y + output_h, topleft_x: topleft_x + output_w]

        resized = cv2.resize(cropped, self.size, interpolation=self.interpolation)

        return resized

    def __repr__(self):
        for name, inter in self.methods.items():
            if inter == self.interpolation:
                inter_name = name

        interpolate_str = inter_name
        format_str = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_str += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_str += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_str += ', interpolation={0})'.format(interpolate_str)

        return format_str


class RandomHorizontalFlip:
    """Horizontally flip the given opencv image with given probability p.

    Args:
        p: probability of the image being flipped
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() < self.p:
            img = cv2.flip(img, 1)

        return img


class ToTensor:
    """convert an opencv image (h, w, c) ndarray range from 0 to 255 to a pytorch
    float tensor (c, h, w) ranged from 0 to 1
    """

    def __call__(self, img):
        """
        Args:
            a numpy array (h, w, c) range from [0, 255]

        Returns:
            a pytorch tensor
        """
        # convert format H W C to C H W
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float() / 255.0

        return img


class Normalize:
    """Normalize a torch tensor (H, W, BGR order) with mean and standard deviation

    for each channel in torch tensor:
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean: sequence of means for each channel
        std: sequence of stds for each channel
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img):
        """
        Args:
            (H W C) format numpy array range from [0, 255]
        Returns:
            (H W C) format numpy array in float32 range from [0, 1]
        """
        assert torch.is_tensor(img) and img.ndimension() == 3, 'not an image tensor'

        if not self.inplace:
            img = img.clone()

        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        img.sub_(mean[:, None, None]).div_(std[:, None, None])

        return img


class Resize:

    def __init__(self, resized=256, interpolation='linear'):
        methods = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4
        }
        self.interpolation = methods[interpolation]

        if isinstance(resized, numbers.Number):
            resized = (resized, resized)

        self.resized = resized

    def __call__(self, img):
        img = cv2.resize(img, self.resized, interpolation=self.interpolation)

        return img
