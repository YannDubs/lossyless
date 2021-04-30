""" Joint augmentations for input and target. When augmentations are out of the invariant range, targets may change.
"""

import numpy as np
import numpy.random as random
from collections.abc import Sequence
import numbers
import torch
from torchvision.transforms import RandomRotation, RandomResizedCrop

__all__ = ["EquivariantRandomRotation", "EquivariantRandomResizedCrop"]


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))

def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


def _torch_random_choice(x, pdf):

    # ! waiting for https://github.com/PyTorchLightning/pytorch-lightning/discussions/6957

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    assert len(x.shape) == 1, "Random choice array must be 1d, your got shape {}".format(x.shape)
    assert np.all(np.array(pdf) >= 0), "PDF need not to be normalized but counts must be positive."


    x_size = torch.tensor(x.size())

    pdf = pdf / np.sum(pdf)
    cdf = torch.tensor(np.cumsum(pdf))

    idx = x_size - torch.sum(torch.rand((1,)) < cdf)
    return int(x[idx].numpy()[0])





class EquivariantTransformation(torch.nn.Module):
    """Abstraction layer for augmentations on the label and the input. If sampled augmentation in invariant range:
        augments image and keeps the orignal target, but if augmentation in the equivariant range, change label with
        probability p.

        Corresponding augmentation parameter range:

        | left equivariant range |    invariant range           |     right equivariant range               |

        If the image is torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of
        leading dimensions. This class refers to torch.transforms.RandomRotation, args and kwargs apply.

        Args:
            p (float): Probability of changing the label if rotation is in the equivariant range.
            num_classes (integer): Size of the sample space for the label.
        """

    def __init__(self, p=1.0, num_classes=10):

        super().__init__()

        self.p = torch.tensor([p])
        self.num_classes = num_classes

    @property
    def pdf(self):
        """ Probability of augmentation being in the [left equivariant range, invariant range, right equivariant range].
            For uniform sampling should  be proportional to size of the respective range.
        """
        return self._pdf

    @pdf.setter
    def pdf(self, value):
        self._pdf = value

    def forward(self, data):
        """
                Args:
                    data (PIL Image or Tensor, int or Tensor): (image, label) to be augmented.

                Returns:
                    PIL Image or Tensor, int or Tensor: Rotated image and possibly shuffled target.
        """

        img, label = data
        aug_idx = _torch_random_choice([0,1,2], pdf=self.pdf)


        if aug_idx == 1:  # invariant range
            img = self.invariant_aug(img)
        elif aug_idx == 0:
            img = self.equivariant_aug_left(img)
            if torch.rand((1,)) < self.p:
                label = torch.randint(high = self.num_classes, size=(1,)).numpy()[0]
        else:
            img = self.equivariant_aug_right(img)
            if torch.rand((1,)) < self.p:
                label = torch.randint(high = self.num_classes, size=(1,)).numpy()[0]

        return img, label

class EquivariantRandomRotation(EquivariantTransformation):
    """EquivariantTransformation based on random rotation.

        If the image is torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of
        leading dimensions. This class refers to torch.transforms.RandomRotation, args and kwargs apply.

        Args:
            invariant_degrees (sequence or number): Range of degrees to select invariant actions from (label stays the
                same). If degrees is a number instead of sequence like (min, max), the range of degrees will be
                (-degrees, +degrees). This should contain the interval invariant_degrees.
            equivariant_degrees (sequence or number): Range of degrees to select equivariant actions from (label can
                change change with propability p). If degrees is a number instead of sequence like (min, max), the range
                of degrees will be (-degrees, +degrees).
        """

    def __init__(self, invariant_degrees, equivariant_degrees, p=1.0, num_classes=10, *args, **kwargs):

        super().__init__(p=p, num_classes=num_classes)

        self.invariant_aug = RandomRotation(degrees=invariant_degrees, *args, **kwargs)

        invariant_degrees = _setup_angle(invariant_degrees, name="EquivariantRandomRotation angles")
        equivariant_degrees = _setup_angle(equivariant_degrees, name="EquivariantRandomRotation angles")

        assert equivariant_degrees[0] <= invariant_degrees[0], "Problem with data augmentations: Range of " \
                                                               "equivariant degrees should entail invariant degrees."
        assert invariant_degrees[1] <= equivariant_degrees[1], "Problem with data augmentations: Range of " \
                                                               "equivariant degrees should entail invariant degrees."

        self.equivariant_aug_left = RandomRotation(degrees=(equivariant_degrees[0], invariant_degrees[0])
                                                   , *args, **kwargs)
        self.equivariant_aug_right = RandomRotation(degrees=(invariant_degrees[1], equivariant_degrees[1])
                                                    , *args, **kwargs)

        self.pdf = [invariant_degrees[0]-equivariant_degrees[0],
                  invariant_degrees[1]-invariant_degrees[0],
                  equivariant_degrees[1]-invariant_degrees[1]]


class EquivariantRandomResizedCrop(EquivariantTransformation):
    """EquivariantTransformation based on randomly resized crops.

        If the image is torch Tensor, it is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

        A crop of random size (default: of 0.08 to 1.0) of the original size and a random
        aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
        is finally resized to given size.
        This is popularly used to train the Inception networks.

        Args:
            size (int or sequence): expected output size of each edge. If size is an
                int instead of sequence like (h, w), a square output size ``(size, size)`` is
                made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
            invariant_scale (tuple of float): scale range of the cropped image before resizing, relatively to the
                origin image. This should contain the interval invariant_scale.
            equivariant_scale (tuple of float): scale range of the cropped image before resizing, relatively to the
                origin image.
        """

    def __init__(self, size, invariant_scale, equivariant_scale, p=1.0, num_classes=10, *args, **kwargs):

        super().__init__(p=p, num_classes=num_classes)

        if not isinstance(invariant_scale, Sequence):
            raise TypeError("Scale should be a sequence")

        if not isinstance(equivariant_scale, Sequence):
            raise TypeError("Scale should be a sequence")

        self.invariant_aug = RandomResizedCrop(size=size,scale=invariant_scale, *args, **kwargs)

        assert equivariant_scale[0] <= invariant_scale[0], "Problem with data augmentations: Range of equivariant " \
                                                              "scale should entail invariant scale."
        assert invariant_scale[1] <= equivariant_scale[1], "Problem with data augmentations: Range of equivariant " \
                                                              "scale should entail invariant scale."

        self.equivariant_aug_left = RandomResizedCrop(size=size,scale=(equivariant_scale[0], invariant_scale[0])
                                                   , *args, **kwargs)
        self.equivariant_aug_right = RandomResizedCrop(size=size,scale=(invariant_scale[1], equivariant_scale[1])
                                                    , *args, **kwargs)

        self.pdf = [invariant_scale[0] - equivariant_scale[0],
                    invariant_scale[1] - invariant_scale[0],
                    equivariant_scale[1] - invariant_scale[1]]
