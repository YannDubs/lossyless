"""
Joint augmentations for input and target.

"""

import random

from torchvision.transforms import RandomRotation

__all__ = ["InvariantRotation"]

class InvariantRotation(object):

    def __init__(self, rot,  off_param=0.1, num_classes=10):

        self.keep_prob = 1. - off_param

        self.invariant_augmentations = RandomRotation(rot * self.keep_prob)
        self.variant_augmentations_left = RandomRotation((rot * -1., -1.* rot * self.keep_prob))
        self.variant_augmentations_right = RandomRotation((rot * self.keep_prob , rot))

    def __call__(self, data):

        img, label = data

        if img:
            if random.random() < self.keep_prob:
                self.label_change = False
                img = self.invariant_augmentations(img)
            else:
                self.label_change = True
                img = self.variant_augmentations_left(img) if random.random() < 0.5 else self.variant_augmentations_right(img)

        if label:
            if self.label_change:
                random.randint(0,9)
            else:
                label

        return img, label