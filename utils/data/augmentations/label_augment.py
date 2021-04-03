"""
Joint augmentations for input and target. I.e. equivariant augmentation.

"""

import random

from torchvision.transforms import RandomRotation

__all__ = ["EquivariantRotation"]

#TODO karen : docstring + probablility of witching that changes with the angle
class EquivariantRotation:

    def __init__(self, rot,  off_param=0.1, num_classes=10):
        pass # TODO

        # self.keep_prob = 1. - off_param

        # self.invariant_augmentations = RandomRotation(rot * self.keep_prob)
        # self.variant_augmentations_left = RandomRotation((rot * -1., -1.* rot * self.keep_prob))
        # self.variant_augmentations_right = RandomRotation((rot * self.keep_prob , rot))

    def __call__(self, data):
        pass # TODO
    
        # img, label = data

        # if img:
        #     if random.random() < self.keep_prob:
        #         self.label_change = False
        #         img = self.invariant_augmentations(img)
        #     else:
        #         self.label_change = True
        #         img = self.variant_augmentations_left(img) if random.random() < 0.5 else self.variant_augmentations_right(img)

        # if label:
        #     if self.label_change:
        #         random.randint(0,9)
        #     else:
        #         label

        # return img, label
