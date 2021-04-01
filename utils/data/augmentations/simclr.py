from pl_bolts.models.self_supervised.simclr.transforms import GaussianBlur
from torchvision import transforms
from torchvision.transforms.transforms import ToPILImage

__all__ = ["get_simclr_augmentations", "get_finetune_augmentations"]
# taken from pl_bolts.models.self_supervised.simclr.transforms
def get_simclr_augmentations(dataset, input_height):

    if dataset == "imagenet":
        jitter_strength = 1.0
        gaussian_blur = True
    elif dataset == "cifar10":
        jitter_strength = 0.5
        gaussian_blur = False
    else:
        raise ValueError(f"Unkown dataset={dataset} for simclr augmentations.")

    color_jitter = transforms.ColorJitter(
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.2 * jitter_strength,
    )

    data_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]

    if gaussian_blur:
        kernel_size = int(0.1 * input_height)
        if kernel_size % 2 == 0:
            kernel_size += 1

        data_transforms.append(GaussianBlur(kernel_size=kernel_size, p=0.5))
        data_transforms.append(ToPILImage())  # put back to PIL
   

    data_transforms = transforms.Compose(data_transforms)

    return data_transforms


# taken from pl_bolts.models.self_supervised.simclr.transforms
def get_finetune_augmentations():


    jitter_strength = 1.0

    color_jitter = transforms.ColorJitter(
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.2 * jitter_strength,
    )

    data_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]
 

    data_transforms = transforms.Compose(data_transforms)

    return data_transforms
