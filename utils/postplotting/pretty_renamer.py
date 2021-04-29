from utils.helpers import StrFormatter

__all__ = ["PRETTY_RENAMER"]

PRETTY_RENAMER = StrFormatter(
    exact_match={},
    subtring_replace={
        # Math stuff
        "H_Q_Zls": r"$\mathrm{H}_{\theta}[Z|S]$",
        "H_Q_Tlz": r"$\mathrm{H}_{\theta}[T|Z]$",
        "H_Q_Z": r"$\mathrm{H}_{\theta}[Z]$",
        "H_Q_S": r"$\mathrm{H}_{\theta}[S]$",
        "H_Ylz": r"$\mathrm{H}[Y|Z]$",
        "H_Zlx": r"$\mathrm{H}[Z|X]$",
        "H_Mlz": r"$\mathrm{H}[M(X)|Z]$",
        "H_Z": r"$\mathrm{H}[Z]$",
        "I_Q_Zx": r"$\mathrm{I}_{\theta}[Z;X]$",
        "I_Q_Zm": r"$\mathrm{I}_{\theta}[Z;M]$",
        "beta": r"$\beta$",
        # General
        "_": " ",
        "Resnet": "ResNet",
        "Ivae": "VIC",
        "Vae": "VC",
        "Dist": "Distortion",
        "Ince": "BINCE",
        "Banana_rot": "Rotation Inv. Banana",
        "Banana_xtrnslt": "X-axis Inv. Banana",
        "Bananay_trnslt": "Y-axis Inv. Banana",
        "Lr": "Learning Rate",
        "Online Loss": r"$\mathrm{H}_{\theta}[Y|Z]$",
    },
    to_upper=[
        "Cifar10",
        "Mnist",
        "Mlp",
        "Adam",
    ],
)
