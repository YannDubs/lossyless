dependencies = ["torch", "compressai"]  # dependencies required for loading a model

import torch

from nts_net.model import attention_net

cub_200_2011_state_dict_url = "https://github.com/nicolalandro/ntsnet_cub200/releases/download/0.2/nts_net_cub200.pt"


def lossyless(pretrained=False, **kwargs):
    """ 


    NtsNET model
    pretrained (bool): kwargs, load pretrained weights into the model
    **kwargs
        topN (int): the number of crop to use
        num_classes (int): the number of output classes
        device (str): 'cuda' or 'cpu'
    """
    net = attention_net(**kwargs)
    if pretrained:
        from bird_classes import bird_classes

        net.load_state_dict(
            torch.hub.load_state_dict_from_url(
                cub_200_2011_state_dict_url, progress=True
            )
        )
        net.bird_classes = bird_classes
        # checkpoint = 'models/nts_net_cub200.pt'
        # state_dict = torch.load(checkpoint)
        # net.load_state_dict(state_dict)
    return net
