import torch
from torchvision.models.video import r3d_18
from torchvision.models.feature_extraction import create_feature_extractor


class Res3D(torch.nn.Module):
    def __init__(self):
        super(Res3D, self).__init__()
        return_nodes = {
            'layer1.1.relu': 'layer1',
            'layer2.1.relu': 'layer2',
            'layer3.1.relu': 'layer3',
            'layer4.1.relu': 'layer4',
        }
        res_3d = r3d_18(pretrained=True)
        self.features = create_feature_extractor(res_3d, return_nodes=return_nodes)


    def forward(self, x):
        res = self.features(x)

        return res