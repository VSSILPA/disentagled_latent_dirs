from torch import nn
from torchvision.models import resnet18


def save_hook(module, input, output):
    setattr(module, 'output', output)


class ResNetRankPredictor(nn.Module):
    def __init__(self, downsample=None, channels=3, num_dirs=10):
        super(ResNetRankPredictor, self).__init__()
        self.features_extractor = resnet18(pretrained=False)
        self.features_extractor.conv1 = nn.Conv2d(
            channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        self.shift_estimator = nn.Linear(512, num_dirs)

    def forward(self, x):
        batch_size = x.shape[0]
        self.features_extractor(x)
        features = self.features.output.view([batch_size, -1])

        shift = self.shift_estimator(features)

        return shift.squeeze()
