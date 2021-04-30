import torch
import torch.nn as nn
import torch.functional as F
from early_stopping import EarlyStopping



class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, in_channel=3, zero_init_residual=False, size=64
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        second_stride = 2 if size > 32 else 1
        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=second_stride)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class CNN_Encoder(nn.Module):
    def __init__(self, in_channel=3, size=64):
        super().__init__()
        init_stride = 2 if size == 64 else 1
        init_padding = 1 if size == 64 else 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, init_stride, init_padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return torch.flatten(self.encoder(x), 1)


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def cnn_encoder(**kwargs):
    return CNN_Encoder(**kwargs)



class Encoder(nn.Module):
    def __init__(self, latent_dimension, backbone="resnet18", f_size=256, **bb_kwargs):
        super().__init__()

        features = 512 if "resnet" in backbone else 1024

        self.backbone = nn.Sequential(
            globals()[backbone](**bb_kwargs),
            nn.ReLU(),
            nn.Linear(features, f_size),
            nn.ReLU(),
        )
        self.latent_head = nn.Linear(f_size, latent_dimension)

    def forward(self, x):
        features = self.backbone(x)
        latent = self.latent_head(features)
        return latent


def _train(model, loader, opt):
    device = torch.device(opt.device + ':' + str(opt.device_id))
    early_stopping = EarlyStopping(patience=10, verbose=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.encoder.latent_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt.encoder.latent_step_size,
        gamma=opt.encoder.latent_gamma,
    )
    criterion = nn.MSELoss()

    for epoch in range(opt.encoder.latent_nb_epochs):
        train_loss = []
        valid_loss = []
        model.train()
        for images, labels in loader['train']:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        scheduler.step()
        model.eval()
        for images, labels in loader['valid']:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss.append(loss.item())

        train_loss_avg = sum(train_loss) / len(train_loss)
        valid_loss_avg = sum(valid_loss) / len(valid_loss)

        #         print('Epoch: {}, train_loss : {}, test_loss : {}'.format(epoch, train_loss_avg, valid_loss_avg))
        early_stopping(valid_loss_avg, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval()

    loss = []
    for images, labels in loader['test']:
        images = images.requires_grad_().to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss.append(criterion(outputs, labels).item())
    loss = sum(loss)/len(loss)
    print('Test_loss : {}'.format(loss))

    return model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out
