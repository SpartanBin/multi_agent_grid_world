import torch
from torch import nn


class residual_blocks(nn.Module):

    def __init__(self):

        super(residual_blocks, self).__init__()

        self.conv1 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)

        self.act3 = nn.ReLU()

    def forward(self, features: torch.Tensor):
        """
        Forward pass in network

        :param features:
        :return:
        """

        features_ = self.act1(self.bn1(self.conv1(features)))
        features_ = self.bn2(self.conv2(features_))
        features_ = self.act3(features_ + features)

        return features_


class body_model(nn.Module):

    def __init__(self, in_channels):

        super(body_model, self).__init__()

        self.in_channels = in_channels

        conv_net = []
        conv_net.append(nn.Conv2d(self.in_channels, 64, (3, 3), padding=(1, 1)))
        conv_net.append(nn.BatchNorm2d(64))
        conv_net.append(nn.ReLU())
        for _ in range(4):
            conv_net.append(residual_blocks())
        conv_net.append(nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1)))
        conv_net.append(nn.BatchNorm2d(128))
        conv_net.append(nn.ReLU())
        conv_net.append(nn.Conv2d(128, 128, (3, 3), padding=(1, 1)))
        conv_net.append(nn.BatchNorm2d(128))
        conv_net.append(nn.ReLU())
        # # for 30 * 30
        # conv_net.append(nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1)))
        # conv_net.append(nn.BatchNorm2d(128))
        # conv_net.append(nn.ReLU())
        # conv_net.append(nn.Conv2d(128, 128, (3, 3), padding=(1, 1)))
        # conv_net.append(nn.BatchNorm2d(128))
        # conv_net.append(nn.ReLU())
        # conv_net.append(nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1)))
        # conv_net.append(nn.BatchNorm2d(128))
        # conv_net.append(nn.ReLU())
        # conv_net.append(nn.Conv2d(128, 128, (3, 3), padding=(1, 1)))
        # conv_net.append(nn.BatchNorm2d(128))
        # conv_net.append(nn.ReLU())
        self.conv_net = nn.Sequential(*conv_net)

        output_net = []
        output_dim = [3200, 64, 32]  # for 10 * 10: 3200, for 30 * 30: 2048
        for i in range(len(output_dim)):
            if i > 0:
                output_net.append(nn.Linear(output_dim[i - 1], output_dim[i]))
                output_net.append(nn.Tanh())
        self.output_net = nn.Sequential(*output_net)

    def forward(self, state_features: torch.Tensor):
        """
        Forward pass in network

        :param state_features:
        :return:
        """

        features = self.conv_net(state_features)
        features = torch.flatten(features, 1, -1)
        output = self.output_net(features)

        return output