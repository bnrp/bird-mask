import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        # Inspired by RseNet architectures: https://arxiv.org/pdf/1512.03385
        self.conv_7 = nn.Conv2d(3, 32, (7,7), stride=2)
        self.maxpool = nn.MaxPool2d((3,3),2)
        self.relu = nn.ReLU()
        self.res1 = nn.Sequential(
            nn.Conv2d(32, 32, (3,3), 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3,3), 1, 1),
        )
        
        #self.res2 = nn.Sequential(
        #    nn.Conv2d(64, 64, (3,3), 1, 1),
        #)

        self.res3 = nn.Sequential(
            nn.Conv2d(32, 256, (3,3), 2, 1),
        )
        
        self.res4 = nn.Sequential(
            nn.Conv2d(256, 256, (3,3), 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), 1, 1),
        )

        self.res6 = nn.Sequential(
            nn.Conv2d(256, 256, (3,3), 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), 1, 1),
        )

        self.avgpool = nn.AvgPool2d((2,2))

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 22),
        )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = self.conv_7(x)
        max_x = self.maxpool(x)
        x = self.res1(max_x) + max_x
        x = self.relu(x)
        x = self.res1(x) + x
        x = self.relu(x)
        x = self.res3(x)
        x = self.relu(x)
        x = self.res4(x) + x
        x = self.relu(x)
        #x = self.res5(x)
        #x = self.relu(x)
        x = self.res6(x) + x
        x = self.relu(x)
        #x = self.res6(x) + x
        #x = self.relu(x)
        #x = self.res6(x) + x
        #x = self.relu(x)
        x = self.avgpool(x)
        x = self.linear(x)
        outs = x
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
