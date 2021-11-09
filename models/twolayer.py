"""
Two Layer Network Model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(TwoLayerNet, self).__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, num_classes)
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        flaten_x = torch.flatten(x, start_dim = 1)
        layer_1 = torch.sigmoid(self.fc_1(flaten_x))
        out = self.fc_2(layer_1)
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
