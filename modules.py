# pooling.py
from torch import nn
import torch.nn.functional as F

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.

    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`

        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    """

    def __init__(self, norm_type, output_size):
        super(GeneralizedMeanPooling, self).__init__()
        self.output_size = output_size
        self.norm_type = norm_type

    def forward(self, input):
        out = F.adaptive_avg_pool2d(input.pow(self.norm_type), self.output_size)
        return out.pow(1. / self.norm_type)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.norm_type) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'

