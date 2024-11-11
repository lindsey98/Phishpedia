from .mobilenet import mobilenet_v2, mobilenetv3_small
from .resnetv2 import resnetv2_50x1
from .nasnet import nasnetamobile

__all__ = {'mobilenet_v2': mobilenet_v2,
           'mobilenet_v3': mobilenetv3_small,
           'resnet_v2': resnetv2_50x1,
           'nasnet': nasnetamobile
           }