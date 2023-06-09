##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/pytorch/pytorch
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" MTL CONV layers. """
import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair

class _ConvNdMtl(Module):
    """The class for meta-transfer convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNdMtl, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        # 参数一共有self.weight,self.bias,self.mtl_weight,self.mtl_bias
        # 其中weight和bias的名称不能改变，因为pretrain使用的是nn.Conv2d,需要把参数load进来的时候这里的名称需要和nn.Conv2d中参数的名称保持一致
        # self.mtl_weight,self.mtl_bias是SS(scaling and shifting)的参数，用于meta learning
        # 注意，weight和bias的requires_grad都会设为False，因为我们这里只需要对SS的参数进行meta learning，pretrain的参数是固定死的
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(in_channels, out_channels // groups, 1, 1))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(out_channels, in_channels // groups, 1, 1))
        self.weight.requires_grad=False
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias.requires_grad=False
            self.mtl_bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mtl_bias', None)
        self.reset_parameters() #初始化参数

    def reset_parameters(self):
        # 初始化参数，主要是mtl_weight和mtl_bias
        # weight和bias无所谓，因为后面会直接载入pretrain的数据
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.mtl_weight.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.mtl_bias.data.uniform_(0, 0)

    def extra_repr(self): # 输出convNd的信息，代码中并没有用到
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2dMtl(_ConvNdMtl):
    """The class for meta-transfer convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dMtl, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, inp):
        # 执行SS操作，先Scaling，再Shifting
        new_mtl_weight = self.mtl_weight.expand(self.weight.shape)
        new_weight = self.weight.mul(new_mtl_weight)
        if self.bias is not None:
            new_bias = self.bias + self.mtl_bias
        else:
            new_bias = None
        return F.conv2d(inp, new_weight, new_bias, self.stride,
                        self.padding, self.dilation, self.groups)
