##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Model for meta-transfer learning. """
import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_mtl import ResNetMtl
from torchvision import models

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w) #kaiming正态分布初始化(对relu较优的初始化参数)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None): # 允许传入特定参数进行forward，这常用于validation/test的时候进行finetune操作
                                               # (我们finetune的时候我们只是希望得到梯度下降几步之后的参数和用这个参数得到的答案和loss值，我们并不希望实际更改我们train之后得到的数据)
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        
        if self.mode == 'meta':
            # self.encoder = ResNetMtl()
            self.encoder = models.resnet50(pretrained=False) # !!! 使用其他数据pretrained好的ResNet50作为pretrain模型
            self.num_ftrs = self.encoder.fc.in_features
            # print("self.num_ftrs=",self.num_ftrs,"\n") # self.num_ftrs=2048
            # for param in self.encoder.parameters():
            #     param.requires_grad = False # False：冻结模型的参数，也就是采用该模型已经训练好的原始参数。只需要训练我们自己定义的Linear层
            
            self.encoder.fc=torch.nn.Identity() # 去除最后一层(即改为Identity())
        else:
            self.encoder = models.resnet50(pretrained=False) # !!! 使用其他数据pretrained好的ResNet50作为pretrain模型
            self.num_ftrs = self.encoder.fc.in_features
            self.encoder.fc=torch.nn.Identity() # 去除最后一层(即改为Identity())
            self.pre_fc = nn.Sequential(nn.Linear(self.num_ftrs, 1000), nn.ReLU(), nn.Linear(1000, num_cls))
        # z_dim = 640
        z_dim = self.num_ftrs # !!! 使用其他数据pretrained好的ResNet50作为pretrain模型
        self.base_learner = BaseLearner(args, z_dim)

    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode=='pre':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        elif self.mode=='metaval':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        return self.pre_fc(self.encoder(inp)) # pretrain就是encoder加一个FC层(我们训练出来的参数到pre validation/test,meta train,meta validation/test中只有encoder是需要的，FC层的参数将会舍弃)(因为要拿最后一层做finetune)

    def meta_forward(self, data_shot, label_shot, data_query): # 既可以用来做meta train，也可以用来做meta validation/test
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        # 这里的整体流程基本和pre validation/test一致，meta validation/test和pre validation/test基本一致很显然，下面考虑meta train使用下述代码的主要流程
        # meta train实际上也是希望和meta validation/test的时候一样，在对最后用于finetune的FC层参数进行多次梯度下降后，再计算此时所有想要更新参数的导数并在外面backward更新
        # 注意最后需要更新的导数有 FC层的参数，SS的参数(分为Scaling和Shifting两类),但是之前进行多次梯度下降的时候只有FC层的参数进行了更新
        # 尤其注意，前面100次对support set进行梯度下降，每次的梯度是对当前修改过的参数进行求导；而最后对query set进行梯度求导即最后一次进行梯度求导是对原参数进行的梯度求导
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)
        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
        logits_q = self.base_learner(embedding_query, fast_weights) 
        return logits_q

    def preval_forward(self, data_shot, label_shot, data_query): # 注意，pre validation/test都用的是这段代码
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)   # encoder是确定的，变的只是后面的finetune参数，所以直接把encoder的结果拿出来，后面就不重复算了
        logits = self.base_learner(embedding_shot) # logit一开始指的是log-odds，odds是胜算，即odds=p/(1-p),logit=log(p/(1-p)),它的取值为(-inf,+inf)。这里把它理解成一个元素取值为(-inf,+inf)的向量，它通过softmax得到的就是概率
        loss = F.cross_entropy(logits, label_shot) # 注 F.cross_entropy(x,y)会对第一参数x做softmax，使其满足归一化要求,并对y做one-hot编码，即返回 - \sum(one-hot(y) * log(softmax(x)))
        grad = torch.autograd.grad(loss, self.base_learner.parameters()) # 获取导数，我们不能直接backward，因为我们是在validation/test中进行finetune，我们不希望训练之后的参数进行修改
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters()))) # fast_weights记录的是finetune之后的参数
        logits_q = self.base_learner(embedding_query, fast_weights)
        # 注意，第一次base_learner的参数是随机kaiming正态分布的，即我们是从一个随机kaiming正态分布的参数开始进行100次梯度下降得到我们想要的参数

        for _ in range(1, 100): # 重复上述梯度下降共99次(包括上面一共有100次)，过程为 求output(logit形式) -> 用cross_entropy求loss -> 求grad -> 梯度下降算出新的参数fast_weights并更新
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
        logits_q = self.base_learner(embedding_query, fast_weights)         #这里其实可以只在最后做一次，但是原代码这么写了，就先不改了
        return logits_q #返回100次梯度下降后的参数对query集计算的output(logit形式)
        