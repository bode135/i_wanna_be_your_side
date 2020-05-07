import torch
import torch.nn as nn
import torch.nn.functional as F  # 激励函数都在这
import numpy as np




class ActorCriticNet(nn.Module):

    def __init__(self,
                 n_input,
                 n_out,
                 n_l1 = 20,     # 隐藏层神经元数量
                 #n_l2 = 10
                 ):
        super(ActorCriticNet, self).__init__()
        self.n_l1 = n_l1

        self.fc1 = nn.Linear(n_input, n_l1)
        self.fc1.weight.data.normal_(0., 0.1)  # initialization
        self.fc1.bias.data.fill_( 0.1)

        self.out = nn.Linear(n_l1, n_out)
        self.out.weight.data.normal_(0., 0.1)  # initialization
        self.out.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.float_tensor(x)
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)

        if(actions_value.dim() > 1):
            acts_prob = F.softmax(actions_value,1)
        else:
            acts_prob = F.softmax(actions_value,0)

        return acts_prob

    def float_tensor(self, x):
        if (x.__class__.__name__ == 'int'):  # 若为int类型，先转换为list
            xx = []
            xx.append(x)
            x = xx

        if (x.__class__.__name__ != 'Tensor'):  # list和numpy类型转换为 FloatTensor 类型
            x = torch.FloatTensor(x)
        return x

    1

if __name__ == '__main__':
    n_input, n_out = 10,2
    x = torch.randn([100, n_input])
    x.requires_grad = True
    x
    x.shape
    a_net = ActorCriticNet(n_input, n_out)
    a_net
    a_net.forward(x)
