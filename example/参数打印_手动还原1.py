import numpy as np
import torch
import torch.nn.functional as F     # 激励函数都在这
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)    # reproducible

# data
if(1):
    import matplotlib.pyplot as plt

    # make fake data
    n0 = np.ones([100,2])
    n_data = torch.ones(100, 2)

    x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer
    x.data.numpy()
    type(x.data.numpy())
    type(x.data)
    import numpy as np

    xx = np.array([[1, 2], [3, 4]], dtype=int)
    xx.dtype

    t_xx = torch.tensor(xx, dtype=torch.int32)
    t_xx
    torch.tensor(t_xx, dtype=torch.float32)
    t_xx.float()
    t_xx.data.numpy()
    xx

    # The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
    #x, y = Variable(x), Variable(y)

    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    plt.show()



class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x


net = Net(2,2,2)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
plt.ion()   # something about plotting


for t in range(100):
    out = net(x)                 # input x and predict based on x

    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    if t % 2 == 0:
        print(loss)

    if (t == 99):   #the last times will not do gradient descent
        out1 = out
        loss1 = loss
        break

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients




# 提取网络参数，手动还原
if(1):

    # param = list(net.parameters())
    param = list(net.named_parameters())

    if(1):  #weights and bias
        weights = []
        bias = []
        for i in range(len(param)):
            if (i%2==0):
                weights.append(param[i][1].data.numpy())
            else:
                bias.append(param[i][1].data.numpy())
        weights = np.array(weights)
        bias = np.array(bias)

    param
    weights
    bias
    # torch.nn.Dropout

    #---    predict x and y
    xi = 1
    pred_x = x[-xi:]
    pred_out = net.forward(pred_x)


    # my pred x
    if(1):
        #weights and bias
        if(1):
            weights = []
            bias = []
            for i in range(len(param)):
                if (i%2==0):
                    weights.append(param[i][1].data.numpy())
                else:
                    bias.append(param[i][1].data.numpy())
            weights = np.array(weights)
            bias = np.array(bias)

        p_x = pred_x.data.numpy()
        w = weights[0]
        b = bias[0]
        def layout(x, weight, bias):
            #x = [1,2,3]
            #x = [[1],[2],[3]]
            #x = torch.tensor([1,2,3])
            x = np.array(x)
            x = x.reshape(-1,1)
            x
            dot = np.dot(weight, x)
            b =  bias.transpose()
            b = b.reshape(len(b),1)
            out_y = dot + b
            return out_y
        l1 = layout(p_x,w,b)
        l1
        def Relu(x):
            xt = x.copy()
            xt[xt < 0] = 0
            return xt
        f1 = Relu(l1)
        f1
        p_x
        w,b = weights[1],bias[1]
        l2 = layout(f1,w,b)
        l2

        #----------------------------------------------------
        #pred_out = net.forward(pred_x)
        #F.softmax(pred_out).max(1)
        #y[-xi:]

        print('my pred:{},torch pred:{}'.format( l2.reshape(-1), pred_out.data.numpy() ))
        F.softmax(pred_out).max(1).indices
        F.softmax(pred_out).max(1)
        torch.max(pred_out, 1)[1]

# Debug
if(1):
    zz = [[0.1, 0.2], [11.8, 0.6]]
    zz = torch.tensor(zz).reshape(len(zz), -1)
    zz
    prediction = torch.max(zz, 1)[1]
    prediction
    prediction = torch.max(out, 1)[1]
    prediction
    out