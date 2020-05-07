import numpy as np
import random
from collections import namedtuple
from grabscreen import grab_screen, plt_img, cv_img
from args import arg
from utils import list_array
import cv2
from mycode.my_time import Time, vk
from A3C_1480 import ACNet, Worker



tt = Time()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
Transition = namedtuple('Transition',
                        ('s', 'a', 'r', 'info', 's_' ))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    1

from Env import Env
from utils import preprocess_state
if __name__ == '__main__':
    env = Env()

    memory = ReplayMemory(1000)

    # from ActorCritic import ActorCriticNet

    N_S, N_A  = arg.N_S , env.N_A

    # net = ActorCriticNet(N_S, N_A)

    len(env.action_name)
    s, position = env.reset(return_s_pos=1)
    s = preprocess_state(s, position, env)
    np.array(s).shape

    # net.forward(s[0])


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class DQN(nn.Module):

    def __init__(self, c, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, info = None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    def process_x(self, x, info = None):
        if(info != None):
            pass

N_C, N_H, N_W, N_A = arg.num_frames, arg.h, arg.w, env.N_A


dqn = DQN(N_C, arg.h, arg.w, N_A)

# from PIL import Image
# plt_img(s[-1])
# s = np.array(s)
# size_to = 40
# resize = T.Compose([T.ToPILImage(),
#                     T.Resize(size_to, interpolation=Image.CUBIC),
#                     # T.Grayscale(),
#                     T.ToTensor()])
# s.shape
# resize(s[0]).shape
# resize(np.array(s)).shape


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # if gpu is to be used

def process_x(s):

    # ss = np.array(s)
    ss = torch.FloatTensor(s)
    ss.shape
    # ss = resize(ss)
    # ss.shape
    ss = ss.unsqueeze(0)
    ss.shape
    # ss = ss.to(device)
    return ss
s.shape
ss = process_x(s)
ss.shape

dqn.forward(ss)
dqn(ss)
# self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
ss_c1 = dqn.conv1.forward(ss)
imgs = ss_c1.data.numpy()[0]
for img in imgs:
    plt_img(img)



dqn.conv1.weight[3]

(648 -  5 )// 2 + 1
img.shape
plt_img(s[-1])


