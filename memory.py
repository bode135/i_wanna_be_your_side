import numpy as np
import random
from collections import namedtuple
from grabscreen import grab_screen, plt_img, cv_img
from args import arg
import cv2
from mycode.my_time import Time, vk
from A3C_1480 import ACNet, Worker
from Env import Env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from utils import preprocess_state, process_x, list_tensor, list_array

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
        # if len(self.memory) < self.capacity:
        #     memory1 = self.memory[:position]
        #     return random.sample(memory1, batch_size)

        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':

    if (1):
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
                def conv2d_size_out(size, kernel_size=5, stride=2):
                    return (size - (kernel_size - 1) - 1) // stride + 1

                convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
                convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
                linear_input_size = convw * convh * 32
                self.head = nn.Linear(linear_input_size, outputs)

            # Called with either one element to determine next action, or a batch
            # during optimization. Returns tensor([[left0exp,right0exp]...]).
            def forward(self, x, info=None):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                x = self.head(x.view(x.size(0), -1))
                action_prob = torch.softmax(x, -1)
                return action_prob


        1

    ######################
    env = Env()
    N_C, N_H, N_W, N_A = arg.num_frames, arg.h, arg.w, env.N_A

    dqn = DQN(N_C, arg.h, arg.w, N_A)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if gpu is to be used
    device = "cpu"
    LR = 0.01
    GAMMA = 0.9
    policy_net = DQN(N_C, arg.h, arg.w, N_A).to(device)
    target_net = DQN(N_C, arg.h, arg.w, N_A).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)

    tt = Time()

    # Transition = namedtuple('Transition',
    #                         ('state', 'action', 'next_state', 'reward'))


    memory = ReplayMemory(1000)

    # env.step(-1)
    # s, position = env.reset(return_s_pos=1)
    # s = preprocess_state(s, position, env)

    # env.action_name -- ['left', 'right', 'shift_press', 'shift_up', 'stop']
    actions = [1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, -1]
    actions.__len__()
    # info
    steps = 0
    for episode in range(2):     # arg.MAX_EPISODE
        try:
            if(break_flag):
                break
        except:
            break_flag = False
        s, position = env.reset(return_s_pos=1)
        s = preprocess_state(s, position, env)

        for i in range(actions.__len__()):  # batch times
            if(tt.stop_alt()):
                break_flag = 1
                break

            steps += 1

            a = actions[i]
            s_, r, done, info = env.step(a)

            if (not done):
                position_, press_shift, pos_passed = info
                s_ = preprocess_state(s_, position_, env)

                # plot
                if (arg.show_pre_image and cv_img(s_[-1])):    break_flag = 1; break
            else:
                r = arg.reward_done
                position_ = (0, 0)
                s_ = preprocess_state(s_, position_, env)

                if (arg.show_pre_image and cv_img(s_[-1])):    break_flag = 1; break
                tt.sleep(0.5)
                break

            # memory.push(s, r, done, info, s_)
            memory.push(s, a, r, info, s_)

            # if(steps % 100 == 0):
            #     td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            #     actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            #     model.learn()

            s = s_
        print(i)
        # position = position_
    cv2.destroyAllWindows()

    # plt_img(s[-1])
    # plt_img(s_[-1])

    # ('s', 'a', 'r', 'info', 's_' )
    print('memory_len: ', len(memory))
    BATCH_SIZE = 10
    transitions = memory.sample(BATCH_SIZE)
    # transitions = memory.memory     ### no random sample
    # BATCH_SIZE = len(transitions)
    batch = Transition(*zip(*transitions))
    # ----------------------


    # buff_s, buff_a, buff_r, buff_info, buff_s_ = batch
    # info_array = np.array(buff_info)
    # buff_position, buff_press_shift, buff_pos_passed = info_array[:, 0], info_array[:, 1], info_array[:, 2]

    # print(len(buff_a), len(buff_s), len(buff_s), len(buff_position), len(info_array) )

    batch_s, batch_a, batch_r, batch_info, batch_s_ = batch

    info_array = np.array(batch_info)
    batch_position, batch_press_shift, batch_pos_passed = info_array[:, 0], info_array[:, 1], info_array[:, 2]


    # batch_s = process_x(batch_s)
    # batch_s_ = process_x(batch_s_)

    batch_s = torch.FloatTensor(batch_s)
    batch_s.shape
    batch_s_ = torch.FloatTensor(batch_s_)
    batch_s_.shape
    # batch_a = torch.LongTensor(list_array(batch_a))
    batch_a = list_tensor(batch_a, 'long')
    batch_r = list_tensor(batch_r)
    # batch_r = torch.FloatTensor(list_array(batch_r))

    q_eval = policy_net(batch_s).gather(1, batch_a)
    q_next = target_net(batch_s_).max(1)[0].view(BATCH_SIZE, 1).detach() # detach from graph, don't backpropagate
    q_target = batch_r + GAMMA * q_next

    loss = loss_func(q_eval, q_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    env.rewards
    env.rewards_d
    env.rewards_p
