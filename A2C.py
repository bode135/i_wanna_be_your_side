import torch
import torch.nn as nn
import torch.nn.functional as F  # 激励函数都在这
from mycode.damo import Time
import matplotlib.pyplot as plt
import numpy as np
import gym
from mycode.my_time import Time


# 控制参数
if(1):      # ALt + s 退出学习过程
    MAX_EPISODE = 3000
    DISPLAY_REWARD_THRESHOLD = 900  # renders environment if total episode reward is greater then this threshold
    MAX_EP_STEPS = 1000  # maximum time step in one episode
    RENDER = False  # rendering wastes time
    GAMMA = 0.9  # reward discount in TD error
    LR_A = 0.01  # learning rate for actor
    LR_C = 0.05  # learning rate for critic
    # --- 调参 tunning parameters
    # GAMMA: 0.9  MAX_EP_STEPS: 1000, THRESHOLD: 900
    # LR_A LR_C = (0.01 0.05); end = (episode: 314, reward: 900)

    # normal: 0.01, 0.04;
    # good: 0.001 0.05
    # bod:0.01 0.03; 0.01, 0.06;
    # gamma = bad: 0.95, 0.99, 0.8
    # gamma, lr_a, lr_c: 0.8, 0.01, 0.04; end: good


# env
if (1):
    # env = gym.make('MountainCar-v0').unwrapped
    env = gym.make('CartPole-v0').unwrapped

    env.action_space.sample()
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    torch.manual_seed(1)  # reproducible
    np.random.seed(1)
    env.seed(1)

class A_Net(nn.Module):
    def __init__(self,
                 n_l1 = 20,
                 #n_l2 = 10
                 ):
        super(A_Net, self).__init__()

        self.n_l1 = n_l1

        self.fc1 = nn.Linear(N_F, n_l1)
        self.fc1.weight.data.normal_(0., 0.1)  # initialization
        self.fc1.bias.data.fill_( 0.1)

        self.out = nn.Linear(n_l1, N_A)
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

class Actor(object):
    def __init__(self):
        self.net = A_Net()
        self.optimizer= torch.optim.Adam(self.net.parameters(), lr=LR_A)

    def one_hot(y):
        if(y.__class__.__name__ == 'int'):
            # 将int转换为list
            y = a
            yy = []
            yy.append(y)
            y = yy

        y_height = len(y)
        y_width = N_A  # !!! Big坑
        y = torch.LongTensor(y).view(y_height, 1)
        y_oh =  torch.zeros(y_height, y_width).scatter_(1, y, 1)
        return y_oh

    def get_act_prob(self, state):
        try:
            probs = torch.softmax( actor.net.forward(torch.FloatTensor(state)), 1 ) # 矩阵形式
        except:
            probs = torch.softmax( actor.net.forward(torch.FloatTensor(state)), 0 ) #向量形式
        return probs

    def choose_action(self, state):
        #prob_weights = self.get_act_prob(state).data.numpy()    # get probabilities for all actions
        prob_weights = self.net.forward(state).data.numpy()
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)
        return action

    def learn(self, s, a, td):
        # get actions_prob
        acts_prob = self.net.forward(s)

        # get Loss ----- minmize: -(log_p * R)
        log_prob = torch.log(acts_prob[a])
        exp_v = torch.mean( log_prob * td )
        loss = -exp_v

        # ------- Grad
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return exp_v

class C_Net(nn.Module):
    def __init__(self,):
        super(C_Net, self).__init__()

        n_hidden = 20

        self.hidden = torch.nn.Linear(N_F, n_hidden)   # hidden layer
        self.hidden.weight.data.normal_(0, 0.1)  # initialization
        self.hidden.bias.data.fill_(0.1)  # initialization

        self.predict = torch.nn.Linear(n_hidden, 1)  # output layer
        self.predict.weight.data.normal_(0, 0.1)  # initialization
        self.predict.bias.data.fill_(0.1)  # initialization
    def float_tensor(self, x):
        if (x.__class__.__name__ == 'int'):  # 若为int类型，先转换为list
            xx = []
            xx.append(x)
            x = xx

        if (x.__class__.__name__ != 'Tensor'):  # list和numpy类型转换为 FloatTensor 类型
            x = torch.FloatTensor(x)
        return x
    def forward(self, x):
        x = self.float_tensor(x)    # data type

        x = self.hidden(x)
        x = F.relu(x)
        state_value = self.predict(x)
        return state_value
    1

class Critic(object):
    def __init__(self):
        self.net = C_Net()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR_C)

    def learn(self, s, r, s_):
        # td_error = self.r + GAMMA * self.v_  - self.v
        v_ = self.net.forward(s_).detach()
        q_eval = r + GAMMA * v_

        v = self.net.forward(s)
        # 优势函数 = 当前的动作值函数 - 当前的值函数
        td_error = q_eval- v
        #loss = td_error.pow(2)/2
        loss = F.mse_loss(input= q_eval, target= v)

        # ------- Grad
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_error.data

if __name__ == '__main__':
    actor = Actor()
    critic = Critic()
    tt = Time()

    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER: env.render()

            a = actor.choose_action(s)

            s_, r, done, info = env.step(a)

            if done: r = -20

            track_r.append(r)

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                break