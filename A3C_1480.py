import multiprocessing
import threading
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import torch
from ActorCritic import ActorCriticNet
from mycode.my_time import Time
import torch.nn as nn
import torch.nn.functional as F  # 激励函数都在这


GAME = 'CartPole-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.005    # learning rate for actor
LR_C = 100   # learning rate for critic
seed = 1        # random_seed


# print(' LR_A, LR_C -------> ', LR_A, LR_C)
# (lr_a, lr_c) - best_reward
# --- seed: 1 ---- max_step: 1000 --------
# lr_c is no important, set 0.01; lr_a == 0.05 -> max
# (0.001, 0.005) - 456.8;
# (0.001, 0.001) - 456.8;
# (0.002, 0.01) - 699.2;

# (0.006, 0.01) - 947.9;    # over fit
# (0.005, 0.01) - 1483.8;   ***
# (0.005, 0.1) - 1483.8;
# (0.004, 0.01) - 273.8;


# (0.01, 0.001) - 194.7;
# (0.01, 0.0012) - 194.7;



GLOBAL_RUNNING_R = []
GLOBAL_EP = 0


env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n
torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)      # reproducible

#OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
#OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
tt = Time()

class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:  # get global network
            #self.a_params, self.c_params = self._build_net(scope)[-2:]
            pass
        else:
            self.s = [torch.FloatTensor([0.]*N_S)]
            self.a_his = []
            self.v_target = np.array([])

            self._build_net(scope)
            self.optimizer_a = torch.optim.Adam(self.a_net.parameters(), lr=LR_A)
            self.optimizer_c = torch.optim.Adam(self.c_net.parameters(), lr=LR_C)


            # self.a_prob, self.v,= self._build_net(scope)
            #
            #
            # # Critic loss
            # td = self.v_target - self.v
            # self.c_loss = torch.mean(td.pow(2))
            # Actor loss
            # log_prob = 0


    def _build_net(self, scope):
        self.a_net = ActorCriticNet(n_input = N_S, n_out = N_A)     # 输出N_A个, 动作概率
        self.c_net = ActorCriticNet(n_input = N_S, n_out = 1)       # 输出1个, 当前状态的价值


    def update_global(self, grad_a, grad_c, worker_name):  # run by a local
        #SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net
        pass

    def pull_global(self):  # run by a local
        pass

    def one_hot(self,y):
        if(y.__class__.__name__ == 'int'):
            # 将int转换为list
            y = a
            yy = []
            yy.append(y)
            y = yy

        y_height = len(y)
        y_width = N_A  # !!! Big坑
        y = torch.LongTensor(y).view(y_height, 1)
        y_oh = torch.zeros(y_height, y_width).scatter_(1, y, 1)

        return y_oh

    def choose_action(self, state):
        #prob_weights = self.get_act_prob(state).data.numpy()    # get probabilities for all actions
        prob_weights = self.a_net.forward(state).data.numpy()
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)
        return action

    def learn(self, buffer_s, buffer_a, buffer_v_target):   # 输入(s,a,v), 输出actor和critic的梯度grad_a,grad_c

        s, a_his, v_target = buffer_s, buffer_a, buffer_v_target
        a_prob, v = self.a_net.forward(s), self.c_net.forward(s)

        # --- c_loss
        buffer_v_target = buffer_v_target.view(len(buffer_v_target), 1)

        advance = torch.FloatTensor( buffer_v_target - v )
        np_advance = advance.data.numpy()

        self.c_loss =F.mse_loss(buffer_v_target , v)

        # --- a_loss

        y_oh = self.one_hot(a_his)
        log_prob = torch.sum(y_oh * torch.log(a_prob + 1e-16))  # 防止概率为0出现nan


        exp_v = torch.mean(torch.FloatTensor(np_advance) * log_prob)
        self.exp_v = exp_v
        self.a_loss = torch.mean(-exp_v)

        # --- grad
        self.optimizer_a.zero_grad()
        self.grad_a = self.a_loss.backward()
        self.optimizer_a.step()

        self.optimizer_c.zero_grad()
        self.grad_c = self.c_loss.backward()
        self.optimizer_c.step()

        return self.grad_a, self.grad_c



class Worker():
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.env.seed(seed)     # !-------- 异步的时候必须重置 seed !
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        break_flag = 0
        while GLOBAL_EP < MAX_GLOBAL_EP:
            if(break_flag):
                break

            s = self.env.reset()
            ep_r = 0
            while True:
                if(tt.stop_alt('p')):
                    break_flag = True
                    break

                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done: r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.AC.c_net.forward(s_).data.numpy()

                    #buffer_v_target = torch.FloatTensor()
                    buffer_v_target = []

                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_     # v_eval
                        buffer_v_target.append(v_s_)
                    buffer_v_target
                    buffer_v_target[0]


                    buffer_v_target.reverse()
                    buffer_v_target = torch.FloatTensor(np.array(buffer_v_target))



                    #buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    #buffer_s, buffer_a, buffer_v_target = torch.FloatTensor(buffer_s), torch.FloatTensor(buffer_a), torch.FloatTensor(buffer_v_target)
                    buffer_s = torch.FloatTensor(buffer_s)

                    buffer_a = torch.LongTensor(buffer_a)


                    # --- learn
                    grad_a, grad_c = self.AC.learn(buffer_s, buffer_a, buffer_v_target)
                    self.AC.update_global(grad_a, grad_c, self.name)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    GLOBAL_EP += 1
                    break



if __name__ == '__main__':
    GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
    workers = []
    # Create worker
    for i in range(N_WORKERS):
        i_name = 'W_%i' % i  # worker name
        workers.append(Worker(i_name, GLOBAL_AC))

    worker = workers[0]
    worker.env.reset()
    worker.name
    # worker.env.render()
    # worker.env.close()
    worker.work()





    print('----- parameters: {}, \n----- episode: {}, Reward:{:.1f} '.format( (LR_A, LR_C),
                                           GLOBAL_RUNNING_R.index( max(GLOBAL_RUNNING_R)), max(GLOBAL_RUNNING_R) ) )
    print('# ({}, {}) - {:.1f};'.format(LR_A, LR_C,  max(GLOBAL_RUNNING_R)))

    if (0):

        a_net = ActorCriticNet(n_input=N_S, n_out=N_A)
        n_input, n_out = N_S, N_A
        x = torch.randn([100, n_input])
        print(a_net.forward(x))


        for i in range(100):
            tt.sleep(0.1)
            if (tt.stop_alt()):
                print('break!')
                break

        1

