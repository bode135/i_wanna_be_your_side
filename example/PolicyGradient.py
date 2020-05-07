import torch
import torch.nn as nn
import torch.nn.functional as F  # 激励函数都在这
from mycode.damo import Time
import matplotlib.pyplot as plt
import numpy as np
import gym
from mycode.my_time import Time

# env
if (1):
    #import retro
    #env = retro.make(game='Airstriker-Genesis')
    #env = gym.make('MountainCar-v0').unwrapped

    env = gym.make('CartPole-v0').unwrapped
    env.action_space.sample()
    N_ACTIONS = env.action_space.n

    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                                  int) else env.action_space.sample().shape  # to confirm the shape
    N_STATES = env.observation_space.shape[0]

torch.manual_seed(1)  # reproducible
np.random.seed(1)
env.seed(1)

# 控制参数
if(1):      # ALt + s 退出学习过程
    # Hyper Parameters
    LR = 0.02  # learning rate
    GAMMA = 0.99  # reward discount
    N_L1 = 100  # 神经元数量
    N_L2 = 10


    EPISODE = 1000      # 最大回合数
    MAX_REWARD = 1000  # 完成任务
    MAX_STEP = 5000  # 每一回合的最大迭代次数

    #END_REWARD = MAX_REWARD - 1
    END_REWARD = 0     # 完成任务给予奖励


    PLOT = 1
    Save_net = 0

    Plot_mean_reward = 0
    TEST = False

    MEMORY_CAPACITY = 200       # useless: 我忘了有啥用了,但删了会报错...


class Net(nn.Module):
    def __init__(self,
                 n_l1 = N_L1,
                 n_l2 = N_L2 ):
        super(Net, self).__init__()

        self.n_l1 = n_l1
        self.n_l2 = n_l2

        self.fc1 = nn.Linear(N_STATES, n_l1)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization

        self.fc2 = nn.Linear(n_l1, n_l2)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization

        self.out = nn.Linear(n_l2, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
    1

class Agent(object):
    def __init__(self):
        self.n_actions = N_ACTIONS
        self.n_states = N_STATES

        self.lr = LR
        self.gamma = GAMMA

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.ep = []

        self.memory = np.zeros([MEMORY_CAPACITY, N_STATES + 2], dtype=float)

        self.step = 0
        self.memory_index = 0


        #----------
        self.net = Net()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        # self.loss = 1
        # loss = self.loss_func(q_eval, q_target)
        # self.loss = loss.data.numpy()
        # self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        if(0):
            with tf.name_scope('loss'):
                # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
                neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
                                                                              labels=self.tf_acts)  # this is negative log of chosen action
                # or in this way
                # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
                loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


    def get_act_prob(self,state):
        state = torch.FloatTensor(state)
        self.act_forward = self.net.forward(state)
        self.act_prob = F.softmax(self.act_forward,dim = 0)  # use softmax to convert to probability
        return self.act_prob

    def choose_action(self,state, test = TEST):

        # --- Test:
        if (test):
            prob_weights = self.get_act_prob(state).data.numpy()
            action = np.where(prob_weights == np.max(prob_weights))[0][0]
            return action

        # --- Train:
        prob_weights = self.get_act_prob(state).data.numpy()
        #print('prob_weights',prob_weights)
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights )

        return action

    def store_transition(self,s,a,r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)


        self.sum_r = sum(self.ep_rs)
        self.length = len(self.ep_rs)
        self.mean_r = self.sum_r/self.length

        transition = np.hstack((s, [a, r]))
        self.ep.append(transition)

        index = self.memory_index % MEMORY_CAPACITY
        self.memory[index,:] = transition
        self.memory_index += 1
        self.step = self.memory_index

    def forget_ep(self):
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data

    def forget_memory(self):
        self.memory_index = 0
        self.memory = np.zeros([MEMORY_CAPACITY, N_STATES + 2], dtype=float)

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)

        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs = discounted_ep_rs/np.std(discounted_ep_rs)
        #discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def learn(self):
        # env.close()

        # get Loss ----- minmize: -(log_p * R)
        if (1):
            def one_hot(y):
                y_height = len(y)
                y_width = N_ACTIONS  # !!! Big坑
                y = torch.LongTensor(y).view(y_height, 1)
                y_oh = torch.zeros(y_height, y_width).scatter_(1, y, 1)
                return y_oh

            ep_obs, ep_as, ep_rs = self.ep_obs, self.ep_as, self.ep_rs
            y_oh = one_hot(ep_as)
            # debug one_hot(y)
            if(0):
                y = ep_as = agent.ep_as

                y_height = len(y)
                y_width = N_ACTIONS  # !!! Big坑

                y = torch.LongTensor(y).view(y_height, 1)
                y_oh = torch.zeros(y_height, y_width).scatter_(1, y, 1)
                y_oh
                1

            discounted_ep_rs = torch.from_numpy(agent._discount_and_norm_rewards()).detach()

            prob_as_forward = self.net.forward(torch.FloatTensor(ep_obs))
            prob_as = torch.softmax(prob_as_forward, 1)

            #print('y : ',self.ep_as)
            #print('y_: ', list(prob_as.max(1).indices.data.numpy() ))


            # cross_entropy
            neg_log_prob = -y_oh * torch.log(prob_as)
            neg_log_prob = neg_log_prob.sum(1)

            loss = torch.mean(neg_log_prob * discounted_ep_rs)
            #loss = torch.mean(neg_log_prob * self.sum_r)

        # debug loss
        if(0):
            def one_hot(y):
                y = torch.LongTensor(y).view(len(y),1)
                y.size()
                y_oh = torch.zeros(len(y), 2).scatter_(1, y, 1)
                return y_oh

            ep_obs, ep_as, ep_rs = agent.ep_obs, agent.ep_as, agent.ep_rs
            discounted_ep_rs = torch.from_numpy(agent._discount_and_norm_rewards()).detach()

            y_oh = one_hot(ep_as)


            prob_as_forward = agent.net.forward(torch.FloatTensor(ep_obs))
            prob_as = torch.softmax(prob_as_forward, 1)
            #prob_as = torch.sigmoid(prob_as_forward)

            # cross_entropy
            neg_log_prob = -y_oh*torch.log(prob_as)
            neg_log_prob = neg_log_prob.sum(1)

            # my loss
            loss = torch.mean(neg_log_prob * discounted_ep_rs)
            loss

        # env.close()

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # forget eperience and save loss
        self.forget_ep()

        self.loss = loss.data.numpy()

        return self.loss


    1

if __name__ == '__main__':


    agent = Agent()
    Rewards = [];loss_s = []; flag = False;tt = Time()

    # ------------------     Play
    for i_episode in range(EPISODE):
        s = env.reset()
        step = 1


        if (flag):
            break

        # ------------------------------
        while True:
            # if(step % 1000 == 0):
            #     print('step: ',step)

            if (tt.stop_0('s')):
                flag = True
                break

            env.render()
            a = agent.choose_action(s)
            s_, r, done, info = env.step(a)

            # modify reward
            if(1):  # mount_car
                if(done and step < MAX_STEP):
                    r = r + END_REWARD

            if (0):     # car_pole
                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r += r1 + r2



            agent.store_transition(s, a, r)

            s = s_

            step += 1
            if (step > MAX_STEP):
                print('--- MAX STEP! ---')
                done = 1

            if (done):
                break

        # ---
        agent.learn()

        print('第{:3}回合:{:6}分, loss: {:>10.6f}'.format(i_episode, agent.sum_r, agent.loss))
        Rewards.append(agent.sum_r)
        loss_s.append(agent.loss)
        if (agent.sum_r > MAX_REWARD):
            print('\n第{:3}回合 超过了{:4}分！ \n------一共获得:{:6}分，经历：{:5}步-----'.format(i_episode, MAX_REWARD, agent.sum_r,
                                                                                agent.step))
            break

        # agent.forget_ep()



        1
    env.close()

    if (agent.sum_r < MAX_REWARD):
        print('\n第 ', agent.step, '步终止\n'
              , '最终得分：:', Rewards[-1], '分; 未到达最大奖励:', MAX_REWARD, '分')

    print('运行时间： ', tt.now())
    # ----------   save model
    if (Save_net):
        model_name = 'policy_net.pkl'
        print('保存模型为： ' + model_name)
        torch.save(agent.net, model_name)

    # ---------------  plot
    if (PLOT):
        color = 'blue'
        ys = Rewards[:-1]  # 放大到达终点前的曲线变化
        xs = np.arange(0, len(ys))
        plt.plot(xs, ys, color=color)

        # plot the (mean ± 2*std)
        if (1):
            ys_avg = ys  # 中心线
            ys_std = np.ones(len(ys)) * np.std(ys)  # 方差

            # 最近邻方差
            if (1):
                cut_len = 10

                for i in range(len(ys)):
                    if (i > (len(ys) - cut_len // 2)):
                        ys_std[i] = np.std(ys[-cut_len:])
                        continue
                    if (i < cut_len // 2):
                        ys_std[i] = np.std(ys[:cut_len])
                    else:
                        # i = 10
                        ys_std[i] = np.std(Rewards[i - cut_len // 2:i + cut_len // 2])

            r1 = list(map(lambda x: x[0] - x[1], zip(ys_avg, ys_std)))
            r2 = list(map(lambda x: x[0] + x[1], zip(ys_avg, ys_std)))

            plt.fill_between(xs, r1, r2, color=color, alpha=0.1)

        plt.xlabel('episode')
        plt.ylabel('Reward')
        plt.show()

        # env.close()

        1

        # print mean_ys
        if (Plot_mean_reward):
            cut_len = 10
            # y = np.zeros(len(Rewards))
            y = []
            mean_y = Rewards[0]
            for i in range(len(Rewards)):
                # if(i%100 == 0 and i <= len(Rewards)-100):
                if (i % cut_len == 0):
                    end_i = i + cut_len
                    if (end_i > len(Rewards) - 1):
                        end_i = len(Rewards) - 1
                        mean_y = np.mean(Rewards[i:end_i])
                    y.append(np.mean(Rewards[i:end_i]))
                # y[i] = mean_y
            y
            plt.plot(y)
            plt.show()