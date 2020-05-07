import numpy as np
from collections import namedtuple
from grabscreen import grab_screen, plt_img, cv_img
from process_image import pre_process_screen
from args import arg
import cv2
from mycode.my_time import Time, vk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from Env import Env
from memory import ReplayMemory
from plot import plot_net

from utils import preprocess_state, process_x, list_tensor, list_array, add_rects

# UPDATE_GLOBAL_ITER = 5
# GAMMA = 0.9
# MAX_EP = 3000

# dqn = DQN(N_C, arg.h, arg.w, N_A)
device =arg.device
LR = arg.LR
GAMMA = arg.GAMMA
EPSILON = arg.EPSILON
TARGET_REPLACE_ITER = arg.TARGET_REPLACE_ITER
BATCH_SIZE = arg.BATCH_SIZE
MEMORY_CAPACITY = arg.MEMORY_CAPACITY

Transition = namedtuple('Transition',
                        ('s', 'a', 'r', 'info', 's_' ))

N_S = arg.h * arg.w
N_A = arg.N_A
N_C, N_H, N_W, N_A = arg.num_frames, arg.h, arg.w, arg.N_A
MAX_EPISODE = arg.MAX_EPISODE
MAX_STEP = arg.MAX_STEP

#  ----------------- class ----------------------
if(1):
    class DQN(nn.Module):

        def __init__(self, c, h, w, outputs):

            super(DQN, self).__init__()
            self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(32)

            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
            def conv2d_size_out(size, kernel_size=3, stride=2, padding = 1):
                return (size - kernel_size + 2*padding ) // stride + 1

            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
            linear_input_size = convw * convh * 32
            self.linear_input_size = linear_input_size
            # linear_input_size = linear_input_size + 3       # info_length: position, press_shift, pos_passed

            # print('---------', linear_input_size)
            self.fc = nn.Linear(linear_input_size, convw * convh )
            self.fc.weight
            self.head = nn.Linear(convw * convh  , outputs)

            self._initialize_weights()

        def _initialize_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    # nn.init.kaiming_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.LSTMCell):
                    nn.init.constant_(module.bias_ih, 0)
                    nn.init.constant_(module.bias_hh, 0)


        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x, info=None):
            if(not isinstance(x, torch.FloatTensor)):
                x = process_x(x)
            x = x.to(device)
            # x = x.permute(1,0,2,3)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = x.reshape(x.size(0), -1)

            if(info != None):       # self.info_add
                if(isinstance(info, tuple)):
                    info = torch.cat(info)
                x = torch.cat((x, info))

            x = F.relu(self.fc(x))
            x = self.head(x)
            # action_prob = torch.softmax(x, -1)

            return x



    class Model(object):
        def __init__(self):
            self.Rewards = []
            self.loss = []

            self.eval_net = DQN(N_C, arg.h, arg.w, N_A).to(device)
            if(arg.Reload_net):
                if(arg.Reload_params):
                    print('========== Reload params! ==========')
                    self.eval_net.load_state_dict(torch.load('net_params.pkl'))
                else:
                    print('========== Reload net! ==========')
                    self.eval_net = torch.load('policy_net.pkl')


            self.target_net = DQN(N_C, arg.h, arg.w, N_A).to(device)
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.target_net.eval()

            self.memory_counter = 0                                          # for storing memory
            self.learn_step_counter = 0                                     # for target updating
            self.memory = ReplayMemory(MEMORY_CAPACITY)  # initialize memory
            self.loss_func = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        def choose_action(self, x):
            self.eval_net.eval()

            N_ACTIONS = N_A
            x = process_x(x)
            x.to(device)
            # input only one sample
            if np.random.uniform() < EPSILON:   # greedy
                actions_value = self.eval_net.forward(x)
                if(device == 'cuda'):
                    action = torch.max(actions_value, 1)[1].cpu().data.numpy()
                if(device == 'cpu'):
                    action = torch.max(actions_value, 1)[1].data.numpy()
                action = action[0]
            else:   # random
                action = np.random.choice(arg.action_space)
                # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            return action

        def store_transition(self, s, a, r, info, s_):

            # transition = np.hstack((s, a, r, info, s_))
            # transition = (s, a, r, info, s_)
            self.memory.push(s, a, r, info, s_)
            self.memory_counter += 1

        def learn(self):
            self.eval_net.train()
            # target parameter update
            if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
                print('------- replace netwark!-------', self.learn_step_counter)
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter += 1

            # sample batch transitions
            memory = self.memory
            transitions = memory.sample(BATCH_SIZE)


            batch = Transition(*zip(*transitions))
            batch_s, batch_a, batch_r, batch_info, batch_s_ = batch

            info_array = np.array(batch_info)
            batch_position, batch_press_shift, batch_pos_passed = info_array[:, 0], info_array[:, 1], info_array[:, 2]

            batch_s = torch.FloatTensor(batch_s)
            batch_s_ = torch.FloatTensor(batch_s_)
            batch_a = list_tensor(batch_a, 'long')
            batch_r = list_tensor(batch_r)

            if(device == 'cuda'):
                batch_s.to(device)
                batch_s_.to(device)
                batch_a.to(device)
                batch_r.to(device)

            q_eval = self.eval_net(batch_s).cpu().gather(1, batch_a)
            q_next = self.target_net(batch_s_).max(1)[0].view(BATCH_SIZE, 1).detach().cpu()  # detach from graph, don't backpropagate
            q_target = batch_r + GAMMA * q_next

            loss = self.loss_func(q_eval, q_target)
            # loss = F.smooth_l1_loss(q_eval, q_target)
            self.loss.append(loss)


            err = q_eval - q_next
            if(arg.print_loss):       print('---- Loss ----> {:6.3f},  --- mean-err  -----> {:6.3f} ) '.format( float(loss.data.numpy()),  float (self.loss_func(q_eval, q_target).data.numpy()) ) )
            # q_eval -  q_target

            self.optimizer.zero_grad()
            loss.backward()


            if(arg.clamp_):
                # clamp_ = arg.LR * 1000
                clamp_ = arg.clamp_
                for param in self.eval_net.parameters():
                    param.grad.data.clamp_(-clamp_, clamp_)

            # print max_grad
            # tmp = 0
            # for param in self.eval_net.parameters():
            #     max_g = param.grad.cpu().data.numpy()
            #     mx = np.max(max_g)
            #     if (mx > tmp):
            #         tmp = mx
            # print('- max_grad: ',tmp)

            if(arg.plot_net):           plot_net(self.eval_net, 0)
            self.optimizer.step()


    # print('---->  name:{:10} ---->  grad_requirs: {} ---->  grad_value: {}'.format("name".rjust(25), "parms.requires_grad".rjust(25), "parms.grad.shape".rjust(25)))

    1


if __name__ == '__main__':

    env = Env()
    model = Model()
    tt = Time()

    steps = 0
    # arg.MAX_EPISODE
    # i_episode = 0
    for i_episode in range(MAX_EPISODE):
        try:
            if(break_flag):
                break
        except:
            break_flag = 0


        s, position = env.reset(return_s_pos=1)
        s = preprocess_state(s, position, env)

        # actions = [0, 0, 0, 1, 1, 1, -1]
        t1 = Time()
        ep_r = 0
        for i in range(MAX_STEP):
            if (tt.stop_alt('s')):
                print('----- break! -----')
                break_flag = 1
                break

            steps += 1
            # a = actions[i]        # len(actions)
            a = model.choose_action(s)
            s_, r, done, info = env.step(a)

            position_, press_shift, pos_passed = info
            img = s_[-1]
            s_ = preprocess_state(s_, position_, env)

            if( i == MAX_STEP - 1 or done):
                r = arg.reward_done

            model.store_transition(s, a, r, info, s_)
            ep_r += r

            # print(sum(env.rewards), sum(env.rewards_d), env.rewards_p)
            # print('{5:2f}, {:5.2f},   {:5.3f},   {:5.3f},'.format( (ep_r, sum(env.rewards), sum(env.rewards_d), sum(env.rewards_p))   )
            # print('{:5.2f},   {:5.3f},   {:5.3f},'.format( ep_r,  sum(env.rewards_d), sum(env.rewards_p)  ))
            # r, r_d, r_p = sum(env.rewards), sum(env.rewards_d), sum(env.rewards_p)
            # if (r_p != 0):
            #     d_p = r_d / r_p
            # else:
            #     d_p = -1

            # plot
            if (arg.show_pre_image):

                # img = s_[-1]
                if(arg.show_orig_img):
                    img = pre_process_screen(img, ret_type=arg.img_ret_type)
                    add_rects(img=img, point=position_, env=env)
                    # img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)

                else:
                    img = s_[-1]
                    if(arg.show_resize_img):
                        img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)

                if(cv_img(img)):
                    break_flag = 1
                    break

            if(model.memory_counter > MEMORY_CAPACITY and steps % arg.LEARN_TMP_STEP == 0):
                model.learn()
                # print('----------------------------- learn ------------------', model.learn_step_counter)
                # if done:
                #     print('Ep: ', i_episode,
                #           '| Ep_r: ', round(ep_r, 2))


            # ----------  importrant data ----------------------
            # ------------- r_d and r_p

            # print('r: {:5.1f}, steps: {:3}, r_d: {:5.2f}, r_p: {:5.2f}, d/p: {:-8.3f}'.format( r, steps, r_d, r_p, d_p))

            if (done or i == MAX_STEP - 1):

                step = i + 1
                model.Rewards.append(ep_r)

                f = round(t1.now() / step, 3)
                print(
                    'episode: {:4d}, reward: {:5.2f}, r_d: {:5.2f}, r_p: {:5.2f}, steps: {:-6d}, fresh_speed: {}'.format(
                        i_episode,
                        round(ep_r, 3),
                        sum(env.rewards_d), sum(env.rewards_p),
                        steps,
                        f,
                    ))
                # r = arg.reward_done
                # position_ = (0, 0)
                # s_ = preprocess_state(s_, position_, env)
                # if (arg.show_pre_image and cv_img(s_[-1])):    break_flag = 1; break
                # tt.sleep(0.5)
                break
            s = s_

    cv2.destroyAllWindows()
    f = round(tt.now() / steps, 3)
    step_per_second = round(1 / f, 3)
    print('##############################################################')
    print(i_episode, ' episodes--steps: ', steps, 'cost_time: ', tt.now(), 'frequence: ', f, '-- step_per_second: ',
          step_per_second)
    1


    s_

    s_ = process_x(s_)
    # model.eval_net.train()
    model.eval_net(s_)
    model.target_net(s_)
    # torch.manual_seed(1)
    # model.target_net.load_state_dict(model.eval_net.state_dict())

    # ---------------  plot
    import matplotlib.pyplot as plt

    PLOT = True
    Plot_mean_reward = 0
    Rewards = model.Rewards
    # def plot_mean_loss(ys):
    #     cut_len = 10
    #     # y = np.zeros(len(ys))
    #     y = []
    #     mean_y = ys[0]
    #     for i in range(len(ys)):
    #         # if(i%100 == 0 and i <= len(ys)-100):
    #         if (i % cut_len == 0):
    #             end_i = i + cut_len
    #             if (end_i > len(ys) - 1):
    #                 end_i = len(ys) - 1
    #                 mean_y = np.mean(ys[i:end_i])
    #             y.append(np.mean(ys[i:end_i]))
    #     plt.plot(y)
    #     plt.xlabel('episode')
    #     plt.ylabel('Loss')
    #     plt.show()

    #
    # if (PLOT_LOSS):
    #     color = 'blue'
    #     ys = loss_s  # 放大到达终点前的曲线变化
    #     xs = np.arange(0, len(ys))
    #     plt.plot(xs, ys, color=color)
    #     plt.xlabel('episode')
    #     plt.ylabel('Loss')
    #     plt.show()
    # if (PLOT_MEAN_LOSS):
    #     plot_mean_loss(loss_s)

    # ----------   save model
    if (arg.Save_net):
        model_name = 'policy_net.pkl'
        print('保存模型为： ' + model_name)
        torch.save(model.target_net, model_name)

    if (PLOT):
        # len(Rewards)
        color = 'blue'
        ys = Rewards  # 放大到达终点前的曲线变化
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





