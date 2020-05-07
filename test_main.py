from args import arg
import numpy as np
import cv2
from mycode.my_time import Time, vk
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from grabscreen import grab_screen, plt_img, cv_img
from process_image import pre_process_screen
from Env import Env
from model import DQN, Model


from plot import plot_list, plot_loss, plot_mean_loss, plot_net
from utils import preprocess_state, process_x, list_tensor, list_array, add_rects


LR = arg.LR
GAMMA = arg.GAMMA
EPSILON = arg.EPSILON
TARGET_REPLACE_ITER = arg.TARGET_REPLACE_ITER
BATCH_SIZE = arg.BATCH_SIZE
MEMORY_CAPACITY = arg.MEMORY_CAPACITY

device = arg.device

Transition = namedtuple('Transition',
                        ('s', 'a', 'r', 'info', 's_' ))

N_S = arg.h * arg.w
N_A = arg.N_A
N_C, N_H, N_W, N_A = arg.num_frames, arg.h, arg.w, arg.N_A
MAX_EPISODE = arg.MAX_EPISODE
MAX_STEP = arg.MAX_STEP


if __name__ == '__main__':

    env = Env()
    env.wind.move_to(0, 0)

    model = Model()
    tt = Time()
    steps = 0

    ############# ---------- Train
    break_flag = 0
    for i_episode in range(MAX_EPISODE):

        if (break_flag):
            break


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

            if (done):
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
                if (arg.show_orig_img):
                    img = pre_process_screen(img, ret_type=arg.img_ret_type)
                    add_rects(img=img, point=position_, env=env)
                    # img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)

                else:
                    img = s_[-1]
                    if (arg.show_resize_img):
                        img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)

                if (cv_img(img)):
                    break_flag = 1
                    break

            if (model.memory_counter > MEMORY_CAPACITY and steps % arg.LEARN_TMP_STEP == 0):
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

                if(model.loss == []):
                    loss = 0
                else:
                    loss = model.loss[-1]

                print(
                    'episode: {:3d}, reward: {:5.2f}, Loss: {:5.3f},   r_d: {:5.2f}, r_p: {:5.2f},---  steps: {:-6d},fresh_speed: {}'.format(
                        i_episode,
                        round(ep_r, 3),
                        loss,
                        sum(env.rewards_d),
                        sum(env.rewards_p),
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
    print(i_episode, ' episodes--steps: ', steps, 'cost_time: ', tt.now(), 'frequence: ', f, '-- step_per_second: ',step_per_second)
    1


    # s_ = process_x(s_)
    # # model.eval_net.train()
    # model.eval_net(s_)
    # model.target_net(s_)
    # torch.manual_seed(1)
    # model.target_net.load_state_dict(model.eval_net.state_dict())



    # ---------------  plot
    #               --> loss
    for i in range(len(model.loss)):
        if(i ==0):
            losses = model.loss[i].data.numpy()
        else:
            losses = np.row_stack( (losses, model.loss[i].data.numpy()) )
    # len(losses)
    # losses.reshape(-1)
    losses = losses.reshape(-1)[10:]
    plot_list(losses)
    plot_list(losses, Plot_mean_reward=1, cut_len=100)

    #               --> rewards
    Rewards = model.Rewards
    plot_list(Rewards, Plot_mean_reward=1, cut_len = 100)
    plot_list(Rewards)


    # ----------   save model
    if (arg.Save_net):
        model_name = 'policy_net.pkl'
        print('保存模型为： ' + model_name)
        torch.save(model.target_net, model_name)
    if(arg.Save_params):
        model_name = 'net_params.pkl'
        print('保存模型参数为： ' + model_name)
        torch.save(model.target_net.state_dict(), 'net_params.pkl')  # 只保存网络中的参数 (速度快, 占内存少)


