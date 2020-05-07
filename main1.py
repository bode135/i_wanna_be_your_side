from Env import Env
from wind import Window
from mycode.my_time import Time, vk
import numpy as np
from template import match_img
import cv2
from grabscreen import grab_screen, plt_img, cv_img
from process_image import pre_process_screen
from scipy.spatial.distance import pdist
from args import arg
from utils import dist_score, img_add_rect
from utils import add_rects
from utils import preprocess_state



def dist_score(x, x_max):
    # dist 对应的 r
    dd = (x_max - x) / x_max
    r_d = 1 - dd
    if (x < x_max):
        return r_d
    else:
        return r_d




def cv_imshow(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    env = Env()
    EPISODE = arg.MAX_EPISODE
    MAX_STEP = arg.MAX_STEP

    tt = Time();
    break_flag = 0;
    steps = 0
    for i_episode in range(EPISODE):
        if (break_flag):
            cv2.destroyAllWindows()
            break

        t1 = Time();rewards = []
        s, position = env.reset(sleep_t= 0.02,return_s_pos=1)



        # if (arg.debug_position_error and isinstance(position, int)):  print('------------ position_error!')

        for step in range(MAX_STEP):
            if (tt.stop_alt('s')):
                print('----- break! -----')
                break_flag = 1
                break

            if (len(s) != env.num_frames):   print('error len!')    # action = model.choose_action(s)
            ############################################################
            action_p = [0.1, 0.7, 0.1, 0.1, 0.]
            # action_p = np.ones(env.N_A) / env.N_A
            action = np.random.choice(env.action_space,  p = action_p)

            # s_ = env.get_state(action)
            # r, done, info = env.get_feedback(s_)

            s_ , r, done, info = env.step(action)

            # -------------  summary
            rewards.append(r)

            r, r_d, r_p = sum(env.rewards), sum(env.rewards_d), sum(env.rewards_p)
            if(r_p != 0):
                d_p = r_d/r_p
            else:
                d_p = -1
            # ------------- r_d and r_p
            print('r: {:5.3f}, r_d: {:5.3f}, r_p: {:5.3f}, d/p: {:-8.3f}'.format( r, r_d, r_p , d_p ))

            ################### ------------------- pre_process_image
            if (not done):  # plot cv_img
                position_ = info[0]
                # print('NOT ------------------------ done', position_)

                # if (arg.preprocess_state):
                s_ = preprocess_state(s_, position_, env, resize=arg.resize)
                    # s_ = preprocess_state(s_)

                # add points
                # if(arg.resize):
                #     for img in s_:
                #         add_rects(img=img, point=position_, env=env)
                #         img = cv2.resize(img, arg.wind_conv_wh, interpolation=cv2.INTER_AREA)
                #         # print(len(s_), env.num_frames)
                # else:
                #     img = s_[-1]
                #     add_rects(img=img, point=position_, env=env)

                # plot
                if (arg.show_pre_image and cv_img(s_[-1])):    break_flag = 1; break


                position = position_


            def memory(s, action, r, info, s_):
                pass
            memory(s, action, r, info, s_)

            s = s_

            # if (arg.debug_reward_d_p):
            #     print('{:5.2},   {:5.3f},   {:5},'.format(rewards[-1], env.rewards_d[-1], env.rewards_p[-1]))
                # print('{:5.2f},   {:5.3f},   {:5.3f},'.format( sum(rewards), sum(env.rewards_d), sum(env.rewards_p) )  )

            if (done or step >= MAX_STEP - 1):
                # print('------------------------ done')


                step += 1
                f = round(t1.now() / step, 3)
                step_per_second = round(1 / f, 3)

                if (arg.debug_reward_d_p_all):
                    print('本回合奖励:{:5.2f},   r_d: {:5.3f},   r_p: {:5.3f},'.format(sum(rewards) - arg.reward_done,
                                                                                  sum(env.rewards_d),
                                                                                  sum(env.rewards_p)))

                # ----------  importrant data ----------------------
                print('episode: {:4d}, reward: {:5.2f}---(r - done): {:5.2f},   r_d: {:5.3f},   r_p: {:5.3f}, fresh_speed: {}'.format(i_episode,
                                                                                                           round(sum(rewards), 3),
                                                                                                           ( sum( rewards) - arg.reward_done ),
                                                                                                           sum(env.rewards_d), sum(env.rewards_p),
                                                                                                                              f,
                                                                                                                        ) )
                env.reset()
                steps += step
                break
    # print(1)
    ##############################################################
    f = round(tt.now() / steps, 3)
    step_per_second = round(1 / f, 3)
    print('##############################################################')
    print(i_episode,' episodes--steps: ', steps, 'cost_time: ',tt.now(), 'frequence: ', f, '-- step_per_second: ', step_per_second)
    1