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
from enum_wind import get_class_winds

class path:
    pic_path = 'picture\\'
    my_underwear = pic_path + 'my_underwear.png'
    destination = pic_path + 'destination.png'
    screen0 = pic_path + 'screen0.png'
    human = pic_path + 'human.png'


class Info:
    position = (0, 0)
    destination = (-1, -1)


# info = Info()
tt = Time()


class Env():
    def __init__(self,
                 i_multiply = 0 ,
                 class_name = 'YYGameMakerYY' ,
                 ):

        if(i_multiply == -1):
            wind = Window(class_name)
        else:

            hwnd_s = get_class_winds(class_name)
            hwnd = hwnd_s[i_multiply]
            wind = Window(class_name, hwnd = hwnd)



        self.wind = wind
        self.num_frames = arg.num_frames

        self.press_shift = False

        # self.action_name = ['left_press', 'left_up', 'right_press', 'right_up', 'shift_press', 'shift_up', 'stop']

        # self.action_name = ['left', 'right', 'shift_press', 'shift_up', 'stop']
        # self.N_A = len(self.action_name)
        # self.action_space = list(range(self.N_A))
        self.action_name = arg.action_name
        self.N_A = arg.N_A
        self.action_space = arg.action_space


        self.screen_0 = cv2.imread(path.screen0)
        self.screen_0_gray = cv2.cvtColor(self.screen_0, cv2.COLOR_RGB2GRAY)

        self.my_underwear = cv2.imread(path.my_underwear)
        self.my_underwear_gray = cv2.cvtColor(self.my_underwear, cv2.COLOR_RGB2GRAY)
        self.my_underwear_bgr = cv2.cvtColor(self.my_underwear, cv2.COLOR_RGB2BGR)

        self.destination = cv2.imread(path.destination)
        self.destination_gray = cv2.cvtColor(self.destination, cv2.COLOR_RGB2GRAY)

        self.human = cv2.imread(path.human)
        self.human_bgr = cv2.cvtColor(self.human, cv2.COLOR_RGB2BGR)

        self.info = Info()  # reserve informations
        self.info.destination = self.get_destinations()

        # --------------------------------------------------------------------------------
        self.end = arg.end
        self.start = arg.start

        self.s, self.position = 0, -1

        self.steps = 0
        # 记录每回合经过的坐标, 奖励, 与终点的距离.
        self.positions, self.rewards, self.distance_s = [], [], []

        self.distance_list_max_l = [self.start, self.end]  # 最长路径
        self.distance_max = pdist(self.distance_list_max_l)[0]
        # print('----------------distance_max: ', self.distance_max)
        self.ds_max = 0
        self.ds_min = self.distance_max


        self.rewards_p = []
        self.rewards_d = []

        # screen0 = self.reset()

        # my_underwear = cv2.imread(path.my_underwear)
        # destination = cv2.imread(path.destination)
        # plt_img(my_underwear)

        pass

    def reset(self, sleep_t=arg.reset_sleep, return_s_pos = 0):
        self.control(-1)
        self.wind.key_dp(vk.r)
        tt.sleep(sleep_t)

        self.rewards_p, self.rewards_d ,self.rewards = [], [], []

        self.ds_min = self.distance_max

        self.positions = []

        if (return_s_pos):
            # screen = self.wind.grab_screen()
            env = self

            s = self.get_state()
            # position = self.get_position(s, template=self.my_underwear_bgr)
            position = arg.start

            # pos_passed = 0
            # info = [position, pos_passed]
            # return s, info

            if (isinstance(position, int)):
                print('--------- Don\'t find position! Please don\'t complete cover the window! ----------------')
                # print('--- ', i_episode,' ---------未找到游戏! 后台模式下勿完全遮挡窗口!-----------')
                # print('------------- ', i_episode, ' ------------')

                # plt_img(s[-1])  # debug position!
                env.wind.ShowWindow()

                if (not np.equal((0, 0, 816, 647), env.wind.get_rect()).all()):
                    env.wind.move_to(0, 0)

                # tt.sleep(0.01)
                s, position = env.reset(sleep_t= 0.02, return_s_pos = 1)
                # continue

                # add points

            from utils import add_rects
            add_rects(img=s[-1], point=position, env=env)


            return s, position

        else:
            return None

    def step(self, action):

        s_ = self.get_state(action)
        s_[-1].dtype
        r, done, info = self.get_feedback(s_)

        if(done):
            nums = arg.num_frames

            ss = np.zeros([arg.wind_h, arg.wind_w, 3], dtype = 'uint8')
            # pre_process_screen(ss)
            # plt_img(ss)
            screens = []
            for i in range(nums):
                screens.append(ss)
            s_ = screens

        self.steps += 1
        return s_, r, done, info

    def control(self, action = None):
        action_name = self.action_name
        wind = self.wind

        # arg.action_name ['left', 'right', 'shift_press', 'shift_up', 'stop']
        if(action == None):
            pass
        elif(action == arg.action_space[-1]):       # stop
            wind.key_up(vk.left)
            wind.key_up(vk.right)
            wind.key_up(vk.shift)
            self.press_shift = False
        elif (action == 0):
            wind.key_up(vk.right)
            if (self.press_shift == True):
                wind.key_up(vk.shift)

            wind.key_press(vk.left)
        elif (action == 1):
            wind.key_up(vk.left)
            if (self.press_shift == True):
                wind.key_up(vk.shift)

            wind.key_press(vk.right)
        # wind.key_dp(vk.r); wind.key_up(vk.right); wind.key_up(vk.shift)
        elif (action == 2):             # shift_right ---> high
            wind.key_up(vk.left)
            wind.key_up(vk.right)
            if (self.press_shift == True):
                wind.key_up(vk.shift)

            wind.key_press(vk.shift)
            tt.sleep(0.01)
            wind.key_press(vk.right)

        elif (action == 3):             # shift_right ---> low
            wind.key_up(vk.left)
            if (self.press_shift == True):
                wind.key_up(vk.shift)
                # tt.sleep(0.011)
            wind.key_press(vk.shift)
            tt.sleep(0.01)
            wind.key_up(vk.shift)

            wind.key_press(vk.right)

        elif (action == 4):  # shift_left ---> high
            wind.key_up(vk.right)
            if (self.press_shift == True):
                wind.key_up(vk.shift)
                # tt.sleep(0.011)
            wind.key_press(vk.shift)
            tt.sleep(0.01)
            wind.key_press(vk.left)
        elif (action == 5):  # shift_left ---> low
            wind.key_up(vk.right)
            if (self.press_shift == True):
                wind.key_up(vk.shift)
                # tt.sleep(0.011)
            wind.key_press(vk.shift)
            tt.sleep(0.01)
            wind.key_up(vk.shift)

            wind.key_press(vk.left)


        #
        # if (action_name[action] == 'stop'):
        #     wind.key_up(vk.left)
        #     wind.key_up(vk.right)
        #     wind.key_up(vk.shift)
        #     self.press_shift = False
        #
        # elif (action_name[action] == 'left'):
        #     wind.key_up(vk.right)
        #     wind.key_press(vk.left)
        # elif (action_name[action] == 'right'):
        #     wind.key_up(vk.left)
        #     wind.key_press(vk.right)
        #
        # elif (action_name[action] == 'shift_press'):
        #     if(self.press_shift == True):
        #         wind.key_up(vk.shift)
        #         tt.sleep(0.011)
        #     wind.key_press(vk.shift)
        #     self.press_shift = True
        # elif (action_name[action] == 'shift_up'):
        #     wind.key_up(vk.shift)
        #     self.press_shift = False

        return 1

    # 获得终点坐标
    def get_destinations(self):
        target = self.screen_0
        template = self.destination

        pass

    # 获取当前环境的状态
    def get_state(self, action = -1):  # get state to frames_bind
        env = self
        nums = env.num_frames
        wind = env.wind

        env.control(action)
        screens = []
        for i in range(nums):
            screen0 = wind.grab_screen()

            screens.append(screen0)
        # print('l ',len(screens))
        # print(env.num_frames)
        return screens

    # get pre_process_state
    def get_pre_state(self, screens=None):
        if (screens == None):
            self.get_state()
        screens_0 = []
        for screen in screens:
            screen = pre_process_screen(screen)
            screens_0.append(screen)
        return screens_0

    # 获取环境反馈
    def get_feedback(self, s_ ):
        # return 1, False, [(0,0), 1]

        env = self
        get_position = self.get_position

        done = False
        position_ = get_position(s_, template=env.my_underwear_bgr)

        # ------ is_done ?
        if (isinstance(position_, int)):
            done = True
        elif (position_.shape == ()):
            if (arg.debug):       print('warning --- len_position_:', position_.shape)
            done = True

        if (done):
            r = arg.reward_done
            position_ = (0, 0)

            info = [position_, -1, -1]
            return r, done, info

        # ------- reward
        distance_list = [position_, env.end]            # 目前位置和目标的距离
        distance = pdist(distance_list)[0]

        rate = (env.distance_max - distance) / env.distance_max  # 已经走了 rate% 的路程

        if (distance < env.ds_min):
            r_d = arg.r_d_positive
            env.ds_min = distance
        else:
            r_d = 0


        # r_d = rate * arg.r_d_positive

        # if (distance < ds_min):
        #     # dd_np = np.round((distance, ds_min, 100 * (distance - ds_min), 100 * (distance - ds_min) / ds_min),2)
        #     # print(dd_np)
        #
        #     r_d = rate * arg.r_d_positive
        #
        #     # r_d = arg.r_d_positive
        #     # r_d = dist_score(distance, env.distance_max) * arg.r_d_positive  # 当前位置和目标的距离越小, 奖励越大, 分布[ -NAN, 1]
        #     env.ds_min = distance
        #     # not_penalty_rp = True
        # else:
        #     r_d = rate * arg.r_d_negtive
        #     # not_penalty_rp = False
        #     # r_d = rate * arg.r_d_negtive
        #
        # r_d = rate**2 * arg.r_d_positive

        if(arg.print_r_d): print('------- r_d: ',round(r_d, 3), round(rate, 2), distance < env.distance_max, round(distance, 1), round(env.distance_max, 1) )

        # 判断 position_ 是否在 env.positions 内
        position_
        # ([119, 565]) --> [ 59, 282]
        # np.array(position_ // 2, dtype='int32')
        # p = np.array([117, 565])
        # np.array(p // 2, dtype='int32')
        # arg.inter_size = 2
        positions = np.array(position_ // arg.inter_size, dtype= 'int32')
        inter = self.is_inter(positions, env.positions)

        if (inter):     # 探索到未知情况时给予奖励, 停留则给予惩罚
            # if(r_d > 0):
            #     r_d = 0     # 老地方不给予距离奖励
            r_p = arg.reward_old_position
            pos_passed = True  # 是否曾经来过此处
        else:
            r_p = arg.reward_new_position  # reward for explore:  new position_
            pos_passed = False
            env.positions.append(positions)

        # 奖励 ← 目标距离 + 探索轨迹
        r = r_d + r_p

        env.rewards.append(r)
        env.rewards_d.append(r_d)
        env.rewards_p.append(r_p)

        if (r_d > 0 and arg.control_prt_feedback):  # debug
            print('r: {:5.3f}, r_d: {:5.3f}, sum_rd: {:8.3} -- steps: {:5}'.format(r,r_d, sum(env.rewards_d), env.steps))

        info = [position_, self.press_shift, pos_passed]
        env.steps += 1
        return r, done, info

    # 获得人物坐标
    def get_position(self, targets, template, min_threshold=0.2, multip_res=1):

        target = targets[-1]  # targets.dtype == BGR

        results = match_img(target, template, min_threshold=min_threshold, multip_res=1)

        # results = []
        if (len(results)):
            result = results
        else:
            # 临时丢失目标, 还是已经死亡? 再匹配一次..
            # tt.sleep(0.001)
            # env.control(action)
            # target = env.wind.grab_screen()
            try:
                target = targets[-2]
            except:
                # tt.sleep(0.01)
                pass
            results = match_img(target, template, min_threshold= min_threshold, multip_res=1)
            if (len(results)):
                results = results[0]
            else:
                # print('--- confirm death! ---')
                return -1
        results = results[0]
        xy0, xy1 = results[0], results[1]
        position = np.array((np.array(xy0) + np.array(xy1)) / 2, dtype=int)
        return position

    # 该坐标是否曾经停留过
    def is_inter(self, a, A):  # a in A
        inter = False
        for i in range(len(A)):
            equal = (a == self.positions[i]).all()
            if (equal):
                # print(i, equal)
                inter = True
                break
        return inter

    def render(self):

        pass


    def close(self):
        pass

    1


# debug
if (0):
    tt = Time()
    class_name = 'YYGameMakerYY'
    wind = Window(class_name)
    wind.move_to(0, 0)

if (0):
    def control(action):
        # _action = 2
        # action = 0
        action_name = ['left', 'right', 'shift', 'left_shift', 'right_shift', 'stop']
        action_space = list(range(len(action_name)))

        tmp = 0.01
        jump_t = 0.1
        jump_T = 0.4

        if (action_name[action] == 'stop'):
            # wind.key_press(vk.left)
            # wind.key_press(vk.left)
            wind.key_up(vk.left)
            wind.key_up(vk.right)
            wind.key_up(vk.shift)

            _action = 'stop'
        if (action_name[action] == 'left'):
            # wind.key_press(vk.left)

            wind.key_press(vk.left)
            # wind.key_up(vk.left)
            wind.key_up(vk.right)
            wind.key_up(vk.shift)

            _action = 'left'

        elif (action_name[action] == 'right'):
            # wind.key_dp(vk.right, tmp)
            wind.key_press(vk.right)

            wind.key_up(vk.left)
            wind.key_up(vk.shift)
            _action = 'right'

        elif (action_name[action] == 'shift'):
            # wind.key_up(vk.left)
            # wind.key_up(vk.right)

            # wind.key_dp(vk.shift, jump_t)
            # wind.key_dp(vk.shift, jump_T)
            # wind.key_dp(vk.shift, 0.4)
            tt.sleep(0.5)
            wind.key_up(vk.shift)
            wind.key_press(vk.shift)

            _action = 'shift'

        elif (action_name[action] == 'left_shift'):
            wind.key_up(vk.right)

            wind.key_press(vk.left)
            wind.key_dp(vk.shift, tmp)
            _action = 'left_shift'

        elif (action_name[action] == 'right_shift'):
            wind.key_up(vk.left)

            wind.key_press(vk.right)
            wind.key_press(vk.shift)
            wind.key_dp(vk.shift, tmp)
            _action = 'right_shift'

        1


    def control(action):
        # _action = 2
        # action = 0
        'stop'
        action_name = ['left', 'right', 'shift_press', 'shift_up', 'stop']
        action_name = ['left', 'right', 'shift_press', 'shift_up', 'left_shift_press', 'left_shift_up',
                       'right_shift_press', 'right_shift_up', 'stop']

        action_space = list(range(len(action_name)))

        tmp = 0.01
        jump_t = 0.1
        jump_T = 0.4

        if (action_name[action] == 'stop'):
            # wind.key_press(vk.left)
            # wind.key_press(vk.left)
            wind.key_up(vk.left)
            wind.key_up(vk.right)
            wind.key_up(vk.shift)

            _action = 'stop'

        if (action_name[action] == 'left'):
            # wind.key_press(vk.left)

            wind.key_press(vk.left)
            # wind.key_up(vk.left)
            wind.key_up(vk.right)
            wind.key_up(vk.shift)

            _action = 'left'
        elif (action_name[action] == 'right'):
            # wind.key_dp(vk.right, tmp)
            wind.key_press(vk.right)

            wind.key_up(vk.left)
            wind.key_up(vk.shift)
            _action = 'right'
        elif (action_name[action] == 'shift_press'):
            wind.key_press(vk.shift)
            _action = 'shift_press'
        elif (action_name[action] == 'shift_up'):
            wind.key_up(vk.shift)
            _action = 'shift_up'

        elif (action_name[action] == 'left_shift_press'):
            wind.key_up(vk.right)

            wind.key_press(vk.left)
            wind.key_press(vk.shift)
            _action = 'left_shift_press'
        elif (action_name[action] == 'left_shift_up'):
            wind.key_up(vk.right)

            wind.key_press(vk.left)
            wind.key_up(vk.shift)
            _action = 'left_shift_up'

        elif (action_name[action] == 'right_shift_press'):
            wind.key_up(vk.left)

            wind.key_press(vk.right)
            wind.key_press(vk.shift)
            _action = 'right_shift_press'

        elif (action_name[action] == 'right_shift_up'):
            wind.key_up(vk.left)

            wind.key_press(vk.right)
            wind.key_up(vk.shift)
            _action = 'right_shift_up'

        1


    1

if __name__ == '__main__':
    class_name = 'class_name'
    Env(i_multiply= 0 )

    env = Env()

    env.action_name
    env.num_frames = arg.num_frames

    EPISODE = arg.MAX_EPISODE
    MAX_STEP = arg.MAX_STEP


    tt = Time(); break_flag = 0; steps = 0
    for i_episode in range(EPISODE):
        if (break_flag):
            cv2.destroyAllWindows()
            break


        t1 = Time(); rewards = []
        s, position = env.reset(sleep_t= arg.reset_sleep,return_s_pos= 1)

        if ( arg.debug_position_error and isinstance(position, int) ):  print('------------ position_error!')

        for step in range(MAX_STEP):
            if (tt.stop_alt('s')):
                print('----- break! -----')
                break_flag = 1
                break

            if (len(s) != env.num_frames):   print('error len!')
            ############################################################
            action = np.random.choice(env.action_space)  # action = model(s)
            s_ = env.get_state(action)

            r, done, info = env.get_feedback(s_)
            ################### -------------------
            if (not done):  # plot cv_img

                position_ = info[0]
                img = s_[-1]

                # point = position_

                if (arg.control_preprocess_img):
                    img = pre_process_screen(img)
                # if (cv_img(img)):    break_flag = 1; break
                img_add_rect(img=img, point=position_, ptype=arg.ptype, pcolor=arg.pcolor, ww=10, hh=20, cut=1)
                img_add_rect(img=img, point=env.end, ptype=arg.ptype_dest, pcolor=arg.pcolor_dest, ww=20, hh=20)
                # if (cv_img(img)):    break_flag = 1; break
                # plt_img(img)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img.shape
                img.dtype
                # img = cv2.resize(img, (600, 400), interpolation=cv2.INTER_AREA)
                # img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)

                # cv2.rectangle(img, xy0, xy1, (0, 0, 0), -1)

                if (arg.show_pre_image and cv_img(img)):    break_flag = 1; break

                position = position_
                rewards.append(r)

            s = s_

            # if (arg.debug_reward_d_p):
                # print('{:5.2},   {:5.3f},   {:5},'.format(rewards[-1], env.rewards_d[-1], env.rewards_p[-1]))
                # print('{:5.2f},   {:5.3f},   {:5.3f},'.format( sum(rewards), sum(env.rewards_d), sum(env.rewards_p) )  )

            if (done or step >= MAX_STEP - 1):
                r = arg.reward_done
                rewards.append(r)

                step += 1
                f = round(t1.now() / step, 3)
                step_per_second = round(1 / f, 3)

                if (arg.debug_reward_d_p_all):
                    print('本回合奖励:{:5.2f},   r_d: {:5.3f},   r_p: {:5.3f},'.format(sum(rewards) - arg.reward_done,
                                                                                  sum(env.rewards_d),
                                                                                  sum(env.rewards_p)))
                # print("episode:", i_episode, "  reward:", round(sum(rewards), 3), 'steps: ', step, 'cost_time: ',
                #       t1.now(), 'frequence: ', f, '-- step_per_second: ', step_per_second)

                print('episode: {}, reward: {}, r - done: {:5.2f},   r_d: {:5.3f},   r_p: {:5.3f},'.format(i_episode, sum(rewards), sum(rewards) - arg.reward_done, sum(env.rewards_d),sum(env.rewards_p)))
                env.reset()
                steps += step
                break

    f = round(t1.now() / step, 3)
    step_per_second = round(1 / f, 3)
    print(i_episode,' times--steps: ', steps, 'cost_time: ',tt.now(), 'frequence: ', f, '-- step_per_second: ', step_per_second)

