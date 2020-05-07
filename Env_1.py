from wind import Window
from mycode.my_time import Time, vk
import numpy as np
from template import match_img
import cv2

from grabscreen import grab_screen, plt_img, cv_img
from process_image import pre_process_screen
from scipy.spatial.distance import pdist
from args import arg
from utils import dist_score

class path:
    pic_path = 'picture\\'
    my_underwear = pic_path + 'my_underwear.png'
    destination = pic_path + 'destination.png'
    screen0 = pic_path + 'screen0.png'
    human = pic_path + 'human.png'

class Info:
    position = (0,0)
    destination = (-1,-1)
info = Info()
tt = Time()


class Env():
    def __init__(self):
        class_name = 'YYGameMakerYY'
        wind = Window(class_name)
        self.wind = wind
        self.num_frames = 2

        # self.action_name = ['left_press', 'left_up', 'right_press', 'right_up', 'shift_press', 'shift_up', 'stop']

        self.action_name = ['left', 'right', 'shift_press', 'shift_up', 'stop']
        self.action_space = list(range(len(self.action_name)))

        self.screen_0 = cv2.imread(path.screen0)
        self.screen_0_gray = cv2.cvtColor(self.screen_0, cv2.COLOR_RGB2GRAY)

        self.my_underwear = cv2.imread(path.my_underwear)
        self.my_underwear_gray = cv2.cvtColor(self.my_underwear, cv2.COLOR_RGB2GRAY)
        self.my_underwear_bgr = cv2.cvtColor(self.my_underwear, cv2.COLOR_RGB2BGR)

        self.destination = cv2.imread(path.destination)
        self.destination_gray = cv2.cvtColor(self.destination, cv2.COLOR_RGB2GRAY)

        self.human = cv2.imread(path.human)
        self.human_bgr = cv2.cvtColor(self.human, cv2.COLOR_RGB2BGR)


        self.info = Info()   # reserve informations
        self.info.destination = self.get_destinations()

        # --------------------------------------------------------------------------------
        self.end = (750, 460)
        self.start = (0, 0)

        self.s, self.position = 0, -1

        self.steps = 0
        # 记录每回合经过的坐标, 奖励, 与终点的距离.
        self.positions, self.rewards, self.distance_s = [], [], []

        self.distance_list_max_l = [self.start, self.end]  # 最长路径
        self.distance_max = pdist(self.distance_list_max_l)[0]
        self.ds_max = 0
        self.ds_min = self.distance_max

        # self.rewards_p = []
        self.rewards_d = []


        #screen0 = self.reset()


        # my_underwear = cv2.imread(path.my_underwear)
        # destination = cv2.imread(path.destination)
        # plt_img(my_underwear)

        pass

    def reset(self, sleep_t = arg.reset_sleep, ret_screen = 0):
        self.control(-1)
        self.wind.key_dp(vk.r)
        tt.sleep(sleep_t)

        self.rewards_p, self.rewards_d = [], []
        self.ds_min = self.distance_max

        if(ret_screen):
            screen = self.wind.grab_screen()
            return screen
        else:
            return 1

    def render(self):

        pass

    def step(self, action):

        s_ = self.get_state(action)
        position = self.get_position(s, template=env.my_underwear_bgr)
        s_ = pre_process_screen(s_)
        r, done, info = self.get_feedback(s, s_)


        # r, s_,info = 0,0,[position]
        r = 0
        info = [self.position]


        self.s = s_
        self.steps += 1
        return s_, r, done, info

    def control(self, action):
        action_name = self.action_name
        wind = self.wind

        if (action_name[action] == 'stop'):
            wind.key_up(vk.left)
            wind.key_up(vk.right)
            wind.key_up(vk.shift)

        elif (action_name[action] == 'left'):
            wind.key_up(vk.right)
            wind.key_press(vk.left)
        elif (action_name[action] == 'right'):
            wind.key_up(vk.left)
            wind.key_press(vk.right)

        elif (action_name[action] == 'shift_press'):
            wind.key_press(vk.shift)
        elif (action_name[action] == 'shift_up'):
            wind.key_up(vk.shift)

        return 1

    # 获得终点坐标
    def  get_destinations(self):
        target = self.screen_0
        template = self.destination


        pass

    def get_state(self, action):  # get state to frames_bind
        env = self
        nums = env.num_frames
        wind = env.wind

        screens = []
        for i in range(nums):
            env.control(action)
            screen0 = wind.grab_screen()

            screens.append(screen0)
        return screens

    # get pre_process_state
    def get_pre_state(self, screens = None):
        if(screens == None):
            self.get_state()
        screens_0 = []
        for screen in screens:
            screen = pre_process_screen(screen)
            screens_0.append(screen)
        return screens_0

    def get_feedback(self, s, s_):
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
            info = None
            return r, done, info

        # ------- reward
        distance_list = [position_, env.end]
        distance = pdist(distance_list)[0]

        # distance_list_max_l = [env.start, env.end]  # 最长路径
        # distance_max = pdist(distance_list_max_l)[0]
        # ds_max = max(distance_s)    # 目前的最近距离

        # ds_max = env.ds_max
        ds_min = env.ds_min
        if (distance < ds_min):
            # r_d = 0.01
            r_d = dist_score(distance, env.distance_max) * arg.r_d_positive  # 当前位置和目标的距离越小, 奖励越大, 分布[ -NAN, 1]
            env.ds_min = distance
        else:
            r_d = arg.r_d_negtive

        # distance_s.append(distance)

        # env.positions.insert(0, [111,222])
        # env.positions.pop()
        # env.positions.append(position_)
        #
        # env.positions
        # position_

        def is_inter(a, A):  # a in A
            inter = False
            for i in range(len(A)):
                equal = (a == env.positions[i]).all()
                if (equal):
                    # print(i, equal)
                    inter = True
                    break
            return inter

        # 判断 position_ 是否在 env.positions 内
        inter = is_inter(position_, env.positions)

        if (inter):
            r_p = arg.reward_old_position  # 探索到未知情况时给予奖励
            pos_passed = True  # 是否曾经来过此处
        else:
            r_p = arg.reward_new_position  # reward for explore:  new position_
            pos_passed = False
            env.positions.append(position_)

        # 奖励 ← 目标距离 + 探索轨迹

        r = r_d + r_p
        env.rewards.append(r);
        env.rewards_d.append(r_d);
        env.rewards_p.append(r_p)
        if (r_d > 0 and arg.control_prt_feedback):
            print('r: {:4.2}, r_d: {:5.3}/{:5.3f}, r_p: {:5.2}/{:5.2f}, steps: {:-5}'.format(r, r_d, sum(env.rewards_d),
                                                                                             r_p, sum(env.rewards_p),
                                                                                             env.steps))

        info = [position_, pos_passed]
        env.steps += 1
        return r, done, info

    # 获得人物坐标
    def get_position(self, targets, template, min_threshold=0.5, multip_res=1):

        target = targets[-1]  # targets.dtype == BGR

        results = match_img(target, template, min_threshold=0.5, multip_res=1)

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
            results = match_img(target, template, min_threshold=0.4, multip_res=1)
            if (len(results)):
                results = results[0]
            else:
                # print('--- confirm death! ---')
                return -1
        results = results[0]
        xy0, xy1 = results[0], results[1]
        position = np.array((np.array(xy0) + np.array(xy1)) / 2, dtype=int)
        return position


    def close(self):
        pass

    1


# debug
if(0):
    tt = Time()
    class_name = 'YYGameMakerYY'
    wind = Window(class_name)
    wind.move_to(0, 0)

if(0):
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
    env = Env()

    env.action_name
    env.num_frames = 3

    EPISODE = 3
    MAX_STEP = 100
    tt = Time()
    for i_episode in range(EPISODE):
        env.reset()
        rewards = []

        s = cv2.cvtColor(env.wind.grab_screen(), cv2.COLOR_BGR2GRAY)
        if(env.num_frames == 2):
            s = np.array([s,s])
        elif(env.num_frames == 3):
            s = np.array([s,s,s])
        s.shape

        for step in range(MAX_STEP):
            if(len(s) != env.num_frames):   print('error len!')
            # tt.sleep(0.1)
            # action = model(s)
            action = np.random.choice(env.action_space)

            s_, r, done, info = env.step(action)
            rewards.append(r)
            s_.shape

            s = s_
            if( done or step >= MAX_STEP -1):
                r = -10
                rewards.append(r)
                env.reset()

                print("episode:", i_episode, "  reward:", sum(rewards), 'steps: ', step)
                break

            # r, s_, done, info =  env.step(action)
            # transition(s, a, r, s_)

            # print(' ', i,'----', r, s_, done, info )
    print(tt.now())

200 /23

1/8