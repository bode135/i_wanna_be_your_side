import gym
import keyboard
import numpy as np
import time
from mycode.damo import Time


#from gym.utils import play
#play.play(gym.make('Breakout-v0'),fps = 10 ,zoom=3)

total_reward = 0
import retro

# import game
if(1):
    games = ['CartPole-v0', 'MountainCar-v0', 'Pendulum-v0', 'SpaceInvaders-v0', 'Airstriker-Genesis',
             'SuperMarioBros-v0']
    i_game = 2
    game = games[i_game]

    if (game == 'SuperMarioBros-v0'):
        from nes_py.wrappers import JoypadSpace
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        # get the path of ROM
        if (0):
            from gym_super_mario_bros._roms.rom_path import rom_path

            rp = rom_path(True, 'vanilla')
            print(rp)
            import os

            ROM_PATH = rp.rsplit('\\', 1)[0]
            os.system('explorer ' + ROM_PATH)
    else:
        try:
            env = gym.make(game).unwrapped
            print('Enviroment: gym')
        except:
            env = retro.make(game=game)
            print('Enviroment: retro')

    1
print('Game name:',game)

#env.close()
if(game == 'Pendulum-v0'):      # continue action
    n_action = 2
else:
    n_action = env.action_space.n
if(0):
    env.action_space.sample()
    env.action_space
    if(1):
        state = env.reset()
        env.render()
        #time.sleep(1)

        action = env.action_space.sample()
        #action = [1]
        print(action)
        next_state,reward,done,_ = env.step(action)
        env.render()



    if(0):
        print(next_state)
        reward
        done
        _

# --- init ---
state = env.reset()
env.render()
time.sleep(1)

# parameter
action = 0;reset = 0
total_reward = 0;over_round = 0

if(game == 'Pendulum-v0'):      # 连续动作很麻烦
    action = [0]

# action key
def abc(x):
    global action,reset

    if x.event_type == "down" and x.name == 'r':
        reset = 1

    # define the control-key of every game
    if(game == 'Airstriker-Genesis'):
        if x.event_type == "down" and x.name == 'a':
            action = 0
        elif x.event_type == "down" and x.name == 's':
            action = 1
        elif x.event_type == "down" and x.name == 'd':
            action = 2
        elif x.event_type == "down" and x.name == 'q':
            action = 3
        elif x.event_type == "down" and x.name == 'w':
            action = 4
        elif x.event_type == "down" and x.name == 'e':
            action = 5
    elif(game == 'MountainCar-v0'):

        if x.event_type == "down" and x.name == '1':
            action = 0
        elif x.event_type == "down" and x.name == '3':
            action = 1
    elif(game == 'CartPole-v0'):
        if x.event_type == "down" and x.name == '1':
            action = 0
        elif x.event_type == "down" and x.name == '3':
            action = 1
    elif(game == 'SuperMarioBros-v0'):
        #actions = list(range(n_action))
        # stop, right, jump_right , right, right_jump, jump, left
        #    0,     1,           2,    3,        4,    5,     6
        actions = {
            'stop':0,
            'right_1':1,
            'jump_right': 2,
            'right_3': 3,
            'right_jump': 4,
            'jump': 5,
            'left': 6,
                   }

        actions['left']
        actions.items()

        #actions = list(range(n_action))
        n_action
        if(x.event_type == 'down'):
            try:
                i_action = int(x.name)
                if (i_action >= 0 or i_action <= n_action - 1):
                   # action = actions[i_action]
                    if(i_action == 1):            action = actions['left']
                    elif(i_action == 3):         action = actions['right_3']
                    elif(i_action == 5):         action = actions['jump']
                    elif(i_action == 2):          action = actions['stop']
                else:
                    pass

            except:

                pass
    elif(game == 'Pendulum-v0'):
        if x.event_type == "down" and x.name == 'a':
            action = [-0.5]
        elif x.event_type == "down" and x.name == 'd':
            action = [0.5]


# 添加hook，以检测用户的按键
keyboard.hook(abc)
tt = Time()
tt.break_flag
for j in range(10000):
    env.render()

    if(tt.break_flag):
        tt.break_flag = 0
        env.close()
        break

    tt.stop_0('s')  # 监听stop按键，默认为P键暂停,以及用于强制退出游戏


    if(game != 'Pendulum-v0'):
        action %= n_action      # env.action_space.n len
    print(action)
    next_state,reward,done,_ = env.step(action)
    if(0):
        print(next_state)
        reward
        done
        _

    #action = 0
    total_reward += reward

    # refresh time
    time.sleep(0.06)

    if (reset == 1):
        done = 1

    if done:
        print('The{:>3d} th round is over, You are loss! --- Reward:{:>4d}'.format(over_round, int(total_reward)))
        time.sleep(0.5)
        #break
        env.reset()
        env.render()
        reset = 0
        total_reward = 0
        action = 0
        time.sleep(0.2)
        over_round += 1

print('---')


if(0):
    env.close()
#keyboard.wait()

#sssssssa

#pyinstaller C:\Users\Administrator\Desktop\torch_policity\mycode\键盘玩gym.py

#C:\Users\Administrator\Desktop\torch_policity\mycode\键盘玩gym.py