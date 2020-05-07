import cv2
from grabscreen import grab_screen, plt_img, cv_img
from args import arg
from process_image import pre_process_screen
from mycode.my_time import Time, vk
import torch

def dist_score(x, x_max):
    # dist 对应的 r
    dd = (x_max - x) / x_max
    r_d = 1 - dd
    if (x < x_max):
        return r_d
    else:
        return r_d

def add_rects(img, point, env):
    img_add_rect(img=img, point=env.start, ptype=arg.ptype_source, pcolor=arg.pcolor_source, ww=20, hh=20)
    img_add_rect(img=img, point=env.end, ptype=arg.ptype_dest, pcolor=arg.pcolor_dest, ww=20, hh=20)
    img_add_rect(img=img, point=point, ptype=arg.ptype, pcolor=arg.pcolor, ww=10, hh=20, cut=1)
    return img

def img_add_rect(img, point, ptype=2, pcolor=(0, 255, 0), ww=10, hh=20, cut = 0):
    # ww = 10
    # hh = 20
    xy0 = point[0] - ww, point[1] - hh
    if(cut):
        xy1 = point[0] + ww, point[1] + hh // 3
    else:
        xy1 = point[0] + ww, point[1] + hh
    xy0, xy1
    # plt_img(img)
    cv2.rectangle(img, xy0, xy1, pcolor, ptype)

def preprocess_state(s_, position_, env, resize = arg.resize):
    if(not arg.preprocess_state):
        s_ = np.array(s_)
        return s_

    for i in range(len(s_)):
        s_[i] = pre_process_screen(s_[i])

    prep_s = []
    for i in range(arg.num_frames):
        img = s_[i]
        if (i == arg.num_frames - 1 and arg.resize):
                add_rects(img=img, point=position_, env=env)
        img = cv2.resize(img, arg.wind_conv_wh, interpolation=cv2.INTER_AREA)
        # img = cv2.resize(img, (arg.wind_w, arg.wind_h), interpolation=cv2.INTER_AREA)
        prep_s.append(img)

    # if (arg.resize):
    #     for i in range(arg.num_frames):
    #         img = s_[i]
    #         if(i == arg.num_frames -1 ):
    #             add_rects(img=img, point=position_, env=env)
    #         img = cv2.resize(img, arg.wind_conv_wh, interpolation=cv2.INTER_AREA)
    #         # img = cv2.resize(img, (arg.wind_w, arg.wind_h), interpolation=cv2.INTER_AREA)
    #         prep_s.append(img)
    #         # print(len(s_), env.num_frames)
    # else:
    #     img = s_[-1]
    #     add_rects(img=img, point=position_, env=env)
    #     prep_s = s_
    prep_s = np.array(prep_s)
    return prep_s           # s_

import numpy as np
def list_array(L):
    # return a list!

    for i in range(len(L)):
        arr = L[i]
        if(i == 0):
            arr_s = np.array(arr)
        else:
            arr_s = np.vstack((arr_s,arr))
    return arr_s


def list_tensor(L, tensor_type = 'float'):
    # return a list!
    L = list_array(L)
    if(tensor_type == 'float'):
        arr_s = torch.FloatTensor(L)
    elif(tensor_type == 'long'):
        arr_s = torch.LongTensor(L)
    if (tensor_type == 'np'):
        arr_s = torch.from_numpy(L)

    return arr_s

aa = [[1,2,3],[4,5,6], [7,8,9]]
list_tensor(aa)


# L = [[1,2],[3,4],[5,6]]
# list_array(L)

def process_x(s, device = 'cpu'):
    if(isinstance(s, tuple)):
        ss = np.array(s)
    ss = torch.FloatTensor(s)
    ss.shape
    # ss = resize(ss)
    # ss.shape
    if(len(ss.shape) == 3):
        ss = ss.unsqueeze(0)
    ss.shape
    if(device == 'cuda'):
        ss = ss.to('cuda')
    return ss

def norm_0_1(x, x_min, x_max):
    dd = (x - x_min) / (x_max - x_min)
    if (x > x_max):
        print('error: x > x_max!')
    return dd
#
# for i in range(11):
#     print(i)
#     i+=1
    # if(i == 2):
    #     i -= 1

###############
# from utils import img_add_rect