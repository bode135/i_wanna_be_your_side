###################################################
# arg.end
# from args import arg
###################################################

# info = [position_, self.press_shift, pos_passed]
class arg():
    num_frames = 1  # channel! 默认灰度图, RGB会出错,还没改...

    Reload_net = False       # True, False
    Reload_params = True

    Save_net = False
    Save_params = False

    MAX_EPISODE = 1000000
    MAX_STEP = 200

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"     # if gpu is to be used [  cpu, cuda ]

    ############################################################
    LR = 0.01
    GAMMA = 0.9
    EPSILON = 0.9
    BATCH_SIZE = 16
    TARGET_REPLACE_ITER = 25
    MEMORY_CAPACITY = 10000
    LEARN_TMP_STEP = 4      # 多少个 step ---> learn一次

    clamp_ = 5              # grad clamp

    ############# --------------- reward ! ---------------------
    inter_size = 2             # 新的探索范围界定, positions = np.array(position_ // arg.inter_size, dtype= 'int32')
    reward_new_position = 0.  # 探索奖励
    reward_old_position = -0  # 停滞惩罚

    r_d_positive = 1  # 正面距离奖励系数
    r_d_negtive = 0  # 负面距离奖励系数

    reward_done = -1

    ############### -------- print -------------------- # True, False
    resize = 64  # resize : conv screen

    print_loss = False
    print_r_d = False
    plot_net = False        # print(list(self.eval_net.parameters()))

    Plot = True
    Plot_reward = True
    Plot_mean_reward = False
    plot_mean_loss = True
    # ----------------- pre_process_state -------------------
    preprocess_state = 1  # 预处理


    fill = -1

    img_ret_type = 'gray'  # img_ret_type ---> ['default', 'gray', 'canny', 'add', 'dim_canny', 'line']
    show_pre_image = False
    show_orig_img = False       # plot original image ----> False, True
    show_resize_img = True       # plot original image ----> False, True    # 有可能 position 和 destination 错位

    # start = (90, 543)
    # end = (758, 466)
    start = (200, 210)
    end = (758, 466)

    # start = (207, 601)
    # ptype, pcolor = fill, (0, 255, 0)
    ptype, pcolor = fill, (255, 255, 255)
    ptype_dest, pcolor_dest = fill, (150, 150, 150)
    ptype_source, pcolor_source = fill, (200, 200, 200)

    debug_reward_d_p = False        # 调试 r_d, r_p , r
    debug_reward_d_p_all = False

    # -----------------------------

    # fill = 2

    wind_w, wind_h = 816, 647
    wind_scale = wind_h / wind_w

    # ----- resize 的转换细节
    conv_k = resize  # resize to ?
    wind_cw = conv_k
    # wind_ch = int(conv_k * wind_scale)
    wind_ch = int(conv_k)  # 转换为正方形
    wind_conv_wh = wind_cw, wind_ch      # conv to (w, h)



    reset_sleep = 0.1

    init_0 = 0
    control_0 = 0

    # False True
    debug = False


    control_preprocess_img = True
    control_prt_feedback = False
    debug_position_error = False


    ###############


    show_s_ = True

    if (resize):
        N_S = wind_ch * wind_cw
        w, h = wind_ch, wind_cw    # (C, H, W)
    else:
        N_S = wind_h * wind_w
        w, h = wind_w, wind_h

    # action_name = ['left', 'right', 'shift_press', 'shift_up', 'stop']
    action_name = ['left', 'right', 'shift_right_high', 'shift_right_low','shift_left_high', 'shift_left_high', 'stop']
    N_A = len(action_name)
    action_space = list(range(N_A))

    # reset_sleep = 0.

###################################################
# arg.control_pre_img
# from args import arg
###################################################

# def debug_f(self, print_str):
#     if(self.debug):     print(print_str)

# if (arg.debug):
#     print('---------- OVER MAX_STEP! ----------')
#     print('warning --- len_position_:', position_.shape)