#imports
from time import time
from time import sleep

class DM:
    def __init__(self):
        try:
            from win32com.client import Dispatch
            #dm = Dispatch('dm.dmsoft')  # 调用大漠插件
            self.dm = Dispatch('dm.dmsoft')  # 调用大漠插件
        except:
            import os
            os.system('regsvr32 dm.dll /s')
            print('注册dm.dll')
            self.dm = Dispatch('dm.dmsoft')  # 调用大漠插件

        print('版本：', self.ver(),'，ID：',self.GetID())

    def reg(self):  #重名Reg报错，原因不明
        return self.dm.Reg('albin7a7a6b9740b219fb4db62c7865e00437', '123')
    # def Reg(self):
    #     return self.dm.Reg('albin7a7a6b9740b219fb4db62c7865e00437', '123')

        #print('reg: ', self.dm.Reg('albin7a7a6b9740b219fb4db62c7865e00437', '123'))
        # print('reg: ',dm.Reg('xxhongdev4dd3dddabe56dfb834fb3b70f0cea8ee', '123'))
        # dm.Reg('fzdxwhl20070b412e2fa2fc4462a6c25de3826e859e','147')
        # dm.Reg('xxhongdev4dd3dddabe56dfb834fb3b70f0cea8ee','Oa0UhMx5rC')

    def GetDmCount(self):
        return self.dm.GetDmCount()
    def GetID(self):
        return self.dm.GetID()

    # base
    if (1):
        def ver(self):
            return self.dm.ver()

        def Reg(self, reg_code, ver_info):
            return self.dm.Reg(reg_code, ver_info)

        def GetDir(self, type):
            return self.GetDir(type)

        def GetBasePath(self):
            return self.dm.GetBasePath()

        def GetPath(self):
            return self.dm.GetPath()

        def GetID(self):
            return self.dm.GetID()

        def SetDisplayInput(self, mode):
            return self.dm.SetDisplayInput(mode)

        def SetShowErrorMsg(self, show):
            return self.dm.SetShowErrorMsg(show)

        1

    # color
    if (1):
        def Capture(self, x1, y1, x2, y2, file):
            return self.dm.Capture(x1, y1, x2, y2, file)

        def FindPic(self, x1, y1, x2, y2, pic_name, delta_color='101010', sim=0.9, dir=0, intX=0, intY=0):
            return self.dm.FindPic(x1, y1, x2, y2, pic_name, delta_color='101010', sim=0.9, dir=0, intX=0, intY=0)

        def FindColor(self, x1, y1, x2, y2, color, sim, dir, intX, intY):
            return self.dm.FindColor(x1, y1, x2, y2, color, sim, dir, intX, intY)

        def LoadPic(self, pic_name):
            return self.dm.LoadPic(pic_name)

        def LoadPic(self, pic_name):
            return self.dm.LoadPic(pic_name)

        def FreePic(self, pic_name):
            return self.dm.FreePic(pic_name)

        def GetColor(self, x, y):
            return self.dm.GetColor(x, y)

        def GetPicSize(self, pic_name):
            return self.dm.GetPicSize(pic_name)

        def BGR2RGB(self, bgr_color):
            return self.dm.BGR2RGB(bgr_color)

        def CmpColor(self, x, y, color, sim):
            return self.dm.CmpColor(x, y, color, sim)

        1

    # window
    if (1):
        def BindWindow(self, hwnd, display=['normal', 'gdi', 'gdi2', 'dx', 'dx2'][1],
                       mouse=['normal', 'windows', 'windows2', 'windows3', 'dx', 'dx2'][1],
                       keypad=['normal', 'windows', 'dx'][1],
                       mode=[0, 1, 2, 3, 4, 5, 6, 7, 101, 103][0]):
            return self.dm.BindWindow(hwnd, display, mouse, keypad, mode)

        def UnBindWindow(self):
            return self.dm.UnBindWindow()

        def IsBind(self, hwnd):
            return self.dm.IsBind(hwnd)

        def MoveWindow(self, hwnd, x, y):
            return self.dm.MoveWindow(hwnd, x, y)

        def FindWindow(self, class_name='', title_name=''):
            return self.dm.FindWindow(class_name, title_name)

        def ClientToScreen(self, hwnd, x, y):
            return self.dm.ClientToScreen(hwnd, x, y)

        def ScreenToClient(self, hwnd, x, y):
            return self.dm.ScreenToClient(hwnd, x, y)

        def FindWindowByProcess(self, process_name, class_name, title_name):
            return self.dm.FindWindowByProcess(process_name, class_name, title_name)

        def FindWindowByProcessId(self, process_id, class_, title):
            return self.dm.FindWindowByProcessId(process_id, class_, title)

        def GetClientRect(self, hwnd, x1, y1, x2, y2):
            return self.dm.GetClientRect(hwnd, x1, y1, x2, y2)

        def GetClientSize(self, hwnd, width, height):
            return self.dm.GetClientSize(hwnd, width, height)

        def GetWindowRect(self, hwnd, x1, y1, x2, y2):
            return self.dm.GetWindowRect(hwnd, x1, y1, x2, y2)

        def GetWindow(self, hwnd, flag):
            return self.dm.GetWindow(hwnd, flag)

        def GetWindowProcessPath(self, hwnd):
            return self.dm.GetWindowProcessPath(hwnd)

        def SetWindowSize(self, hwnd, width, height):
            return self.dm.SetWindowSize(hwnd, width, height)

        def SetWindowState(self, hwnd, flag):
            return self.dm.SetWindowState(hwnd, flag)

        def SetWindowText(self, hwnd, title):
            return self.dm.SetWindowText(hwnd, title)

        def SetWindowTransparent(self, hwnd, trans):
            return self.dm.SetWindowTransparent(hwnd, trans)

        def EnumWindow(self, parent, title, class_name, filter):
            return self.dm.EnumWindow(parent, title, class_name, filter)

        def EnumWindowByProcess(self, process_name, title, class_name, filter):
            return self.dm.EnumWindowByProcess(process_name, title, class_name, filter)

        def EnumWindowSuper(self, spec1, flag1, type1, spec2, flag2, type2, sort):
            return self.dm.EnumWindowSuper(spec1, flag1, type1, spec2, flag2, type2, sort)

        1

    # key_mouse
    if (1):
        def GetCursorPos(self, x=0, y=0):
            return self.dm.GetCursorPos(x, y)

        def GetKeyState(self, vk_code):
            return self.dm.GetKeyState(vk_code)

        def SetKeypadDelay(self, type=['normal', 'windows', 'dx'][-1], delay=[0.03, 0.01, 0.05][-1]):
            return self.dm.SetKeypadDelay(type, delay)

        def SetMouseDelay(self, delay=[0.03, 0.01, 0.04][-1], type=['normal', 'windows', 'dx'][-1]):
            return self.dm.SetMouseDelay(type, delay)

        def WaitKey(self, vk_code, time_out=0):
            # vk_code = 'a'
            # vk_code.__class__.__name__ == 'str'
            # vk_code.upper()
            # kk
            # if(vk_code.__class__.)
            return self.dm.WaitKey(vk_code, time_out)

        # def WaitKey(vk_code,time_out = 0):
        #     # vk_code = 'a'
        #     # vk_code.__class__.__name__ == 'str'
        #     # vk_code.upper()
        #     # kk
        #     # if(vk_code.__class__.)
        #
        #     return print(vk_code, time_out)
        # WaitKey('a')

        def KeyDown(self, vk_code):
            return self.dm.KeyDown(vk_code)

        def KeyDownChar(self, key_str):
            return self.dm.KeyDownChar(key_str)

        def KeyPress(self, vk_code):
            return self.dm.KeyPress(vk_code)

        def KeyPressChar(self, key_str):
            return self.dm.KeyPressChar(key_str)

        def KeyPressStr(self, key_str, delay):
            return self.dm.KeyPressStr(key_str, delay)

        def KeyUp(self, vk_code):
            return self.dm.KeyUp(vk_code)

        def KeyUpChar(self, key_str):
            return self.dm.KeyUpChar(key_str)

        def LeftClick(self, ):
            return self.dm.LeftClick()

        def LeftDoubleClick(self, ):
            return self.dm.LeftDoubleClick()

        def LeftDown(self, ):
            return self.dm.LeftDown()

        def LeftUp(self, ):
            return self.dm.LeftUp()

        def MiddleClick(self, ):
            return self.dm.MiddleClick()

        def MoveR(self, rx, ry):
            return self.dm.MoveR(rx, ry)

        def MoveTo(self, x, y):
            return self.dm.MoveTo(x, y)

        def MoveToEx(self, x, y, w, h):
            return self.dm.MoveToEx(x, y, w, h)

        def RightClick(self, ):
            return self.dm.RightClick()

        def RightDown(self, ):
            return self.dm.RightDown()

        def RightUp(self, ):
            return self.dm.RightUp()

        def SetKeypadDelay(self, type, delay):
            return self.dm.SetKeypadDelay(type, delay)

        def SetMouseDelay(self, type, delay):
            return self.dm.SetMouseDelay(type, delay)

        def WaitKey(self, vk_code, time_out):
            return self.dm.WaitKey(vk_code, time_out)

        def WheelDown(self, ):
            return self.dm.WheelDown()

        def WheelUp(self, ):
            return self.dm.WheelUp()

        1

    # memmory
    if (1):
        def FindData(self, hwnd, addr_range, data):
            return self.dm.FindData( hwnd, addr_range, data)
        def FindDataEx(self, hwnd, addr_range, data, step, multi_thread, mode):
            return self.dm.FindDataEx(hwnd, addr_range, data, step, multi_thread, mode)

        def DoubleToData(self, value):
            return self.dm.DoubleToData(value)

        def FloatToData(self, value):
            return self.dm.FloatToData(value)

        def GetModuleBaseAddr(self, hwnd, module):
            return self.dm.GetModuleBaseAddr(hwnd, module)

        def IntToData(self, value, type):
            return self.dm.IntToData(value, type)

        def ReadData(self, hwnd, addr, len):
            return self.dm.ReadData(hwnd, addr, len)

        def ReadDouble(self, hwnd, addr):
            return self.dm.ReadDouble(hwnd, addr)

        def ReadFloat(self, hwnd, addr):
            return self.dm.ReadFloat(hwnd, addr)

        def ReadInt(self, hwnd, addr, type):
            return self.dm.ReadInt(hwnd, addr, type)

        def ReadString(self, hwnd, addr, type, len):
            return self.dm.ReadString(hwnd, addr, type, len)

        def StringToData(self, value, type):
            return self.dm.StringToData(value, type)

        def WriteData(self, hwnd, addr, data):
            return self.dm.WriteData(hwnd, addr, data)

        def WriteDouble(self, hwnd, addr, v):
            return self.dm.WriteDouble(hwnd, addr, v)

        def WriteFloat(self, hwnd, addr, v):
            return self.dm.WriteFloat(hwnd, addr, v)

        def WriteInt(self, hwnd, addr, type, v):
            return self.dm.WriteInt(hwnd, addr, type, v)

        def WriteString(self, hwnd, addr, type, v):
            return self.dm.WriteString(hwnd, addr, type, v)

        1

    # file
    if (1):
        def CopyFile(self, src_file, dst_file, over):
            return self.dm.CopyFile(src_file, dst_file, over)

        def CreateFolder(self, folder):
            return self.dm.CreateFolder(folder)

        def DecodeFile(self, file, pwd):
            return self.dm.DecodeFile(file, pwd)

        def DeleteFile(self, file):
            return self.dm.DeleteFile(file)

        def DeleteFolder(self, folder):
            return self.dm.DeleteFolder(folder)

        def DeleteIni(self, section, key, file):
            return self.dm.DeleteIni(section, key, file)

        def DeleteIniPwd(self, section, key, file, pwd):
            return self.dm.DeleteIniPwd(section, key, file, pwd)

        def DownloadFile(self, url, save_file, timeout):
            return self.dm.DownloadFile(url, save_file, timeout)

        def EncodeFile(self, file, pwd):
            return self.dm.EncodeFile(file, pwd)

        def GetFileLength(self, file):
            return self.dm.GetFileLength(file)

        def IsFileExist(self, file):
            return self.dm.IsFileExist(file)

        def MoveFile(self, src_file, dst_file):
            return self.dm.MoveFile(src_file, dst_file)

        def ReadFile(self, file):
            return self.dm.ReadFile(file)

        def ReadIni(self, section, key, file):
            return self.dm.ReadIni(section, key, file)

        def ReadIniPwd(self, section, key, file, pwd):
            return self.dm.ReadIniPwd(section, key, file, pwd)

        def SelectDirectory(self, ):
            return self.dm.SelectDirectory()

        def SelectFile(self, ):
            return self.dm.SelectFile()

        def WriteFile(self, file, content):
            return self.dm.WriteFile(file, content)

        def WriteIni(self, section, key, value, file):
            return self.dm.WriteIni(section, key, value, file)

        def WriteIniPwd(self, section, key, value, file, pwd):
            return self.dm.WriteIniPwd(section, key, value, file, pwd)

    # system
    if (1):
        def GetNetTime(self, ):
            return self.dm.GetNetTime()

        def GetOsType(self, ):
            return self.dm.GetOsType()

        def GetScreenHeight(self, ):
            return self.dm.GetScreenHeight()

        def GetScreenWidth(self, ):
            return self.dm.GetScreenWidth()

        def GetTime(self, ):
            return self.dm.GetTime()

        def Is64Bit(self, ):
            return self.dm.Is64Bit()

        def RunApp(self, app_path, mode):
            return self.dm.RunApp(app_path, mode)

        def Play(self, media_file):
            return self.dm.Play(media_file)

        def Stop(self, id):
            return self.dm.Stop(id)

        def Delay(self, mis):
            return self.dm.Delay(mis)

        def ExitOs(self, type):
            return self.dm.ExitOs(type)

        def Beep(self, duration=500, f=500):
            return self.dm.Beep(f, duration)

    # My function
    def stop_0(self, ch='p', continue_key='p'):
        # ch = 'a'
        break0 = 0
        ch = ch.upper()

        if (self.dm.GetKeyState(ord(ch))):
            print('------- Stop! --------')
            from win32gui import MessageBox
            # 'A'.upper()
            break0 = (MessageBox(0, 'Do you continue?', 'Stop!', 1) - 1)
            # break0 = self.WaitKey(ord(continue_key.upper()),0)

            if (break0 == 1):
                print('------ Break!!! ---------')
            # return break0
        return break0
    1

class Time():
    def __init__(self):

        self.t0 = time()
        self.t1 = time()
        self.time = time
        self.sleep = sleep
        self.now()
        self.break_flag = 0       #break_flag

    def now(self,round_ = 3):

        self.t1 = time()
        now = self.t1 - self.t0
        #now = 3.1415926
        now_r = round(now, round_)
        return now_r

    def exceed(self, T):
        if (self.now() >= T):
            return 1
        else:
            return 0


    def during(self,T):
        if (self.now() <= T):
            return 1
        else:
            return 0

    # 检查是否按下了暂停按键p
    def stop(self,ch='p'):
        ch = ch.upper()

        from win32api import GetKeyState
        nVirtKey = GetKeyState(ord(ch))

        if (nVirtKey == -127 or nVirtKey == -128):      #按下
            self.break_flag = 1
            return 1
        else:
            self.break_flag = 0
            return 0

    def stop_0(self,ch='p', continue_key='p'):
        # ch = 'a'
        break0 = 0
        ch = ch.upper()

        from win32api import GetKeyState
        nVirtKey = GetKeyState(ord(ch))

        if (nVirtKey == -127 or nVirtKey == -128):
            print('------- Stop! --------')
            from win32gui import MessageBox
            # 'A'.upper()
            break0 = not (MessageBox(0, 'Do you break?', 'Stop!', 1) - 1)   #询问是否退出
            if (break0):
                print('------ Break!!! --------')
                self.break_flag = 1
                return break0
        self.break_flag = 0
        return break0

    def get_key_state(self,ch='p'):
        ch = ch.upper()

        from win32api import GetKeyState
        nVirtKey = GetKeyState(ord(ch))

        if (nVirtKey == -127 or nVirtKey == -128):
            return 1
        else:
            return 0

    def sleep_check_stop(self, t, type = 1, ch = 'p'):
        self.sleep(t)

        if(type):
            self.stop(ch)
        else:
            self.stop_0(ch)

        pass


    1

class vk:
    Time = 0.1
    Constant = 8000 #内部标记

    backspace = 8
    tab = 9
    clear = 12
    enter = 13
    shift = 16
    ctrl = 17
    alt = 18
    pause = 19
    caps_lock = 20
    esc = 27
    spacebar = 32
    page_up = 33
    page_down = 34
    end = 35
    home = 36
    left_arrow = 37
    up_arrow = 38
    right_arrow = 39
    down_arrow = 40
    select = 41
    print = 42
    execute = 43
    print_screen = 44
    insert = 45
    delete = 46
    help = 47
    num_0 = 48
    num_1 = 49
    num_2 = 50
    num_3 = 51
    num_4 = 52
    num_5 = 53
    num_6 = 54
    num_7 = 55
    num_8 = 56
    num_9 = 57
    a = 65
    b = 66
    c = 67
    d = 68
    e = 69
    f = 70
    g = 71
    h = 72
    i = 73
    j = 74
    k = 75
    l = 76
    m = 77
    n = 78
    o = 79
    p = 80
    q = 81
    r = 82
    s = 83
    t = 84
    u = 85
    v = 86
    w = 87
    x = 88
    y = 89
    z = 90
    numpad_0 = 96
    numpad_1 = 97
    numpad_2 = 98
    numpad_3 = 99
    numpad_4 = 100
    numpad_5 = 101
    numpad_6 = 102
    numpad_7 = 103
    numpad_8 = 104
    numpad_9 = 105
    multiply_key = 106
    add_key = 107
    separator_key = 108
    subtract_key = 109
    decimal_key = 110
    divide_key = 111
    F1 = 112
    F2 = 113
    F3 = 114
    F4 = 115
    F5 = 116
    F6 = 117
    F7 = 118
    F8 = 119
    F9 = 120
    F10 = 121
    F11 = 122
    F12 = 123
    F13 = 124
    F14 = 125
    F15 = 126
    F16 = 127
    F17 = 128
    F18 = 129
    F19 = 130
    F20 = 131
    F21 = 132
    F22 = 133
    F23 = 134
    F24 = 135
    num_lock = 144
    scroll_lock = 145
    left_shift = 160
    right_shift = 161
    left_control = 162
    right_control = 163
    left_menu = 164
    right_menu = 165
    browser_back = 166
    browser_forward = 167
    browser_refresh = 168
    browser_stop = 169
    browser_search = 170
    browser_favorites = 171
    browser_start_and_home = 172
    volume_mute = 173
    volume_Down = 174
    volume_up = 175
    next_track = 176
    previous_track = 177
    stop_media = 178
    play = 179
    pause_media = 179
    start_mail = 180
    select_media = 181
    start_application_1 = 182
    start_application_2 = 183
    attn_key = 246
    crsel_key = 247
    exsel_key = 248
    play_key = 250
    zoom_key = 251
    clear_key = 254

    symbol = {'+': 0xBB,
               ',': 0xBC,
               '-': 0xBD,
               '.': 0xBE,
               '/': 0xBF,
               '`': 0xC0,
               ';': 0xBA,
               '[': 0xDB,
               '\\': 0xDC,
               ']': 0xDD,
               "'": 0xDE,
               '\`': 0xC0}
    ##########
    ctrl = 17
    alt = 18
    shift = 16

    f1 = 112
    f2 = 113
    f3 = 114
    f4 = 116
    f5 = 117

    enter = 13
    space = 32
    back = 8

    # 小键盘数字
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = 96, 97, 98, 99, 100, 101, 102, 103, 104, 105

    left, up, right, down = 37, 38, 39, 40

class Key:
    def __init__(self,dm = 0, key='k'):
        if(dm == 0):
            self.dm = DM()
        else:
            self.dm = dm


        if (key.__class__.__name__ == 'str'):
            self.chr = key.upper()
        else:
            self.chr = 'None'
        self.ord = self.conv_ord(key)

    def get_ord(self, key):
        key = key.upper()
        return ord(key)

    def __main__(self):
        return self.ord

    def conv_ord(self, key0):
        key = key0
        if (key.__class__.__name__ == 'str'):
            # key = key.upper()
            key = ord(key.upper())
        elif (key.__class__.__name__ == 'int'):
            if (key >= 0 and key <= 9):
                key = str(key)
                key = ord(key)
        return key

    def conv_chr(self, key):
        key = chr(key)
        return key

    def conv(self, key0):
        key = key0
        if (key == vk.Constant):
            key = self.ord
        else:
            key = self.conv_ord(key)
        return key

    def state(self, key=vk.Constant):
        key = self.conv(key)
        return self.dm.GetKeyState(key)

    def press(self, key0=vk.Constant):
        key = self.conv_ord(key0)

        return self.dm.KeyPress(key)

    def down(self, key0=vk.Constant):
        key = self.conv_ord(key0)

        return self.dm.KeyDown(key)

    def up(self, key0=vk.Constant):
        key = self.conv_ord(key0)

        return self.dm.KeyUp(key)

    def down_up(self, key0=vk.Constant, t=vk.Time):
        key = self.conv_ord(key0)
        self.down(key)
        sleep(t)
        self.up(key)

    def dp(self, key=vk.Constant, t=vk.Time):
        self.down_up(key, t)

    def test_dp(self, key, delay=1, t=vk.Time):
        sleep(delay)
        self.dp(key, t)

    1

    1

class Mouse:
    def __init__(self,dm = 0):
        if (dm == 0):
            self.dm = DM()
        else:
            self.dm = dm

    def position(self):
        return self.dm.GetCursorPos(x=0, y=0)[1:]

    def set_delay(self, delay, type='dx'):
        return self.dm.SetMouseDelay(type, delay)

    def move_to(self, x, y):
        return self.dm.MoveTo(x, y)

    def click_left(self,x, y, t=0.5):
        self.dm.MoveTo(x, y)
        return self.dm.LeftClick()

    def test_click_left(self,x, y, t=0.5, delay=1):
        sleep(delay)
        return self.click_left(x, y, t)

    def click_right(self,x, y, t=0.5):
        self.dm.MoveTo(x, y)
        return self.dm.RightClick()

    def test_click_right(self,x, y, t=0.5, delay=1):
        sleep(delay)
        return self.click_right(x, y, t)

    def LeftClick(self):
        x,y = self.position()
        self.click_left(x, y)

    1

if(0):
    # 初始化
    dm = DM()

    ms = Mouse(dm)
    ms = Mouse()
    ms.position()
    x, y = (0, 0)
    ms.move_to(x, y)
    sleep(1)
    ms.click_left(x, y, 2)
    sleep(1)
    ms.click_right(x, y, 1)

    # 键盘操作
    kk = Key(dm)
    kk.test_dp('a', 1)  # 测试用，1秒后按下a键
    kk.dp('a')  # 按下a键


# debug
# code = {'backspace': 0x08,
#                'tab': 0x09,
#                'clear': 0x0C,
#                'enter': 0x0D,
#                'shift': 0x10,
#                'ctrl': 0x11,
#                'alt': 0x12,
#                'pause': 0x13,
#                'caps_lock': 0x14,
#                'esc': 0x1B,
#                'spacebar': 0x20,
#                'page_up': 0x21,
#                'page_down': 0x22,
#                'end': 0x23,
#                'home': 0x24,
#                'left_arrow': 0x25,
#                'up_arrow': 0x26,
#                'right_arrow': 0x27,
#                'down_arrow': 0x28,
#                'select': 0x29,
#                'print': 0x2A,
#                'execute': 0x2B,
#                'print_screen': 0x2C,
#                'insert': 0x2D,
#                'delete': 0x2E,
#                'help': 0x2F,
#                'num_0': 0x30,
#                'num_1': 0x31,
#                'num_2': 0x32,
#                'num_3': 0x33,
#                'num_4': 0x34,
#                'num_5': 0x35,
#                'num_6': 0x36,
#                'num_7': 0x37,
#                'num_8': 0x38,
#                'num_9': 0x39,
#                'a': 0x41,
#                'b': 0x42,
#                'c': 0x43,
#                'd': 0x44,
#                'e': 0x45,
#                'f': 0x46,
#                'g': 0x47,
#                'h': 0x48,
#                'i': 0x49,
#                'j': 0x4A,
#                'k': 0x4B,
#                'l': 0x4C,
#                'm': 0x4D,
#                'n': 0x4E,
#                'o': 0x4F,
#                'p': 0x50,
#                'q': 0x51,
#                'r': 0x52,
#                's': 0x53,
#                't': 0x54,
#                'u': 0x55,
#                'v': 0x56,
#                'w': 0x57,
#                'x': 0x58,
#                'y': 0x59,
#                'z': 0x5A,
#                'numpad_0': 0x60,
#                'numpad_1': 0x61,
#                'numpad_2': 0x62,
#                'numpad_3': 0x63,
#                'numpad_4': 0x64,
#                'numpad_5': 0x65,
#                'numpad_6': 0x66,
#                'numpad_7': 0x67,
#                'numpad_8': 0x68,
#                'numpad_9': 0x69,
#                'multiply_key': 0x6A,
#                'add_key': 0x6B,
#                'separator_key': 0x6C,
#                'subtract_key': 0x6D,
#                'decimal_key': 0x6E,
#                'divide_key': 0x6F,
#                'F1': 0x70,
#                'F2': 0x71,
#                'F3': 0x72,
#                'F4': 0x73,
#                'F5': 0x74,
#                'F6': 0x75,
#                'F7': 0x76,
#                'F8': 0x77,
#                'F9': 0x78,
#                'F10': 0x79,
#                'F11': 0x7A,
#                'F12': 0x7B,
#                'F13': 0x7C,
#                'F14': 0x7D,
#                'F15': 0x7E,
#                'F16': 0x7F,
#                'F17': 0x80,
#                'F18': 0x81,
#                'F19': 0x82,
#                'F20': 0x83,
#                'F21': 0x84,
#                'F22': 0x85,
#                'F23': 0x86,
#                'F24': 0x87,
#                'num_lock': 0x90,
#                'scroll_lock': 0x91,
#                'left_shift': 0xA0,
#                'right_shift ': 0xA1,
#                'left_control': 0xA2,
#                'right_control': 0xA3,
#                'left_menu': 0xA4,
#                'right_menu': 0xA5,
#                'browser_back': 0xA6,
#                'browser_forward': 0xA7,
#                'browser_refresh': 0xA8,
#                'browser_stop': 0xA9,
#                'browser_search': 0xAA,
#                'browser_favorites': 0xAB,
#                'browser_start_and_home': 0xAC,
#                'volume_mute': 0xAD,
#                'volume_Down': 0xAE,
#                'volume_up': 0xAF,
#                'next_track': 0xB0,
#                'previous_track': 0xB1,
#                'stop_media': 0xB2,
#                'play': 0xB3,
#                 'pause_media': 0xB3,
#                'start_mail': 0xB4,
#                'select_media': 0xB5,
#                'start_application_1': 0xB6,
#                'start_application_2': 0xB7,
#                'attn_key': 0xF6,
#                'crsel_key': 0xF7,
#                'exsel_key': 0xF8,
#                'play_key': 0xFA,
#                'zoom_key': 0xFB,
#                'clear_key': 0xFE,
#                'symbol_+': 0xBB,
#                'symbol_,': 0xBC,
#                'symbol_-': 0xBD,
#                'symbol_.': 0xBE,
#                'symbol_/': 0xBF,
#                'symbol_`': 0xC0,
#                'symbol_;': 0xBA,
#                'symbol_[': 0xDB,
#                'symbol_\\': 0xDC,
#                'symbol_]': 0xDD,
#                "symbol_'": 0xDE,
#                'symbol_\`': 0xC0}
# code.keys()
# code.values()
#
# for k,v in code.items():
#     #print(k,v)
#     string = ' {} = {}'.format(k,v)
#     print(string)
