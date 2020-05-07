import ctypes
import time
#------------------------ win32 api
if(1):
    ########################################
    # 前台后台
    import win32api, win32con, win32gui
    import sys, os

    pre_path = os.getcwd().rsplit('\\', 1)[0]
    sys.path.append(pre_path + '\\mycode')
    from mycode.my_time import Time, vk
    from Env import Env
    import win32com
    import win32com.client

    tt = Time()
    env = Env()
    env.wind.hwnd
    hwnd = 395422
    tt.sleep(1)
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    shell = win32com.client.Dispatch("WScript.Shell")
    shell.SendKeys('%')
    win32gui.SetForegroundWindow(hwnd)
    tt.sleep(1)


    win32gui.SetBkMode(hwnd, win32con.TRANSPARENT)

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# directx scan codes http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
# while (0):
#     PressKey(0x11)
#     time.sleep(1)
#     ReleaseKey(0x11)
#     time.sleep(1)

#scan codes: https://blog.csdn.net/gatieme/article/details/21095013
from time import time
from time import sleep
class Time:

    def __init__(self):

        self.t0 = time()
        self.t1 = time()
        self.now()

    def now(self):

        self.t1 = time()
        return self.t1 - self.t0

    def exceed(self, T):
        if (self.now() >= T):
            return 1
        else:
            return 0

    1
class VK:
    Time = 0.1
    Constant = 800

    code = {'backspace': 0x08,
               'tab': 0x09,
               'clear': 0x0C,
               'enter': 0x0D,
               'shift': 0x10,
               'ctrl': 0x11,
               'alt': 0x12,
               'pause': 0x13,
               'caps_lock': 0x14,
               'esc': 0x1B,
               'spacebar': 0x20,
               'page_up': 0x21,
               'page_down': 0x22,
               'end': 0x23,
               'home': 0x24,
               'left_arrow': 0x25,
               'up_arrow': 0x26,
               'right_arrow': 0x27,
               'down_arrow': 0x28,
               'select': 0x29,
               'print': 0x2A,
               'execute': 0x2B,
               'print_screen': 0x2C,
               'ins': 0x2D,
               'del': 0x2E,
               'help': 0x2F,
               '0': 0x30,
               '1': 0x31,
               '2': 0x32,
               '3': 0x33,
               '4': 0x34,
               '5': 0x35,
               '6': 0x36,
               '7': 0x37,
               '8': 0x38,
               '9': 0x39,
               'a': 0x41,
               'b': 0x42,
               'c': 0x43,
               'd': 0x44,
               'e': 0x45,
               'f': 0x46,
               'g': 0x47,
               'h': 0x48,
               'i': 0x49,
               'j': 0x4A,
               'k': 0x4B,
               'l': 0x4C,
               'm': 0x4D,
               'n': 0x4E,
               'o': 0x4F,
               'p': 0x50,
               'q': 0x51,
               'r': 0x52,
               's': 0x53,
               't': 0x54,
               'u': 0x55,
               'v': 0x56,
               'w': 0x57,
               'x': 0x58,
               'y': 0x59,
               'z': 0x5A,
               'numpad_0': 0x60,
               'numpad_1': 0x61,
               'numpad_2': 0x62,
               'numpad_3': 0x63,
               'numpad_4': 0x64,
               'numpad_5': 0x65,
               'numpad_6': 0x66,
               'numpad_7': 0x67,
               'numpad_8': 0x68,
               'numpad_9': 0x69,
               'multiply_key': 0x6A,
               'add_key': 0x6B,
               'separator_key': 0x6C,
               'subtract_key': 0x6D,
               'decimal_key': 0x6E,
               'divide_key': 0x6F,
               'F1': 0x70,
               'F2': 0x71,
               'F3': 0x72,
               'F4': 0x73,
               'F5': 0x74,
               'F6': 0x75,
               'F7': 0x76,
               'F8': 0x77,
               'F9': 0x78,
               'F10': 0x79,
               'F11': 0x7A,
               'F12': 0x7B,
               'F13': 0x7C,
               'F14': 0x7D,
               'F15': 0x7E,
               'F16': 0x7F,
               'F17': 0x80,
               'F18': 0x81,
               'F19': 0x82,
               'F20': 0x83,
               'F21': 0x84,
               'F22': 0x85,
               'F23': 0x86,
               'F24': 0x87,
               'num_lock': 0x90,
               'scroll_lock': 0x91,
               'left_shift': 0xA0,
               'right_shift ': 0xA1,
               'left_control': 0xA2,
               'right_control': 0xA3,
               'left_menu': 0xA4,
               'right_menu': 0xA5,
               'browser_back': 0xA6,
               'browser_forward': 0xA7,
               'browser_refresh': 0xA8,
               'browser_stop': 0xA9,
               'browser_search': 0xAA,
               'browser_favorites': 0xAB,
               'browser_start_and_home': 0xAC,
               'volume_mute': 0xAD,
               'volume_Down': 0xAE,
               'volume_up': 0xAF,
               'next_track': 0xB0,
               'previous_track': 0xB1,
               'stop_media': 0xB2,
               'play/pause_media': 0xB3,
               'start_mail': 0xB4,
               'select_media': 0xB5,
               'start_application_1': 0xB6,
               'start_application_2': 0xB7,
               'attn_key': 0xF6,
               'crsel_key': 0xF7,
               'exsel_key': 0xF8,
               'play_key': 0xFA,
               'zoom_key': 0xFB,
               'clear_key': 0xFE,
               '+': 0xBB,
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
vk = VK
vk.code['F1']

tt = Time()


title_ = '1 (2).txt - 记事本'
class_ = 'Notepad'

title_ = ''
class_ = 'Edit'
hwnd_0 = win32gui.FindWindow(class_,title_)
hwnd_0
hwnd = win32gui.FindWindowEx(hwnd_0,0,None,None)
hwnd



win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

win32gui.SendMessage(hwnd,win32con.MOUSEEVENTF_LEFTDOWN)


win32api.keybd_event(vk.code['ctrl'],0,0,0)  #ctrl键位码是17

win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
sleep(1)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

win32gui.SendMessage(hwnd,win32con.MOUSEEVENTF_LEFTDOWN)

sleep(1)





win32api.SendMessage(hwnd, win32con.WM_CHAR, vk.r ,0)

#SendMessage
x,y = 10,30
long_position = win32api.MAKELONG(x,y);print(long_position)

#I wanna be the guy
#title_ = '3213.txt - 记事本'
#class_0 = 'Notepad'
title_ = 'I wanna be the guy'
class_0 = 'Mf2MainClassTh'
hwnd_0 = win32gui.FindWindow(class_0,title_)
hwnd_0
hwnd = win32gui.FindWindowEx(hwnd_0,0,None,None)
hwnd
sleep(1)
if(1):
    # 激活窗口
    if(1):
        sleep(0.5)
        import win32com.client

        aaa = win32com.client.Dispatch("WScript.Shell")
        aaa.SendKeys('%')
        sleep(0.5)
        win32gui.SetForegroundWindow(hwnd)
    # 输入特殊键
    if (1):
        sleep(0.5)
        win32gui.SendMessage(hwnd, win32con.WM_KEYDOWN, vk.left, 0)
        sleep(0.5)
        win32gui.SendMessage(hwnd, win32con.WM_KEYDOWN, vk.right, 0)
    # 字符 按下和松开
    if(1):
        for i in range(5):

            sleep(1)
            win32api.SendMessage(hwnd, win32con.WM_CHAR, vk.code['r'],0)
        # win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, vk.f1, 0)
        # win32api.SendMessage(hwnd, win32con.WM_KEYUP, vk.code['a'], 0)
        # win32api.SendMessage(hwnd, win32con.WM_CHAR, vk.code['a'],
        #                      win32api.MAKELONG(vk.code['a'], win32con.WM_KEYDOWN))
        # #sleep(0.001)
        # win32api.SendMessage(hwnd, win32con.WM_CHAR, vk.code['a'],
        #                      win32api.MAKELONG(vk.code['a'], win32con.WM_KEYUP))
    # 输入字符
    if (1):
    win32api.SendMessage(hwnd, win32con.WM_CHAR, vk.code['a'],
                         win32api.MAKELONG(vk.code['a'], win32con.WM_KEYUP))

    #输入字符串
    win32gui.SendMessage(hwnd, win32con.WM_SETTEXT, None, '11')


win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, vk.right, 0)

win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, vk.right, 0)
sleep(0.5)
win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, vk.left, 0)
##############
sleep(1)
win32gui.SetForegroundWindow(hwnd)
sleep(0.5)
win32api.SendMessage(hwnd,win32con.WM_KEYDOWN, vk.f1, 0)
sleep(0.5)
win32api.SendMessage(hwnd,win32con.WM_KEYUP, vk.code['a'], 0)


win32api.SendMessage(hwnd,win32con.WM_KEYDOWN, win32con.SC_KEYMENU, win32con.MAKELONG(0x1e))
#(win32con.WPARAM)'E'
win32api.PostMessage(hwnd, win32con.WM_SYSCOMMAND, win32con.SC_KEYMENU, )
win32api.MAKELONG(0x1e)
win32api.MapVirtualKey(0x1e)

win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, ord('c'), 0)

win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, win32con.MK_LBUTTON, long_position)

win32api.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, long_position)
win32api.SendMessage(hwnd_0, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, long_position)

win32gui.SendMessage(hwnd, win32con.WM_SETTEXT,None,'1a')

vk.left
win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_LEFT, 0)
sleep(1)
win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, ord('S'), 0)

sleep(1)
win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, ord('A'), 0)
sleep(1)
win32api.SendMessage(hwnd, win32con.WM_KEYUP, ord('A'), 0)

sleep(1)
win32api.keybd_event(0x0D, hwnd, 0, 0)

