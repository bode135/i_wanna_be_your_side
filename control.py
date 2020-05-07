import win32com.client, win32con, win32gui
from mycode.my_time import Time, vk



for i in range(100):
    if(tt.stop('s')):   break
    tt.sleep(0.5)
    win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_CLICKACTIVE, 0)
    win32gui.SendMessage(hwnd, win32con.WM_SETFOCUS, 0, 0)
    # tt.sleep(0.5)
    # win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)

    tt.sleep(0.5)
    win32gui.SendMessage(hwnd, win32con.WM_KEYDOWN, vk.r, 0)
    tt.sleep(0.5)
    win32gui.SendMessage(hwnd, win32con.WM_KEYUP, vk.r, 0)

    tt.sleep(0.5)
    win32gui.SendMessage(hwnd, win32con.WM_KEYDOWN, vk.left, 0)
    tt.sleep(0.5)
    win32gui.SendMessage(hwnd, win32con.WM_KEYUP, vk.left, 0)
    win32gui.SendMessage(hwnd, win32con.WM_KEYDOWN, vk.right, 0)
    tt.sleep(0.5)
    win32gui.SendMessage(hwnd, win32con.WM_KEYDOWN, vk.right, 0)