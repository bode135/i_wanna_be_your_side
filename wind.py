import win32api, win32gui, win32con, win32process
from grabscreen import grab_screen, plt_img, cv_img
from mycode.my_time import Time, vk
import ctypes


tt = Time()

class Window():
    def __init__(self,
                 class_name = '',
                 title_name = '',
                 hwnd = None
                 ):
        if(hwnd == None):
            self.class_name = class_name
            self.title_name = title_name

            self.hwnd = win32gui.FindWindow(class_name, None)
        else:
            # hwnd= win32gui.FindWindow(class_name, None)

            self.hwnd = hwnd
            self.class_name = win32gui.GetClassName(hwnd)
            self.title_name = win32gui.GetWindowText(hwnd)

        self.get_rect()




        # self.h = self.y2 - self.y1
        # self.w = self.x2 - self.x1
        self.w, self.h = 816, 647

        self.process_id = win32process.GetWindowThreadProcessId(self.hwnd)[1]

    def grab_screen(self):
        self.get_rect()
        screen = grab_screen(self.rect, self.hwnd)
        return screen

    def get_rect(self):
        self.rect = win32gui.GetWindowRect(self.hwnd)
        self.x1, self.y1, self.x2, self.y2 = self.rect

        return self.rect

    def move_to(self,x,y, ):
        win32gui.MoveWindow(self.hwnd, x, y, self.w, self.h, 1)
        return self.get_rect()

    def plt_screen(self):
        screen = self.grab_screen()
        return plt_img(screen)

    def activate(self):
        win32gui.SendMessage(self.hwnd, win32con.WM_ACTIVATE, win32con.WA_CLICKACTIVE, 0)
        win32gui.SendMessage(self.hwnd, win32con.WM_SETFOCUS, 0, 0)

    def key_press(self, key):
        win32gui.SendMessage(self.hwnd, win32con.WM_KEYDOWN, key, 0)

    def key_up(self, key):
        win32gui.SendMessage(self.hwnd, win32con.WM_KEYUP, key, 0)

    def key_dp(self, key, t = 0.001):
        win32gui.SendMessage(self.hwnd, win32con.WM_KEYDOWN, key, 0)
        tt.sleep(t)
        win32gui.SendMessage(self.hwnd, win32con.WM_KEYUP, key, 0)

    def ShowWindow(self, param = win32con.SW_RESTORE):
        # import win32com.client
        # shell = win32com.client.Dispatch("WScript.Shell")
        # shell.SendKeys('%')
        # tt.sleep(0.1)
        # win32gui.SetForegroundWindow(self.hwnd)
        # tt.sleep(0.1)

        win32gui.ShowWindow(self.hwnd, param)

    1


if __name__ == '__main__':
    class_name = 'YYGameMakerYY'
    wind = Window(class_name)

    tt.sleep(1)
    wind.ShowWindow()
    plt_img(wind.grab_screen())
    wind.move_to(0,0)


    wind.hwnd
    wind.process_id
    wind.move_to(0,0)

    process_handle = win32api.OpenProcess(0x1F0FFF, False, wind.process_id)

    kernel32 = ctypes.windll.LoadLibrary(r'C:\Windows\System32\kernel32.dll')
    kernel32.ReadProcessMemory(int(process_handle), )
    int(process_handle)

    win32api.GetModuleHandle()


    wind.key_dp(vk.r, 0.5)
    for i in range(10):
        if(tt.stop_alt('s')):   print('--- break ---'); break

        tt.sleep(0.1)
        if(i % 2):
            wind.key_dp(vk.left, 0.5)
        else:
            wind.key_dp(vk.right, 0.5)

    press_t = 0.5

    wind.key_dp(vk.r, 0.5)

    t0 = Time(); i = 0;
    while(t0.during(10)):
        #t0.sleep(0.1)
        if(1):  # control_parameters
            i += 1
            if (tt.stop_alt('s')):   break

        wind.key_dp(vk.shift, 0.1)

        if (i % 2):
            wind.key_dp(vk.left, press_t)
        else:
            wind.key_dp(vk.right, press_t)
        wind.plt_screen()   # 打印游戏屏幕

        screen0 = wind.grab_screen()
        screen0.shape
        plt_img(screen0)



