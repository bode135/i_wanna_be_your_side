# Done by Frannecklp

import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import matplotlib.pyplot as plt

def grab_screen(region=None, hwin = win32gui.GetDesktopWindow()):
    #hwin = win32gui.GetDesktopWindow()
    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
# functions
if(1):
    def cv_img(image, conv = 'rgb'):  # cv plot image, default: dtype(image) == BGR
        if(conv == 'rgb'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif(conv == 'gray'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imshow('window', image)  # show img

        if cv2.waitKey(25) & 0xFF == ord('1'):
            cv2.destroyAllWindows()
            return 1
        return 0
        # cv2.imshow('image', image)
        # print(cv2.waitKey(0))
        # cv2.destroyAllWindows()
    def plt_img(image):  # plt plot image
        plt.imshow(image, cmap='gray')
        plt.show()

    1

if __name__ == '__main__':
    from wind import Window

    class_name = 'YYGameMakerYY'
    wind = Window(class_name)
    wind.hwnd
    wind.rect

    wind.move_to(0, 0)

    screen = wind.grab_screen()
    # screen = grab_screen(wind.rect, wind.hwnd)
    plt_img(screen)