import win32api, win32gui, win32con
from grabscreen import grab_screen, plt_img, cv_img
from mycode.my_time import Time, vk
import cv2
from wind import Window
import numpy as np
from args import arg

# def func
if(1):
    def process_img(original_image):
        processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)


        return processed_img


    def roi(img, vertices):
        # blank mask:
        mask = np.zeros_like(img)
        # fill the mask
        cv2.fillPoly(mask, vertices, 255)
        # now only show the area that is the mask
        masked = cv2.bitwise_and(img, mask)
        return masked


    def draw_lines(img, lines):
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)

    def pre_process_screen(screen0, ret_type = arg.img_ret_type):
        if (ret_type == 'default'):
            return screen0

        # bgr - gray - canny - gaussian - roi

        screen_gray = cv2.cvtColor(screen0, cv2.COLOR_BGR2GRAY)
        screen_canny = cv2.Canny(screen_gray, threshold1=200, threshold2=300)
        # cv_img(screen_gray)
        # cv_img(screen0)
        # processed_img = screen_gray

        # processed_img = screen_gray + screen_canny
        # processed_img = cv2.add(screen_gray , screen_canny)

        dim_canny = np.array(screen_canny / 1.3 ,dtype= 'uint8')
        # processed_img = np.array((screen_gray)/ 2.1 ,dtype= 'uint8') + screen_canny

        # processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)

        vertices = np.array([[20, 34],
                             [800, 34],
                             [800, 624],
                             [20, 624],
                             ], np.int32)
        vertices


        # processed_img = screen_canny

        if(ret_type == 'add'):
            processed_img = screen_canny + screen_gray
            # processed_img = cv2.cvtColor(screen0, cv2.COLOR_BGR2GRAY)
            # processed_img= screen0
            # processed_img = screen_gray

        elif(ret_type == 'gray'):
            processed_img = screen_gray
        elif (ret_type == 'canny'):
            processed_img = screen_canny
        elif (ret_type == 'dim_canny'):
            processed_img = dim_canny
        elif (ret_type == 'line'):
            processed_img = screen_canny
            lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, 20, 1)
            draw_lines(processed_img, lines)
        # elif (ret_type == 'gray_resize'):
        #     screen_gray_resize = cv2.resize(screen_gray, (600, 400), interpolation=cv2.INTER_AREA)
        #     processed_img = screen_gray_resize

        # cv_img(processed_img)
        # cv_img(roi(processed_img, [vertices]))
        processed_img = roi(processed_img, [vertices])

        return processed_img

    # np.zeros([100,100,3])+255

    # find my_location and destination_location
    def get_loacation():


        pass

    1

if __name__ == '__main__':
    from template import match_img

    tt = Time()
    class_name = 'YYGameMakerYY'
    wind = Window(class_name)
    wind.move_to(0,0)




    if(0):      # read image
        # screen0 = wind.grab_screen()
        # screen_gray = cv2.cvtColor(screen0, cv2.COLOR_BGR2GRAY)
        # screen_canny = cv2.Canny(screen_gray, threshold1=200, threshold2=300)
        #
        #
        # screen_gray
        # screen_canny
        # sg_sc = screen_gray + screen_canny
        # sg_sc0 = cv2.add(screen_gray, screen_canny)


        pic_path = 'picture\\'
        filename = pic_path + 'screen1.png'
        # filename = pic_path + 'my_underwear.bmp'

        # ------- save img
        screen = screen0
        screen = cv2.cvtColor(screen0, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, screen)

        # ------- read img
        screen = cv2.imread(filename)
        screen.shape
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        plt_img(screen)

    RAMDOM_MOVE = 1

    # control_key and print_screen
    if(1):
        # template picture
        pic_path = 'picture\\'
        filename = pic_path + 'my_underwear.png'
        # filename = pic_path + 'human.png'
        template = cv2.imread(filename)
        # template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)

        wind.key_dp(vk.r)
        tt = Time()
        screens = []
        tmp = 0.01
        death_n = 0     # 防止瞬间的目标丢失
        for i in range(10000):

            screen0 = wind.grab_screen()

            # tracking human
            if (1):                     # tracking human
                screen = screen0

                try:
                    results = match_img(screen, template, min_threshold= 0.5,multip_res=1)[0]
                    death_n = 0

                    # if(results == []):
                    #     print('!!!!!!!--- death? ---')
                    #     plt_img(screen)
                    #     tt.sleep(2)
                    #     results = match_img(screen, template, min_threshold=0.5, multip_res=1)[0]
                    #
                    # else:
                    #     results = results[0]

                except:  # 若人物死亡
                    death_n += 1
                    print(death_n)
                    # tt.sleep(0.0001)
                    if(death_n < 3):
                        continue

                    death_n = 0
                    results = ((0, 0), (100, 100))
                    print('--- death! ---')

                    # tt.sleep(1)
                    for j in range(10): #plot death
                        tt.sleep(0.01)

                        screen0 = wind.grab_screen()

                        # screen = screen0
                        #screen = pre_process_screen(screen0)
                        screen = cv2.cvtColor(screen0, cv2.COLOR_BGR2RGB)  # RGB
                        if(cv_img(screen)): break


                    wind.key_dp(vk.r, tmp)
                    continue

                xy0, xy1 = results[0], results[1]

                # xy0, xy1 = (xy0[0] - 10, xy0[0] - 10), (xy1[0] + 10, xy1[0] + 10)
                position = np.array((np.array(xy0) + np.array(xy1)) / 2, dtype= int)
                # print(results)
                print('第{:4}次, position: {}'.format(i, position))

                cv2.rectangle(screen, xy0, xy1, (255, 220, 0), -1)

            #screen = cv2.cvtColor(screen0, cv2.COLOR_BGR2GRAY)
            #screen = cv2.cvtColor(screen0, cv2.COLOR_BGR2GRAY)

            ##################### ---- pre_process screen! ----- #############################


            screen = cv2.cvtColor(screen0, cv2.COLOR_BGR2RGB)     # RGB
            # screen = cv2.resize(screen, (1200, 800), interpolation=cv2.INTER_AREA)

            # process screen
            if(0):      # process screen

                # screen_gray = cv2.cvtColor(screen0, cv2.COLOR_BGR2GRAY)
                # screen_canny = cv2.Canny(screen_gray, threshold1=200, threshold2=300)

                processed_img = cv2.Canny(screen0, threshold1=200, threshold2=300)
                processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)

                # lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, 20, 1)
                # draw_lines(processed_img, lines)

                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

                screen = cv2.add(screen, processed_img)

                # screen = pre_process_screen(screen0)

            screens.append(screen)      # ---- return screen ----
            if(cv_img(screen)): break

            if(RAMDOM_MOVE):
                #wind.key_dp(vk.r, 0.01)
                #tt.sleep(1)
                actions = ['left', 'right', 'shift']
                action = np.random.choice(actions, p = [0.1, 0.9, 0.0])
                if(action == 'left'):
                    wind.key_dp(vk.left, tmp)
                elif(action == 'right'):
                    wind.key_dp(vk.right, tmp)
                elif (action == 'shift'):
                    wind.key_dp(vk.shift, tmp)


            if(0):
                if (i % 2):
                    wind.key_dp(vk.left, tmp)
                else:
                    wind.key_dp(vk.right, tmp)
                1

        print(tt.now())

        cv2.destroyAllWindows()



