import cv2
import matplotlib.pyplot as plt


def match_img(target, template, min_threshold = 0.5, multip_res = 1 ):
    '''
    :param target:              目标图
    :param template:            模板图
    :param min_threshold:       相似度阈值
    :param multip_res:          最多返回多少个结果
    :return:                    模板图的位置矩阵, list类型
    '''

    # 获得模板图片的高宽尺寸
    if(0):      # 压缩为 1/2 大小
        targ_h, targ_w, _ = target.shape  # scale = 648/817

        targ_h, targ_w = int(targ_h // 2), int(targ_w // 2)
        targ_h, targ_w
        target = cv2.resize(target, (targ_w, targ_h), interpolation=cv2.INTER_AREA)
        target.shape

        targ_h, targ_w, _ = template.shape
        targ_h, targ_w = int(targ_h // 2), int(targ_w // 2)
        template = cv2.resize(target, (targ_w, targ_h), interpolation=cv2.INTER_AREA)

    theight, twidth = template.shape[:2]

    # 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED , 此时应返回最小匹配值(需要查文档)
    # target = s_[-1]
    # target.shape
    # template.shape

    result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)


    # result = cv2.matchTemplate(target, template, method = cv2.TM_CCOEFF)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)


    # min_loc : width * height, numpy: row * column
    results = []
    for i in range(multip_res):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if( min_val < min_threshold):
            #print(min_val)
            #result[min_loc[1], min_loc[0]]
            #results.append([min_loc[1], min_loc[0]])  # append min_loc
            results.append( ( min_loc, (min_loc[0] + twidth, min_loc[1] + theight)  ))  # append min_loc

            # avoid rep
            tmp_x, tmp_y = 10, 10
            x0, x1 = min_loc[1] - tmp_x, min_loc[1] + tmp_x     # wigth
            y0, y1 = min_loc[0] - tmp_y, min_loc[0] + tmp_y     # height
            result[ x0:x1, y0:y1 ] = 0.9

        else:
            break

    return results

def cv2_show(target):
    cv2.imshow("MatchResult----MatchingValue", target)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # 读取目标图片
    pic_path = 'picture\\'
    # filename = pic_path + 'screen0.png'
    # filename = pic_path + 'my_underwear.bmp'
    filename = pic_path + 'screen0.png'
    target = cv2.imread(filename)

    # 读取模板图片
    filename = pic_path + 'my_underwear.png'
    # filename = pic_path + 'destination.png'
    template = cv2.imread(filename)

    target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    # match
    if(1):
        results = match_img(target, template)

        for result in results:
            # min_loc = result
            print(result)
            xy0, xy1 = result[0], result[1]
            cv2.rectangle(target, xy0, xy1, (255, 255, 255), 2)

    # plt
    if(1):
        # 显示结果,并将匹配值显示在标题栏上
        cv2.imshow("MatchResult----MatchingValue", target)
        cv2.waitKey()
        cv2.destroyAllWindows()





    # cv2_show(target)
    #
    # from mycode.my_time import Time, vk
    # tt = Time()
    # for i in range(100):
    #     results = match_img(target, template)
    # print(tt.now())
    #
    # tracker = cv2.multiTracker_create()
    # cv2.TrackerKCF_create()