import cv2 as cv
import numpy as np


# 图片区域显示
def rectangle_space(img, x1, x2, y1, y2):
    # img = cv.imread(r"F:\python\opencv-4.x\samples\data\starry_night.jpg")
    # cv.imshow("img", img)
    # img2 = img[0:256, 256:512, 1:2]  # 第一个0:256表示高度所在位置，第二个0:256表示宽度所在位置，第三个1:2表示输出通道数
    # cv.imshow("rectangle_space", img2)
    cv.imshow("rectangle_space", img[y1:y2, x1:x2, 0:3])

    cv.waitKey(1)


# 除了截图部分其余均变暗
def rectangle_dark(img, x1, x2, y1, y2):
    # img = cv.imread(r"F:\python\opencv-4.x\samples\data\starry_night.jpg")
    # cv.imshow("img", img)
    # img2 = np.zeros_like(img)
    # img2[:, :, :] = (np.uint8(60), np.uint8(60), np.uint8(60))
    # img2[0:256, 256:512, :] = 0
    # result = cv.subtract(img, img2)
    # cv.imshow("result", result)
    img2 = np.zeros_like(img)
    img2[:, :, :] = (np.uint8(60), np.uint8(60), np.uint8(60))
    img2[y1:y2, x1:x2, :] = 0
    result = cv.subtract(img, img2)
    cv.imshow("mouse_demo", result)

    cv.waitKey(1000)


# 当前程序执行部分
if __name__ == '__main__':
    print("Hello world.")
    rectangle_dark()