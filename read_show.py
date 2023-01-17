import cv2
import cv2 as cv
import numpy as np
import math
import time
from matplotlib import pyplot as plt
import test


# 1.打印opencv版本号
def show_version():
    print(cv.__version__)


# 2.读取本项目目录下的一张图片
def show_image_local():  # 打印当前项目文件里的图片
    image = cv.imread("../data/primary_opencv/cat.jpg")  # BGR的读取顺序
    image = cv.resize(image, None, fx=0.2, fy=0.2)  # 调整图片大小
    cv.imshow("cat", image)
    cv.waitKey(1000)
    cv.destroyAllWindows()  # 销毁该程序创建的所有窗口


# 3.读取和展示图片
def show_image():  # 打印电脑其他位置的图片
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")  # BGR的读取顺序，路径不要写中文
    # image = cv.resize(image, None, fx=0.2, fy=0.2)  # 调整图片大小
    cv.imshow("lena", image)
    cv.waitKey(1000)
    cv.destroyAllWindows()  # 销毁该程序创建的所有窗口


# 4.图像色彩转换
def color_space_demo():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    cv.imshow("lena", image)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    gray = cv.cvtColor(image, 6)  # 相当于cv.COLOR_BGR2GRAY
    image2 = cv.cvtColor(gray, 8)  # 相当于cv.COLOR_GRAY2BGR
    # cv.imshow("hsv", hsv)
    # cv.imshow("ycrcb", ycrcb)
    cv.imshow("gray", gray)
    cv.imshow("image2", image2)
    print("image's shape:", image.shape)
    print("gray's shape:", gray.shape)
    print("image2's shape:", image2.shape)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 5.创建一个numpy array的数组
def make_numpy():
    m = np.zeros((3, 3, 3), dtype=np.uint8)
    print(m)
    m[:] = 255
    print(m)
    m[:] = (255, 0, 0)
    print(m)


# 6.自己创建一些numpy array并且输出图片
def make_numpy_show():
    m = np.zeros((512, 512, 3), dtype=np.uint8)
    m.shape  # 分别是H W C
    m[:] = 255
    cv.imshow("m1", m)
    n = np.zeros_like(m)  # 用m的大小去创造一个n，并且n中元素全为0
    # print(n)
    cv.imshow("n", n)
    n[:256] = (255, 0, 0)
    cv.imshow("n2", n)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 7.尝试给画布左右分隔添加颜色
def try_color():
    m = np.zeros((512, 512, 3), dtype=np.uint8)
    print(m.shape)  # 分别是H W C
    m[:256] = (255, 0, 0)
    m[0:256, 0:256] = (0, 0, 255)
    print(m)
    cv.imshow("m1", m)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 8.像素读写操作
def visit_pixel_demo():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    cv.imshow("lena", image)
    h, w, c = image.shape
    print("h: ", h, "w: ", w, "c:", c)
    print(image.dtype)
    for row in range(h):
        for col in range(w):
            b, g, r = image[row, col]
            image[row, col] = (255 - b, 255 - g, 255 - r)  # 取反色，把原图读取出来的BGR，通过255-，变成反色。
    cv.imshow("visited", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 9.在图片右上角添加颜色块
def visit_pixel_demo2():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    cv.imshow("lena", image)
    h, w, c = image.shape
    print("h: ", h, "w: ", w, "c:", c)
    print(image.dtype)
    image[0:256, 256:512] = (0, 0, 255)  # 在右上角添加了一个红色块
    cv.imshow("visited", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 10.算术操作实现加、减、乘、除
def arithmetic_demo():
    image1 = cv.imread(r"F:\python\opencv-4.x\samples\data\opencv-logo.png")
    image2 = np.zeros_like(image1)
    image2[:, :] = (110, 0, 250)
    image1 = cv.resize(image1, None, fx=0.5, fy=0.5)  # 调整图片大小
    image2 = cv.resize(image2, None, fx=0.5, fy=0.5)  # 调整图片大小
    cv.imshow("img1", image1)
    cv.imshow("img2", image2)
    added = cv.add(image1, image2)  # 加法
    subbed = cv.subtract(image1, image2)  # 减法
    multiplied = cv.multiply(image1, image2)  # 乘法
    divided = cv.divide(image1, image2)  # 除法
    cv.imshow("added", added)
    cv.imshow("subbed", subbed)
    cv.imshow("multiplied", multiplied)
    cv.imshow("divided", divided)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 11.使用mask实现的算术（加法、减法）操作实现加法
def arithmetic_demo_mask():
    image1 = cv.imread(r"F:\python\opencv-4.x\samples\data\opencv-logo.png")
    image2 = np.zeros_like(image1)
    image2[:, :] = (110, 0, 250)
    image1 = cv.resize(image1, None, fx=0.5, fy=0.5)  # 调整图片大小
    image2 = cv.resize(image2, None, fx=0.5, fy=0.5)  # 调整图片大小
    cv.imshow("img1", image1)
    cv.imshow("img2", image2)
    h, w, c = image1.shape
    print("h: ", h, "w: ", w, "c:", c)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[100:200, 100:250] = 1
    added = cv.add(image1, image2, mask=mask)  # 使用mask参数对[100:200, 100:250]部分进行add操作，其余位置补0
    subbed = cv.subtract(image2, image1, mask=mask)  # 使用mask参数对[100:200, 100:250]部分进行sub操作，其余位置补0
    cv.imshow("added", added)
    cv.imshow("subbed", subbed)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 12.创建一个滚动条调整亮度
def trackbar_callback(pos):  # trackbar_callback函数里可以什么都不做，但是必须有这个函数
    print(pos)


def trackbar_demo():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\butterfly.jpg")
    cv.namedWindow("trackbar_demo", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("lightness", "trackbar_demo", 0, 200, trackbar_callback)  # callback先注册后使用
    cv.imshow("trackbar_demo", image)
    while True:
        pos = cv.getTrackbarPos("lightness", "trackbar_demo")
        image2 = np.zeros_like(image)
        # image2是个常量，通过原图片和image2做加减法来提升和降低亮度
        image2[:, :] = (np.uint8(pos), np.uint8(pos), np.uint8(pos))
        # 提升亮度
        result = cv.add(image, image2)
        # 降低亮度
        # result = cv.subtract(image, image2)
        cv.imshow("trackbar_demo", result)
        # 1ms获取一次键值，默认是 - 1，ESC是27
        c = cv.waitKey(1)
        if c == 27:  # 按ESC建终止调整亮度功能
            break
    cv.waitKey(0)  # 按任意建关闭窗口
    cv.destroyAllWindows()


#  定义一个返回键值的函数
#  按ESC退出程序
#  按1显示HSV图像
#  按2显示YCrCb图像
#  按3显示RGB图像
#  按0恢复原图BGR显示
def keyboard_demo():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\butterfly.jpg")
    cv.namedWindow("keyboard_demo", cv.WINDOW_AUTOSIZE)
    cv.imshow("keyboard_demo", image)
    while True:
        c = cv.waitKey(10)  # 停顿10ms
        # ESC
        if c == 27:
            break
        # key = 0
        elif c == 48:
            cv.imshow("keyboard_demo", image)
        # key = 1
        elif c == 49:
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            cv.imshow("keyboard_demo", hsv)
        # key = 2
        elif c == 50:
            ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
            cv.imshow("keyboard_demo", ycrcb)
        # key = 3
        elif c == 51:
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            cv.imshow("keyboard_demo", rgb)
        else:
            if c != -1:
                print("Key: ", c, "is not define.")
    cv.waitKey(0)
    cv.destroyAllWindows()


# 13.自定义颜色查找表与系统自带的颜色查找表
def lut_demo():
    cv.namedWindow("lut-demo", cv.WINDOW_NORMAL)
    # 构建一个查找表，lut数组是一些随机的颜色
    lut = [[255, 0, 255], [125, 0, 0], [127, 255, 200], [200, 127, 127], [0, 255, 255]]
    m1 = np.array([[2, 1, 3, 0], [2, 2, 1, 1], [3, 3, 4, 4], [4, 4, 1, 1]])
    m2 = np.zeros((4, 4, 3), dtype=np.uint8)
    # 用索引和颜色做对应
    for i in range(4):
        for j in range(4):
            index = m1[i, j]
            m2[i, j] = lut[index]
    # 按照索引输出对应颜色
    cv.imshow("lut-demo", m2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 读取图片
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\butterfly.jpg")
    cv.imshow("input", image)
    cv.namedWindow("butterfly-gamma", cv.WINDOW_AUTOSIZE)

    # 1.建立查找表，直接使用索引对应颜色表示图片
    lut2 = np.zeros((256), dtype=np.uint8)
    gamma = 0.7  # 假定gamma为0.7
    for i in range(256):
        if i == 0:  # 这个是因为i = 0的时候，log(o/255.0)不存在，所以我自己找了个-6.0代替了一下
            print(i, "---", -6.0)
            lut2[i] = int(np.exp(-6.0 * gamma) * 255.0)  # gamma校正的公式
        else:
            print(i, "---", np.log(i / 255.0))
            lut2[i] = int(np.exp(np.log(i / 255.0) * gamma) * 255.0)  # gamma校正的公式
    print(lut2)
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            b, g, r = image[row, col]
            image[row, col] = (lut2[b], lut2[g], lut2[r])  # 直接用查找表的索引表示图片
    cv.imshow("butterfly-gamma", image)
    cv.waitKey(0)

    # 2.自定义颜色查找表
    # 注意自己定义颜色查找表的时候第一个维度永远是256，表示总共有0-255种颜色
    # 第二个维度是宽度只有1列，因为每个位置上面有一个值就可以了
    # 第三个维度表示通道数，可以是三通道也可以是单通道
    lut3 = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        if i == 0:  # 这个是因为i = 0的时候，log(o/255.0)不存在，所以我自己找了个-6.0代替了一下
            print(i, "---", -6.0)
            c = int(np.exp(-6.0 * gamma) * 255.0)  # gamma校正的公式
        else:
            print(i, "---", np.log(i / 255.0))
            c = int(np.exp(np.log(i / 255.0) * gamma) * 255.0)  # gamma校正的公式
        lut3[i, 0] = (c, c, c)
    print(lut3)
    dst = cv.LUT(image, lut3)  # 使用自定义的查找表
    cv.imshow("butterfly-gamma", dst)
    cv.waitKey(0)

    # 3.使用系统自带的COLORMAP_PINK颜色查找表
    dst = cv.applyColorMap(image, cv.COLORMAP_PINK)
    cv.imshow("butterfly-pink", dst)
    cv.waitKey(0)

    # 4.使用系统自带的COLORMAP_JET颜色查找表
    dst = cv.applyColorMap(image, cv.COLORMAP_JET)
    cv.imshow("butterfly-jet", dst)
    cv.waitKey(0)

    cv.destroyAllWindows()


# 14.用滚动条做系统自带的颜色查找表
def trackbar_lut_callback(pos):
    print(pos)


def trackbar_lut_demo():
    arr = [0, 1, 17, 8, 21, 11, 9, 14, 2, 13, 5, 12, 10, 15, 4, 7, 6, 20, 18, 19]
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\butterfly.jpg")
    cv.namedWindow("trackbar_lut_demo", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("colormap", "trackbar_lut_demo", 0, 21, trackbar_lut_callback)  # callback先注册后使用
    cv.imshow("trackbar_lut_demo", image)
    while True:
        pos = cv.getTrackbarPos("colormap", "trackbar_lut_demo")
        # 颜色查找表
        if pos in arr:
            dst = cv.applyColorMap(image, pos)
            cv.imshow("trackbar_lut_demo", dst)
        else:
            cv.imshow("trackbar_lut_demo", image)
        # 1ms获取一次键值，默认是 - 1，ESC是27
        c = cv.waitKey(1)
        if c == 27:  # 按ESC建终止调整亮度功能
            break
    cv.waitKey(0)  # 按任意建关闭窗口
    cv.destroyAllWindows()


# 15.通道分离与合并
def channel_splits():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\butterfly.jpg")
    cv.namedWindow("butterfly", cv.WINDOW_AUTOSIZE)
    cv.imshow("butterfly", image)

    # 通道分割
    mv = cv.split(image)
    cv.imshow("B", mv[0])
    cv.imshow("G", mv[1])
    cv.imshow("R", mv[1])
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 通道合并
    mv2 = cv.merge(mv)
    cv.imshow("merge", mv2)
    mv[1][:, :] = 255  # 修改其中一个通道的颜色为全白
    mv3 = cv.merge(mv)
    cv.imshow("merge_1", mv3)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # BGR2RGB
    dst = np.zeros_like(image)
    cv.mixChannels([image], [dst], fromTo=[0, 1, 2, 2, 1, 0])  # (0,1,2)->(2,1,0)
    cv.imshow("mix_channels", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 通道阈值
    # mask = cv.inRange(image, (43, 46, 100), (128, 200, 200))  #阈值范围为(43, 46, 100), (128, 200, 200)
    mask = cv.inRange(image, (20, 46, 80), (128, 230, 180))  # 阈值范围为(20, 46, 80), (128, 230, 180)
    cv.imshow("inRange", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 16.BGR2HSV + inRange
def bgr2rgb_inrange():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\butterfly.jpg")
    cv.namedWindow("butterfly", cv.WINDOW_AUTOSIZE)
    cv.imshow("butterfly", image)

    # BGR2HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # inRange
    mask = cv.inRange(hsv, (43, 46, 100), (128, 200, 200))
    mask2 = cv.inRange(hsv, (20, 46, 80), (128, 230, 180))
    cv.imshow("bgr2rgb_inrange_1", mask)
    cv.imshow("bgr2rgb_inrange_2", mask2)
    cv. waitKey(0)
    cv.destroyAllWindows()


# 17.图像像素统计 -- 均值、方差、极值（最大、最小）
def stats_demo_1():
    # 计算均值和方差
    roi = np.array([[5, 3, 4], [9, 6, 7], [8, 2, 3]], dtype=np.uint8)  # 定义一个3*3的数组
    mask = np.array([[0, 3, 0], [0, 6, 0], [0, 2, 0]], dtype=np.uint8)  # 定义一个mask区域
    m1 = cv.meanStdDev(roi)  # 计算全图均值，方差
    m2 = cv.meanStdDev(roi, mask=mask)  # 计算mask区域的均值，方差
    minx, maxx, minx_loc, max_loc = cv.minMaxLoc(roi)  # 计算最小值，最大值，最小值坐标，最大值坐标
    print("roi:\n", roi, "\n", "mask:\n", mask)
    print("m1:", m1, "\n", "m2: ", m2)
    print("min: ", minx, " max: ", maxx, " min_loc: ", minx_loc, " max_loc: ", max_loc)
    # 计算均值
    m3 = cv.mean(roi)  # 计算全图均值
    m4 = cv.meanStdDev(roi, mask=mask)  # 计算mask区域的均值和方差
    print("roi:\n", roi, "\n", "mask:\n", mask)
    print("m3: ", m3, "\n", "m4: ", m4)


# 18.图像像素统计 -- 改变对比度 + 均值
def stats_demo_2():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\butterfly.jpg")
    cv.namedWindow("butterfly", cv.WINDOW_AUTOSIZE)
    cv.imshow("butterfly", image)
    # 计算全图均值
    bgr_m = cv.mean(image)
    # 对原图像设置低对比度
    sub_m = np.float32(image)[:, :] - (bgr_m[0], bgr_m[1], bgr_m[2])
    result = sub_m * 0.5  # 提升差值
    result = result[:, :] + (bgr_m[0], bgr_m[1], bgr_m[2])  # 提升差值之后还要把均值加上去
    cv.imshow("low-contrast-butterfly", cv.convertScaleAbs(result))  # convertScaleAbs转换为绝对值，然后转成CV_8UC
    # 对原图像设置高对比度
    result2 = sub_m * 2.0  # 提升差值
    result2 = result2[:, :] + (bgr_m[0], bgr_m[1], bgr_m[2])  # 提升差值之后还要把均值加上去
    cv.imshow("high-contrast-butterfly", cv.convertScaleAbs(result2))  # convertScaleAbs转换为绝对值，然后转成CV_8UC
    # 输出不同对比度下的图片均值
    m1 = cv.mean(image)
    m2 = cv.mean(cv.convertScaleAbs(result))
    m3 = cv.mean(cv.convertScaleAbs(result2))
    print("image:", m1)
    print("result_low:", m2)
    print("result_high:", m3)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 19.图像几何形状绘制
def draw_demo_1():
    # 创建一个512*512*3大小的图像作为画布
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    # (100, 100)是起点坐标，(300, 300)是终点坐标
    # 绘制一个红色矩形
    cv.rectangle(canvas, (100, 100), (300, 300), (0, 0, 255), 2, 8)
    # 填充一个紫色矩形
    cv.rectangle(canvas, (400, 100), (450, 150), (255, 0, 255), -1, 8)
    # 绘制一个蓝色圆形
    cv.circle(canvas, (250, 250), 50, (255, 0, 0), 2, cv.LINE_8)
    # 填充一个蓝色圆形
    cv.circle(canvas, (425, 200), 20, (255, 0, 0), -1, cv.LINE_8)
    # 绘制一个绿色线段，lineType=8指的是8联通线型，涉及到线的产生算法，另一种是lineType=4指的是4联通线型
    cv.line(canvas, (100, 100), (300, 300), (0, 255, 0), 2, 8)
    # 添加一个文本
    cv.putText(canvas, "OpenCV-Python", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
    cv.imshow("canvas", canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 20.动态显示文本区域
def draw_demo_2():
    # 创建一个512*512*3大小的图像作为画布
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    # 深度学习对象检测经典颜色:土耳其蓝(140, 199, 0)
    font_color = (140, 199, 0)
    cv.rectangle(canvas, (100, 100), (300, 300), font_color, 2, 8)

    label_txt = "OpenCV-Python"
    label_txt2 = "Hello world is a nice sentence."
    font = cv.FONT_HERSHEY_SIMPLEX  # 字体选择建议就是cv.FONT_HERSHEY_SIMPLEX 或者 cv.FONT_HERSHEY_PLAIN
    font_scale = 0.5  # 字体大小0.5
    thickness = 1  # 线宽1
    # cv.getTextSize动态获取文本，(fw, uph)是宽、高，dh是基线
    (fw, uph), dh = cv.getTextSize(label_txt, font, font_scale, thickness)
    (fw2, uph2), dh2 = cv.getTextSize(label_txt2, font, font_scale, thickness)
    cv.rectangle(canvas, (100, 80-uph-dh), (100+fw, 80), (255, 255, 255), -1, 8)
    cv.rectangle(canvas, (100, 100-uph2-dh2), (100+fw2, 100), (255, 255, 255), -1, 8)
    cv.putText(canvas, label_txt, (100, 80-dh), font, font_scale, (255, 0, 255), thickness)
    cv.putText(canvas, label_txt2, (100, 100-dh), font, font_scale, (255, 0, 255), thickness)
    cv.imshow("canvas", canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 21.随机颜色+随机噪声
def random_demo_1():
    # 设置512*512*3的图像作为画布
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    # 随即绘制
    while True:
        # 随机的颜色，size 随机数的尺寸，size=3表示返回三个随机数
        b, g, r = np.random.randint(0, 256, size=3)
        # 随机坐标起点(x1, y1)，终点（x2, y2）
        x1 = np.random.randint(0, 512)
        x2 = np.random.randint(0, 512)
        y1 = np.random.randint(0, 512)
        y2 = np.random.randint(0, 512)
        cv.rectangle(canvas, (x1, y1), (x2, y2), (int(b), int(g), int(r)), -1, 8)
        cv.imshow("canvas", canvas)
        # 50ms获取一次键盘值，默认是-1，ESC是27
        c = cv.waitKey(50)
        if c == 27:
            break
        # 重新绘制背景画布
        cv.rectangle(canvas, (0, 0), (512, 512), (0, 0, 0), -1, 8)  # 擦除之前所绘制的内容

    # 随机产生噪声图片
    # cv.randn(canvas, (40, 200, 140), (10, 50, 10))
    cv.randn(canvas, (120, 100, 140), (30, 50, 20))
    cv.imshow("noise image", canvas)

    cv.waitKey(0)
    cv.destroyAllWindows()


# 22.给图片添加噪声
def random_demo_2():
    # 设置512*512*3的图像作为画布
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    # 给512*512*3的图片添加噪声
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    # 定义三个随机数
    while True:
        n1 = np.random.randint(0, 100)
        n2 = np.random.randint(0, 100)
        n3 = np.random.randint(0, 100)
        # 将随机数作为噪声的均值和方差
        cv.randn(canvas, (n1, n2, n3), (n1, n2, n3))
        # 在原图中添加噪声
        dst = cv.add(image, canvas)
        cv.imshow("add noise image", dst)
        # 1000ms即1s获取一次键盘值，默认是-1，ESC是27
        c = cv.waitKey(1000)
        if c == 27:
            break
        # 重新绘制背景画布
        cv.rectangle(canvas, (0, 0), (512, 512), (0, 0, 0), -1, 8)  # 擦除之前所绘制的内容

    cv.waitKey(0)
    cv.destroyAllWindows()


# 23.多边形绘制
def poly_demo():
    # 设置画布
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    # pts = [(100, 100), (200, 50), (280, 100), (290, 300), (50, 300)]
    pts = []
    pts.append((100, 100))
    pts.append((200, 50))
    pts.append((280, 100))
    pts.append((290, 300))
    pts.append((50, 300))
    pts = np.array(pts, dtype=np.int32)
    print(pts.shape)

    pts2 = []
    pts2.append((300, 300))
    pts2.append((400, 250))
    pts2.append((500, 300))
    pts2.append((500, 500))
    pts2.append((250, 500))
    pts2 = np.array(pts2, dtype=np.int32)
    print(pts2.shape)

    # 同时绘制两个点集
    cv.polylines(canvas, [pts, pts2], True, (0, 0, 255), 2, 8)
    # 填充
    cv.fillPoly(canvas, [pts, pts2], (255, 0, 0), 8, 0)
    cv.imshow("poly-demo", canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 24.鼠标操作绘制矩形
b1 = cv.imread(r"F:\python\opencv-4.x\samples\data\starry_night.jpg")
img = np.copy(b1)
# (x1, y1)表示左上角，（x2, y2）表示右下角点
x1 = -1
x2 = -1
y1 = -1
y2 = -1


# 25.定义绘制矩形的注册函数
def mouse_drawing_rectangle(event, x, y, flags, parm):
    # 全局参数
    global x1, y1, x2, y2
    # 鼠标放下，赋值左上角点给x1，y1
    if event == cv.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
    # 鼠标移动
    if event == cv.EVENT_MOUSEMOVE:
        # x1，y1初始值都是-1，如果移动过程<0说明鼠标没有摁下
        if x1 < 0 or y1 < 0:
            return
        x2 = x
        y2 = y
        dx = x2 - x1
        dy = y2 - y1
        # 移动有一定距离才会绘制
        if dx > 0 and dy > 0:
            # 矩形绘制到b1（读入的图片）上
            # img是原图
            b1[:, :, :] = img[:, :, :]  # 用原图覆盖擦除之前的绘制结果
            cv.putText(b1, "searching...", (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv.rectangle(b1, (x1, y1), (x2, y2), (255, 0, 255), 2, 8, 0)  # 移动过程中用紫色线
            # 写在test脚本中的api函数
            # test.rectangle_space(img, x1, x2, y1, y2)  # 实时响应截取
            # test.rectangle_dark(img, x1, x2, y1, y2)  # 实时截取加灰色背景
    if event == cv.EVENT_LBUTTONUP:
        x2 = x
        y2 = y
        dx = x2 - x1
        dy = y2 - y1
        if dx > 0 and dy > 0:
            # 矩形绘制到b1（读入的图片）上
            # img是原图
            b1[:, :, :] = img[:, :, :]  # 用原图覆盖擦除之前的绘制结果
            cv.putText(b1, "Finished", (x1, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.rectangle(b1, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)  # 鼠标抬起之后用红色线
            # 使用自己文件目录下的脚本文件定义截图显示程序
        # 重新赋值为下一次绘制做准备
        x1 = -1
        y1 = -1
        x2 = -1
        y2 = -1


# 26.鼠标操作绘制圆形
b2 = cv.imread(r"F:\python\opencv-4.x\samples\data\starry_night.jpg")
img2 = np.copy(b2)
# (c1, c2)表示圆心坐标，r1表示半径
c1 = -1
c2 = -1


# 27.定义圆形的注册函数
def mouse_drawing_circle(event, x, y, flags, parm):
    # 全局参数
    global c1, c2, r1
    # 鼠标放下，赋值左上角点给x1，y1
    if event == cv.EVENT_LBUTTONDOWN:
        c1 = x
        c2 = y
    # 鼠标移动
    if event == cv.EVENT_MOUSEMOVE:
        # c1，c2初始值都是-1，如果移动过程<0说明鼠标没有摁下
        if c1 < 0 or c2 < 0:
            return
        dr = int(math.sqrt(pow((x-c1), 2) + pow((y-c2), 2)))
        # 移动有一定距离才会绘制
        if dr > 0:
            # 圆形绘制到b1（读入的图片）上
            # img是原图
            b2[:, :, :] = img2[:, :, :]  # 用原图覆盖擦除之前的绘制结果
            cv.circle(b2, (c1, c2), dr, (255, 0, 255), 2, cv.LINE_8)  # 移动过程中用紫色线
    if event == cv.EVENT_LBUTTONUP:
        dr = int(math.sqrt(pow((x - c1), 2) + pow((y - c2), 2)))
        if dr > 0:
            # 圆形绘制到b1（读入的图片）上
            # img是原图
            b2[:, :, :] = img2[:, :, :]  # 用原图覆盖擦除之前的绘制结果
            cv.circle(b2, (c1, c2), dr, (0, 0, 255), 2, cv.LINE_8)  # 移动过程中用红色线
        # 重新赋值为下一次绘制做准备
        c1 = -1
        c2 = -1


def mouse_demo():
    cv.namedWindow("mouse_demo", cv.WINDOW_AUTOSIZE)
    # 实时关注mouse_demo画布上的响应，如果发生mouse_drawing中定义的事件，就返回响应
    # cv.setMouseCallback("mouse_demo", mouse_drawing_circle)  # 绘制圆形
    cv.setMouseCallback("mouse_demo", mouse_drawing_rectangle)  # 绘制矩形
    while True:
        # cv.imshow("mouse_demo", b2)  # 绘制圆形
        cv.imshow("mouse_demo", b1)  # 绘制矩形
        # 每过10ms就获取一次键盘键值，默认是-1，ESC键是27
        c = cv.waitKey(10)
        if c == 27:
            break
    cv.destroyAllWindows()


# 28.图像像素类型转换与归一化
def norm_demo():
    image_uint8 = cv.imread(r"F:\python\opencv-4.x\samples\data\ml.png")
    cv.imshow("image_uint8", image_uint8)
    img_f32 = np.float32(image_uint8)
    cv.imshow("image_f32", img_f32)
    cv.normalize(img_f32, img_f32, 1, 0, cv2.NORM_MINMAX)
    cv.imshow("norm-img_f32", img_f32)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 29.定义滚动条的注册响应
def trackbar_norm_callback(pos):
    print(pos)


# 30.用滚动条实现四种归一化方式
def norm_trackbar_demo():
    image_uint8 = cv.imread(r"F:\python\opencv-4.x\samples\data\ml.png")
    cv.namedWindow("norm-demo", cv.WINDOW_AUTOSIZE)
    # 注意使用cv.createTrackbar滚动条之前线要定义注册响应trackbar_norm_callback
    cv.createTrackbar("normtype", "norm-demo", 0, 3, trackbar_norm_callback)
    while True:
        dst = np.float32(image_uint8)
        pos = cv.getTrackbarPos("normtype", "norm-demo")
        if pos == 0:
            cv.normalize(dst, dst, 1, 0, cv.NORM_MINMAX)
        if pos == 1:
            cv.normalize(dst, dst, 1, 0, cv.NORM_L1)
        if pos == 2:
            cv.normalize(dst, dst, 1, 0, cv.NORM_L2)
        if pos == 3:
            cv.normalize(dst, dst, 1, 0, cv.NORM_INF)
        cv.imshow("norm-demo", dst)
        # 每过50ms就获取一个键值，默认键值为-1，ESC键值为27
        c = cv.waitKey(50)
        if c == 27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


# 31.图像几何旋转
def affine_demo():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\ml.png")
    h, w, c = image.shape
    # 获取中心位置
    cx = int(w/2)
    cy = int(h/2)
    cv.imshow("image", image)
    # 定义原图放缩0.7倍，上下均平移50的矩阵M
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = .7
    M[1, 1] = .7
    M[0, 2] = 50
    M[1, 2] = 50
    print("M(2x3) = \n", M)
    dst = cv.warpAffine(image, M, (int(w*.7), int(h*.7)))
    cv.imshow("rescale-demo", dst)
    # 在指定路径写一个图片
    cv.imwrite(r"F:\python\opencv-4.x\result.png", dst)

    # 定义旋转，获取旋转矩阵degree>0，表示逆时针旋转，原点在左上角
    # (w/2, h/2)表示旋转中心,逆时针旋转45°，放缩1.0即不做放缩操作。
    M = cv.getRotationMatrix2D((w/2, h/2), 45.0, 1.0)
    dst = cv.warpAffine(image, M, (w, h))
    cv.imshow("rotate-demo", dst)

    # 图像翻转，0表示水平翻转
    dst = cv.flip(image, 0)
    cv.imshow("flip-demo", dst)

    # 图像特殊角度旋转，顺时针旋转90°
    dst = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    cv.imshow("rotate-90-demo", dst)

    cv.waitKey(0)
    cv.destroyAllWindows()


# 32.视频读写处理
def video_demo():
    # 设置为0表示调用默认摄像头
    cap = cv.VideoCapture(r"F:\python\opencv-4.x\samples\data\vtest.avi")
    # 获取视频帧率fps
    fps = cap.get(cv.CAP_PROP_FPS)
    # 获取每帧的宽度
    frame_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    # 获取每帧的高度
    frame_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print("fps:", fps, "frame_w", frame_w, "frame_h", frame_h)
    # 编码方式
    # 指定编码方式为vp09
    # fourcc = cv.VideoWriter_fourcc(*'vp09')
    # 计算机自动获取编码格式
    fourcc = cap.get(cv.CAP_PROP_FOURCC)
    # 注意将编码格式fourcc转换成int类型
    # 定义写入一个视频
    writer_mp4 = cv.VideoWriter('output.mp4', int(fourcc), fps, (int(frame_w), int(frame_h)))
    # 循环读取图片完成视频
    while True:
        # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
        ret, frame = cap.read()
        if ret is not True:
            break
        cv.imshow("frame", frame)
        # 间隔1ms播放下一帧
        c = cv.waitKey(1)
        if c == 27:
            break
        # 写入视频帧，注意初始的帧数要和实际返回帧数一致
        writer_mp4.write(frame)

    # 释放资源
    cap.release()
    writer_mp4.release()

    cv.waitKey(0)
    cv.destroyAllWindows()
    # 视频文件执行之后会有警告但是不影响使用


# 33.调用摄像头读写
def video_face_demo():
    # 程序执行开始时间
    a = time.time()
    print(a)
    # 读取视频文件
    cap = cv.VideoCapture(0)
    # # 更改分辨率大小和fps大小
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    # cap.set(cv2.CAP_PROP_FPS, 70)
    # resize = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    # print(resize)

    # 获取视频帧率fps
    fps = cap.get(cv.CAP_PROP_FPS)
    # 获取每帧的宽度
    frame_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    # 获取每帧的高度
    frame_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print("fps:", fps, "frame_w", frame_w, "frame_h", frame_h)
    # 记录调用时长
    print(time.time() - a)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    while True:
        # 获取每一帧的帧率
        fps = cap.get(cv.CAP_PROP_FPS)
        print(fps)
        # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
        ret, frame = cap.read()
        # 摄像头是和人对立的，将图像垂直翻转
        frame = cv2.flip(frame, 1)
        cv.imshow("video", frame)
        # 10ms显示一张图片
        c = cv.waitKey(10)
        if c == 27:
            break
    # 释放资源
    cap.release()
    cv.waitKey(0)
    cv.destroyAllWindows()
    # 视频文件执行之后会有警告但是不影响使用


# 34.图像直方图
def image_hist():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\ml.png")
    cv.imshow("input", image)
    color = ('blue', 'green', 'red')
    # enumerate遍历数组类型同时返回下标和对应数组值
    for i, color in enumerate(color):
        # 一共分32类，每一类256/32步长，按照B,G,R的通道顺序一次使用blue,green,red颜色绘制
        hist = cv.calcHist([image], [i], None, [32], [0, 255])
        print(hist.dtype)
        plt.plot(hist, color=color)
        plt.xlim([0, 32])
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


# 35.图像直方图均衡化
def image_eq_demo():
    # 直接读入灰度图片
    # image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg", cv.IMREAD_GRAYSCALE)
    # 读入RGB彩色图片，切分通道，取单通道
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    image = cv.split(image)
    cv.imshow("input", image[0])
    # 提取原图的B通道作图像直方图
    hist = cv.calcHist([image[0]], [0], None, [32], [0, 255])
    print(hist.dtype)
    plt.plot(hist, color="gray")
    # 灰度等级设定为256/32 = 8
    plt.xlim([0, 32])
    plt.show()

    eqimg = cv.equalizeHist(image[0])
    cv.imshow("eq", eqimg)
    # 确保均衡化的输入是一个八位(0,255)单通道图像
    hist = cv.calcHist([eqimg], [0], None, [32], [0, 255])
    print(hist.dtype)
    plt.plot(hist, color="gray")
    plt.xlim([0, 32])
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


# 36.图像卷积操作
def conv_demo():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    dst = np.copy(image)
    cv.imshow("input", image)
    h, w, c = image.shape
    # 自定义方法实现卷积
    # 从上到下，从左到右，从1开始说明边缘有一个没有绑定会有缝隙，从2开始就可以边缘填满
    for row in range(2, h-2, 1):
        for col in range(2, w-2, 1):
            # 求均值的区域范围是(0, 4)就是0, 1, 2, 3, 4，也就是一个5*5的卷积核
            m = cv.mean(image[row-2:row+2, col-2:col+2])
            # 把卷积后的均值结果赋值给中心位置
            dst[row, col] = (int(m[0]), int(m[1]), int(m[2]))
    cv.imshow("convolution-demo", dst)

    # 用官方自带的卷积api函数
    blured = cv.blur(image, (5, 5), anchor=(-1, -1))
    cv.imshow("blur-demo", blured)

    cv.waitKey(0)
    cv.destroyAllWindows()


# 37.窗口大小不是正方形的图像卷积
def conv_demo_2():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    cv.imshow("input", image)
    # 用官方自带的卷积api函数
    # 水平抖动
    blured1 = cv.blur(image, (15, 1), anchor=(-1, -1))
    cv.imshow("blur-demo1", blured1)
    # 垂直抖动
    blured2 = cv.blur(image, (1, 15), anchor=(-1, -1))
    cv.imshow("blur-demo2", blured2)
    # 模糊操作，系数相同，均值卷积
    blured3 = cv.blur(image, (25, 25), anchor=(-1, -1))
    cv.imshow("blur-demo3", blured3)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 38.高斯模糊
def gaussian_blur_demo():
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    cv.imshow("input", image)
    # 用官方自带的卷积api函数
    # 窗口大小ksize为0表示从sigmaX计算生成ksize
    # ksize窗口大小，必须是正数而且是奇数
    g1 = cv.GaussianBlur(image, (0, 0), 15)
    # ksize大于0表示从ksize计算生成sigmaX
    # 此时的sigmaX由计算式子σ = 0.3*((size - 1)*0.5 - 1) + 0.8 计算为：2.6
    # ksize窗口大小，必须是正数而且是奇数
    g2 = cv.GaussianBlur(image, (15, 15), 15)
    cv.imshow("GaussianBlur-demo1", g1)
    cv.imshow("GaussianBlur-demo2", g2)

    cv.waitKey(0)
    cv.destroyAllWindows()


# 39.滚动条实现像素重定向
def trackbar_remap_callback(pos):
    print(pos)


def remap_demo():
    cv.namedWindow("remap-demo", cv.WINDOW_AUTOSIZE)
    # 注意使用cv.createTrackbar滚动条之前线要定义注册响应trackbar_remap_callback
    cv.createTrackbar("remap-type", "remap-demo", 0, 3, trackbar_remap_callback)
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    cv.imshow("lena", image)
    h, w, c = image.shape
    # map_x和map_y只存储映射规则的坐标
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    # 完成map_x和map_y的定义
    while True:
        pos = cv.getTrackbarPos("remap-type", "remap-demo")
        # 倒立,x方向不变，修改y方向
        if pos == 0:
            # map_x.shape[0]对应h
            # 一行一行的操作x
            for i in range(map_x.shape[0]):
                map_x[i, :] = [x for x in range(map_x.shape[1])]
            # map_y.shape[1]对应w
            # 一列一列的操作y
            for j in range(map_y.shape[1]):
                map_y[:, j] = [map_y.shape[0] - y for y in range(map_y.shape[0])]
        # 镜像，x方向修改，y方向不变
        elif pos == 1:
            # map_x.shape[0]对应h
            # 一行一行的操作x
            for i in range(map_x.shape[0]):
                map_x[i, :] = [map_x.shape[1] - x for x in range(map_x.shape[1])]
            # map_y.shape[1]对应w
            # 一列一列的操作y
            for j in range(map_y.shape[1]):
                map_y[:, j] = [y for y in range(map_y.shape[0])]
        # 对角线对称，x方向修改，y方向也修改
        elif pos == 2:
            # map_x.shape[0]对应h
            # 一行一行的操作x
            for i in range(map_x.shape[0]):
                map_x[i, :] = [map_x.shape[1] - x for x in range(map_x.shape[1])]
            # map_y.shape[1]对应w
            # 一列一列的操作y
            for j in range(map_y.shape[1]):
                map_y[:, j] = [map_y.shape[0] - y for y in range(map_y.shape[0])]
        # 放大两倍，x方向修改，y方向也修改
        elif pos == 3:
            # map_x.shape[0]对应h
            # 一行一行的操作x
            for i in range(map_x.shape[0]):
                map_x[i, :] = [int(x/2) for x in range(map_x.shape[1])]
            # map_y.shape[1]对应w
            # 一列一列的操作y
            for j in range(map_y.shape[1]):
                map_y[:, j] = [int(y/2) for y in range(map_y.shape[0])]
        # 像素重映射remap函数
        dst = cv.remap(image, map_x, map_y, cv.INTER_LINEAR)
        cv.imshow("remap-demo", dst)
        # 每50ms获取一次键值，ESC键值为27
        c = cv.waitKey(50)
        if c == 27:
            break
    cv.destroyAllWindows()


# 40.图像二值化
def binary_demo():
    # 彩色图像变成灰度图像
    image = cv.imread("../data/primary_opencv/bin_test.png")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)

    # 手动定义阈值，假设为120，做二值化
    # ret表示返回分割阈值，binary表示返回的二值图像
    ret, binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
    print("ret-myself: ", ret)
    cv.imshow("binary-myself", binary)

    # 用均值作为阈值，做二值化
    # mean的结果是四通道的数，因为灰度图像只有一个通道所以只取[0]即可
    m = cv.mean(gray)[0]
    print("m-mean: ", m)
    # ret表示返回分割阈值，binary表示返回的二值图像
    ret, binary = cv.threshold(gray, m, 255, cv.THRESH_BINARY)
    print("ret-mean: ", ret)
    cv.imshow("binary-mean", binary)

    cv.waitKey(0)
    cv.destroyAllWindows()


# 41.全局与自适应二值化
def binary_demo_2():
    # 彩色图像变成灰度图像
    image = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)

    # 手动阈值，大津法
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("ret-OTSU: ", ret)
    cv.imshow("binary-OTSU", binary)
    cv.waitKey(0)

    # 手动阈值，三角法
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print("ret-TRIANGLE: ", ret)
    cv.imshow("binary-TRIANGLE", binary)
    cv.waitKey(0)

    # 高斯自适应法
    # 高斯自适应背景是白色
    # blocksize必须为奇数，C表示要减去的权重，可以是正数，负数，0，通常取25，10效果比较好
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow("binary-gaussian-white", binary)
    # 高斯自适应背景修改为黑色
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 10)
    cv.imshow("binary-gaussian-black", binary)
    cv.waitKey(0)

    cv.destroyAllWindows()


# 42.人脸识别需要的文件
model_bin ="../data/opencv_face_detector_uint8.pb"
config_text = "../data/opencv_face_detector.pbtxt"


# 43.实时人脸识别摄像头
def video_face_demo():
    # 记录开始时间
    a = time.time()
    print(a)
    # 获取摄像头
    cap = cv.VideoCapture(0)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    # 部署tensorflow模型
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    # 记录调用时长
    print(time.time() - a)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    while True:
        e1 = cv.getTickCount()
        # 获取每一帧的帧率
        fps = cap.get(cv.CAP_PROP_FPS)
        print(fps)
        # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
        ret, frame = cap.read()
        # 摄像头是和人对立的，将图像垂直翻转
        frame = cv.flip(frame, 1)
        if ret is not True:
            break
        h, w, c = frame.shape
        print("h:", h, "w: ", w, "c: ", c)
        # 模型输入:1x3x300x300
        # 1.0表示不对图像进行缩放，设定图像尺寸为(300, 300)，减去一个设定的均值(104.0, 177.0, 123.0)，是否交换BGR通道和是否剪切都选False
        blobimage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blobimage)
        # forward之后，模型输出:1xNx7
        cvout = net.forward()
        print(cvout.shape)

        t, _ = net.getPerfProfile()
        # 推理时间
        label = "Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency())
        # 绘制检测矩形
        # 只考虑后五个参数
        for detection in cvout[0, 0, :]:
            # 获取置信度
            score = float(detection[2])
            objindex = int(detection[1])
            # 置信度>0.5说明是人脸
            if score > 0.5:
                # 获取实际坐标
                left = detection[3] * w
                top = detection[4] * h
                right = detection[5] * w
                bottom = detection[6] * h

                # 绘制矩形框
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)

                # 绘制类别跟得分
                # 置信度
                label_txt = "score:%.2f" % score
                # 获取文本的位置和基线
                (fw, uph), dh = cv.getTextSize(label_txt, font, font_scale, thickness)
                cv.rectangle(frame, (int(left), int(top) - uph - dh), (int(left) + fw, int(top)), (255, 255, 255), -1, 8)
                cv.putText(frame, label_txt, (int(left), int(top) - dh), font, font_scale, (255, 0, 255), thickness)

        e2 = cv.getTickCount()
        fps = cv.getTickFrequency() / (e2 - e1)
        # 帧率
        cv.putText(frame, label + (" FPS: %.2f" % fps), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv.imshow("face-dectection-demo", frame)
        # 10ms显示一张图片
        c = cv.waitKey(10)
        if c == 27:
            break
        # 释放资源
    cap.release()
    cv.waitKey(0)
    cv.destroyAllWindows()
    # 视频文件执行之后会有警告但是不影响使用


# 44.识别一张图片中的人脸
def frame_face_demo():
    # 记录开始时间
    a = time.time()
    print(a)
    # 获取摄像头
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    # 部署tensorflow模型
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    # 记录调用时长
    print(time.time() - a)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    e1 = cv.getTickCount()
    # 摄像头是和人对立的，将图像垂直翻转
    frame = cv.imread(r"F:\python\opencv-4.x\samples\data\lena.jpg")
    h, w, c = frame.shape
    print("h:", h, "w: ", w, "c: ", c)
    # 模型输入:1x3x300x300
    # 1.0表示不对图像进行缩放，设定图像尺寸为(300, 300)，减去一个设定的均值(104.0, 177.0, 123.0)，是否交换BGR通道和是否剪切都选False
    blobimage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blobimage)
    # forward之后，模型输出:1xNx7
    cvout = net.forward()
    print(cvout.shape)

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency())
    # 绘制检测矩形
    # 只考虑后五个参数
    for detection in cvout[0, 0, :]:
        # 获取置信度
        score = float(detection[2])
        objindex = int(detection[1])
        # 置信度>0.5说明是人脸
        if score > 0.5:
            # 获取实际坐标
            left = detection[3] * w
            top = detection[4] * h
            right = detection[5] * w
            bottom = detection[6] * h

            # 绘制矩形框
            cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)

            # 绘制类别跟得分
            label_txt = "score:%.2f" % score
            # 获取文本的位置和基线
            (fw, uph), dh = cv.getTextSize(label_txt, font, font_scale, thickness)
            cv.rectangle(frame, (int(left), int(top) - uph - dh), (int(left) + fw, int(top)), (255, 255, 255), -1, 8)
            cv.putText(frame, label_txt, (int(left), int(top) - dh), font, font_scale, (255, 0, 255), thickness)

    e2 = cv.getTickCount()
    fps = cv.getTickFrequency() / (e2 - e1)
    cv.putText(frame, label + (" FPS: %.2f" % fps), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv.imshow("face-dectection-demo", frame)

    # 释放资源
    cv.waitKey(0)
    cv.destroyAllWindows()
    # 视频文件执行之后会有警告但是不影响使用


# 函数执行部分 ctrl+左键可以跳转到对应函数部分

# show_version()
# show_image()
# color_space_demo()
# make_numpy()
# make_numpy_show()
# try_color()
# visit_pixel_demo()
# visit_pixel_demo2()
# arithmetic_demo()
# arithmetic_demo_mask()
# trackbar_demo()
# keyboard_demo()
# lut_demo()
# trackbar_lut_demo()
# channel_splits
# bgr2rgb_inrange()
# stats_demo_1()
# stats_demo_2()
# draw_demo_1()
# draw_demo_2()
# random_demo_1()
# random_demo_2()
# poly_demo()
# mouse_demo()
# norm_demo()
# norm_trackbar_demo()
# affine_demo()
# video_demo()
# video_face_demo()
# image_hist()
# image_eq_demo()
# conv_demo()
# conv_demo_2()
# gaussian_blur_demo()
# remap_demo()
# binary_demo()
# binary_demo_2()
# video_face_demo()
# frame_face_demo()


# 45.程序执行部分
if __name__ == '__main__':  # 在当前.py文件中执行的部分
    print(cv.__version__)
    mouse_demo()




