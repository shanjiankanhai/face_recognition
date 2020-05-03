"""
# 椭圆肤色模型
# 用于识别adaboost、算法无法识别的地方
# 先识别脸部区域，再使用adaboost识别眼睛区域
"""
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
from outline import outline_cut

eye_path = r'haarcascade_eye_tree_eyeglasses.xml'
eye_cascade = cv2.CascadeClassifier(eye_path)

# 构建椭圆肤色模型
oval_matrix = np.zeros((256, 256), dtype=np.uint8)   # 建立一个用于存储椭圆模型的矩阵
center = (round(113), round(152))
size = (25, 12)
angle = -20
color = 255
oval_model = cv2.ellipse(oval_matrix, center, size, angle, 0, 360, color, thickness=-1)  # 取得椭圆肤色模型


# 使用椭圆肤色模型检测
def oval_test(img):

    x, y, d = img.shape
    x_matrix = np.zeros((x, y), dtype=np.uint8)          # 建立一个匹配的二值矩阵图像
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)   # 图像色彩空间转换
    Y, Cr, Cb = cv2.split(img_YCrCb)                     # 取得三种矩阵

    # 遍历图片的每一个像素点
    for row in range(x):
        for col in range(y):
            Cr_v = Cr[row, col]
            Cb_v = Cb[row, col]
            x_matrix[row, col] = oval_model[Cb_v, Cr_v]

    # 将图片和三维矩阵进行与运算
    and_matrix = np.zeros((x, y, 3), dtype=np.uint8)
    and_matrix[:, :, 0] = x_matrix
    and_matrix[:, :, 1] = x_matrix
    and_matrix[:, :, 2] = x_matrix
    img_and = cv2.bitwise_and(img, and_matrix)                       # 得到去除干扰的完整肤色图片

    draw = outline_cut(img_and, img)
    '''
    # 图像腐蚀操作
    # 暂时不用
    img_gray = cv2.cvtColor(img_and, cv2.COLOR_BGR2GRAY)                  # 脸部转换成灰度图
    r, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 脸部转换成二值图，准备进行腐蚀操作
    img_read_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             # 完整图片转换
    # 图像腐蚀操作
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(img_binary, kernel)
    # print(Y, Cr, Cb)
    # cv2.imshow('img_and', img_and)
    '''

    return draw


