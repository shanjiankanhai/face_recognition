"""
# 程序起点
# 调用摄像头读取数据
# 调用关系为：
"""
import cv2
import numpy as np
from adaboost import adaboost_test
from oval_model import oval_test

# 调用摄像头，读取视频数据
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while capture.isOpened():        # 判断摄像头是否正常开启
    ret, frame = capture.read()  # 抽取视频帧为图片
    # frame_oval = frame
    # adaboost算法识别
    # adaboost_output = adaboost_test(frame)         # 调用自定义模块，使用adaboost算法检测
    # cv2.imshow('adaboost_output', adaboost_output)
    oval_output = oval_test(frame)                 # 调用自定义模块，使用肤色模型去除肤色区域
    cv2.imshow('oval_output', oval_output)

    c = cv2.waitKey(25)  # 按ESC结束
    if c == 27:
        break
capture.release()   # 释放摄像头
cv2.destroyAllWindows()



