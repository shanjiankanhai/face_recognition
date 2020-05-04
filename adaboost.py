"""
# adaboost算法识别人脸部区域
# 框出人脸和眼睛区域
"""
import cv2
import numpy as np
from calculate import cal_eyes_pos

face_path = r'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(face_path)
eye_path = r'haarcascade_eye_tree_eyeglasses.xml'
eye_cascade = cv2.CascadeClassifier(eye_path)
smile_path = r'haarcascade_smile.xml'                   # 笑脸检测
smile_cascade = cv2.CascadeClassifier(smile_path)


def adaboost_test(img):
    """
    函数逻辑：先检测人脸位置，接着从检测到的人脸中检测眼睛所在位置，
                         同时在人脸中检测嘴角位置
    :param img:
    :return:
    """
    # 人脸位置检测
    faces = face_cascade.detectMultiScale(img, 1.2, 2)   # 检测人脸所在的位置
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 用颜色为BGR（255,0,0）粗度为2的线条在img画出识别出的矩型
        face_re = img[y:y + h, x:x + w]                             # 抽取出框出的脸部部分，注意顺序y在前
        # cv2.imwrite(file_name, face_re)                           # 把剪出的面部区域存储
        circle = (int(round(x+w/2)), int(round(y+h/2)))                     # 面部中心位置
        cv2.circle(img, circle, 5, (0, 0, 255), -1)                         # 画出中心点所在位置
        circle_string = 'center:{x},{y}'.format(x=int(round(x+w/2)), y=int(round(y+h/2)))
        cv2.putText(img, circle_string, (x+w+5, y - 7), 3, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # 眼睛位置识别
        eyes = eye_cascade.detectMultiScale(face_re)                # 在框出的脸部部分识别眼睛
        lean_angle, revolve_angle = cal_eyes_pos(eyes)              # 返回头部的倾斜和旋转
        lean_string = 'lean:{:.02f}'.format(lean_angle)
        revolve_string = 'revolve:{}'.format(revolve_angle)
        cv2.putText(img, lean_string, (x+w+5, y + 20), 3, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, revolve_string, (x + w + 5, y + 50), 3, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        for (ex, ey, ew, eh) in eyes[:2]:                           # 一张脸中只采集两只眼睛的轮廓
            # ex,ey等实际上是在脸部face_re上的位置，是一个相对数值
            if int(ey+round(eh/2)) < int(round(h/2)):               # 减少鼻孔的误检
                cv2.rectangle(face_re, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # 笑脸检测
        # 识别笑脸并框出
        roi_gray = cv2.cvtColor(face_re, cv2.COLOR_BGR2GRAY)        # 图像转换成灰度图
        smile = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.16,
            minNeighbors=35,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # 框出上扬的嘴角并对笑脸打上Smile标签
        # 使用for循环的时候smile已经检测出来，否则不会进行循环
        for (x2, y2, w2, h2) in smile:
            if int(round(y2+h2/2)) > int(round(2*h/3)):               # 防止误检
                # print('intround', int(round(y+y2+(h+h2)/2)))
                cv2.rectangle(face_re, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
                cv2.putText(img, 'Smile', (x, y - 7), 3, 0.75, (0, 0, 255), 2, cv2.LINE_AA)  # 第5个参数是字体大小

        # 在方框旁边显示数据
    return img


