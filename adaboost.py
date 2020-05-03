"""
# adaboost算法识别人脸部区域
# 框出人脸和眼睛区域
"""
import cv2
import numpy as np

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

        # 眼睛位置识别
        eyes = eye_cascade.detectMultiScale(face_re)                # 在框出的脸部部分识别眼睛
        for (ex, ey, ew, eh) in eyes:
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
        for (x2, y2, w2, h2) in smile:
            cv2.rectangle(face_re, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
            cv2.putText(img, 'Smile', (x, y - 7), 3, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    return img


