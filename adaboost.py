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
    faces = face_cascade.detectMultiScale(img, 1.2, 2)   # 检测人脸所在的位置
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 用颜色为BGR（255,0,0）粗度为2的线条在img画出识别出的矩型
        face_re = img[y:y + h, x:x + w]                             # 抽取出框出的脸部部分，注意顺序y在前
        # cv2.imwrite(file_name, face_re)                           # 把剪出的面部区域存储
        eyes = eye_cascade.detectMultiScale(face_re)                # 在框出的脸部部分识别眼睛
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_re, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        roi_gray = cv2.cvtColor(face_re, cv2.COLOR_BGR2GRAY)        # 图像转换成灰度图

        # 笑脸检测
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


'''
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return_number, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(frame.shape, np.uint8)  # 做一个纯黑的画布
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    cv2.imshow('mask', mask)
 
    # 图片镜像，
    # 镜像过后会有明显卡顿，不建议镜像，或者采用其他方法进行镜像
    rows, cols = frame.shape[:2]
    mapx = np.zeros(frame.shape[:2], np.float32)
    mapy = np.zeros(frame.shape[:2], np.float32)
    for i in range(rows):
        for j in range(cols):
            mapx.itemset((i, j), cols-1-j)
            mapy.itemset((i, j), i)
    rst_mask = cv2.remap(mask, mapx, mapy, cv2.INTER_LINEAR)
    cv2.imshow('mask', rst_mask)
'''