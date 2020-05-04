"""
# 椭圆肤色模型调用的子程序
# 轮廓识别
# 主要目的是框出人脸区域
"""
import cv2
import numpy as np
from calculate import cal_eyes_pos

eye_path = r'haarcascade_eye_tree_eyeglasses.xml'
eye_cascade = cv2.CascadeClassifier(eye_path)


def outline_cut(img, img_face):
    """

    :param img: 椭圆头部
    :param img_face: 原始图片
    :return:
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                     # 椭圆人脸图像转换成灰度图
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)   # 灰度图转换成二值图像
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 画出图像的轮廓

    # 获取轮廓最大值的索引
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)                                          # 获得最大值索引

    # 画出最大轮廓的中心点
    M = cv2.moments(contours[max_idx])
    if M['m00'] !=0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    cv2.circle(img_face, (cX, cY), 5, (0, 0, 255), -1)

    mask = np.zeros(img.shape, np.uint8)                               # 做一个纯黑的画布
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)   # 画出所有轮廓线
    cv2.fillConvexPoly(mask, contours[max_idx], (255, 255, 255))       # 填充最大轮廓线区域
    loc = cv2.bitwise_and(img_face, mask)                                   # 得到取出的脸部位置    #########
    x, y, w, h = cv2.boundingRect(contours[max_idx])                   # 得到最大连通区域的矩形框坐标

    br = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]])
    cv2.drawContours(img_face, [br], -1, (255, 255, 255), 2)                   # 画出矩形框框住人脸

    # 画出眼睛所在的区域
    face_re = img_face[y:y + h, x:x + w]                            # 抽取出框出的脸部部分，注意顺序y在前
    eyes = eye_cascade.detectMultiScale(face_re)                    # 在框出的脸部部分识别眼睛
    lean_angle, revolve_angle = cal_eyes_pos(eyes)                  # 返回头部的倾斜和旋转
    lean_string = 'lean:{:.02f}'.format(lean_angle)
    revolve_string = 'revolve:{}'.format(revolve_angle)
    cv2.putText(img_face, lean_string, (x + w + 5, y + 20), 3, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img_face, revolve_string, (x + w + 5, y + 50), 3, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_re, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    # cv2.circle(loc, (cX, cY), 5, (0, 0, 255), -1)                         # 画出中心点所在位置

    return img_face

