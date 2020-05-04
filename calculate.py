"""
# RGB图像计算角度参数
# adaboost算法和椭圆肤色模型都可以调用
"""
import math


def cal_angle():
    pass


# adaboost算法调用，计算脸部中心位置
def cal_faces_pos(faces):
    pass


# adaboost算法调用，计算眼睛中心位置
# 椭圆肤色模型调用，在识别出人脸位置后调用
def cal_eyes_pos(eyes):
    """

    :param eyes: 有n个列表，每个列表有4个数字
    :return:  返回脸部的倾斜角度、旋转角度
    """
    # 首先判断捕捉的眼睛个数
    if len(eyes) == 0:
        return False, False
    elif len(eyes) == 1:

        return False, False

    # 正常情况
    else:
        # 左右眼判断
        first_eye_ex = eyes[0][0]    # 第一只眼睛在脸部的x坐标
        second_eye_ex = eyes[1][0]
        left_eye = []                # 定义左眼数据
        right_eye = []               # 定义右眼数据
        if first_eye_ex < second_eye_ex:      # 对左右眼进行判断
            left_eye = eyes[0]
            right_eye = eyes[1]
        else:
            left_eye = eyes[1]
            right_eye = eyes[0]      # 到此确定左右眼
        # 脸部倾斜角度判断
        lean_angle = 0                    # 定义脸部倾斜角度指标
        revolve_angle = 0                 # 定义脸部旋转角度指标
        d_value_y = left_eye[1] - right_eye[1]                    # 定义左眼和右眼在y方向的差值
        d_value_x = left_eye[0] - right_eye[0]                    # 这是个负值
        # print('d_x', d_value_x)
        if d_value_y > 5:                                         # 脸部向一个方向倾斜
            lean_radian = math.atan2(d_value_y, -d_value_x)         # 以弧度表示脸部的正切数据
            lean_angle = math.degrees(lean_radian)                  # 弧度转角度
        elif (d_value_y <= 5) and (d_value_y >= -5):              # 这种情况不计算角度
            lean_angle = 0
        else:                                                     # 脸部向另一个方向倾斜
            lean_radian = math.atan2(d_value_y, -d_value_x)
            lean_angle = math.degrees(lean_radian)
        return lean_angle, revolve_angle    # 返回脸部倾斜角度和旋转角度

