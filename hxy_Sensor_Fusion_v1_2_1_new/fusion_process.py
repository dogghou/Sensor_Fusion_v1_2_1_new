# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:53:50 2019

@author: Administrator
"""

import time
import numpy as np
from funs import UTM2W84, sigmoid, tanh


def combine_camera_output(camera_output):
    #    print('combine_camera_output:\n' + str(camera_output))
    # 认为速度大于3m/s的行人是bike, 把就近的bike和行人合并
    camera_output = camera_output[np.where(np.linalg.norm(camera_output[:, 1:3], axis=1) < 8000)[0]].copy()
    vehicle = camera_output[np.where(camera_output[:, 0] >= 3)]
    bike = camera_output[np.where(camera_output[:, 0] == 2)]
    bike = bike[np.where(np.linalg.norm(bike[:, [1, 2]], axis=1) < 6000)[0]]
    person = camera_output[np.where(camera_output[:, 0] == 1)]
    person = person[np.where(np.linalg.norm(person[:, [1, 2]], axis=1) < 3000)[0]]
    if bike.shape[0] > 0 and person.shape[0] > 0:  # 当目标中同时存在行人和自行车时
        select = [i for i in range(len(person)) if np.max(abs(bike - person[i])[:, [1, 2]]) > 150]
        person = person[select]  # 和bike距离小于1.5m的person认为是骑在车上，
    person[([np.where(np.linalg.norm(person[:, [3, 4]], axis=1) > 350)[0]]), 0] = 2  # 速度大于3.5m/s的行人认为是骑手
    return np.vstack((vehicle, bike, person))


def combine_radar_output(radar_output):
    # radar_targets_state: [[class, Xw, Yw, Vx, Vy]...]
    # 将 radar_targets_state 里位置和速度接近的目标合成为一个目标
    dr = np.linalg.norm(radar_output[:, 1:3], axis=1)
    radar_output = radar_output[np.argsort(-dr)]  # 距离从远到近进行排序
    X_scale, Y_scale, V_scale = 250, 250, 200  # 车辆合成 X, Y, V 的差的阈值，X, Y, V 的差小于阈值的两个车辆目标合成为一个
    # 每次循环将列表第一个目标及其他与该目标接近的目标全部取出，合成为一个新目标，放入结果数组 combine_result 中
    combine_result = np.empty((0, 6), np.int32)
    while len(radar_output) > 0:
        ori_target = radar_output[0]
        tmp = np.sum(np.abs(radar_output[:, 1:5] - ori_target[1:5]) // [X_scale, Y_scale, V_scale, V_scale], axis=1)
        same_targets_index = np.where(tmp == 0)[0]
        tmp = radar_output[same_targets_index]
        radar_output = np.delete(radar_output, same_targets_index, axis=0)
        combine_result = np.vstack((combine_result, tmp[0]))
    return combine_result

