# -*- coding: utf-8 -*-
"""
Spyder Editor

Modified on 2021-1-14, v1.1.0

Modified by Lixiaohui
"""

import numpy as np
cimport numpy as np
import cython
cimport cython


cdef extern from "math.h":  # 这些函数用C语言里面的
    double cos(double)
    double sin(double)
    double sqrt(double)
    double pow(double x, double y)
    double ceil(double)
    double floor(double)
    double tanh(double)
    double atan(double)
    double M_PI  # 继承 pi


@cython.boundscheck(False)
@cython.wraparound(False)
def heading(np.ndarray[np.float64_t, ndim=1] vxy):
    cdef double vx = vxy[0], vy = vxy[1], heading
    if vy > 0: heading = atan(vx / vy)
    elif vy < 0: heading = atan(vx / vy) + M_PI
    else:
        if vx > 0: heading = M_PI / 2.
        elif vx < 0: heading = M_PI * (3 / 2.)
        else: heading = 2 * M_PI
    heading = np.degrees(heading) + 360 if -M_PI/2 < heading < 0 else np.degrees(heading)
    return np.int32(heading*10) # 返回10倍角度值


@cython.boundscheck(False)
@cython.wraparound(False)
def data_fusion(np.ndarray[np.int32_t, ndim=2] last_target_state, np.ndarray[np.int32_t, ndim=2] camera_output,
                np.ndarray[np.int32_t, ndim=2] radar_output, double dt=0.1):
    # camera_output/radar_output: class, Xw, Yw, Vx, Vy, heading_init
    # last_target_state: ID, class, Xw, Yw, Vx, Vy, heading, camera_identify
    # target_state: ID, class, Xw, Yw, Vx, Vy, heading, camera_identify    
    cdef long i, j, lt = 0, X = 0, Y = 1, Kx, Ky 
    cdef long idex, target_class, camera_identify, target_ID
    cdef double Dr, Vr, Dc, Kd, Kv, X_scale = 300, Y_scale = 300, Prop  # delta_t  X、Y方向误差尺度
    cdef np.ndarray[np.int32_t, ndim=2] target_state, target_predict = np.copy(last_target_state[:,2:6])
    cdef np.ndarray[np.int32_t, ndim=2] RC_bias, trace_bias    
    cdef np.ndarray[np.int32_t, ndim=1] tmp_state, RC_d, trace_d, ID_list
    # 初始化，行数为雷达、摄像头及历史检测目标之和，列数为8...不需要再进行生命跟踪
    target_state = np.zeros((last_target_state.shape[0] + radar_output.shape[0] + camera_output.shape[0], \
                             last_target_state.shape[1]), np.int32)
    tmp_state = np.zeros(target_state.shape[1], np.int32)  # tmp_state: ID(0), class(1), Xw(2), Yw(3), Vx(4), Vy(5), heading(6), camera_identify(7)
    # delta_t = dt*(target_position.shape[2]-np.int32(1)) # 用距离计算速度所需要的时间
    for i in range(target_predict.shape[0]):  # target_predict: Xw, Yw, Vx, Vy
        target_predict[i,0] = long(last_target_state[i,2] + dt*last_target_state[i,4])
        target_predict[i,1] = long(last_target_state[i,3] + dt*last_target_state[i,5])    
    # radar 检测到的目标 i 和 camera 进行融合, 得到该目标的位置和速度, 若被相机检测到还可以得到初步的 target_class
    for i in range(radar_output.shape[0]): # 对radar中的每一个目标: class[0]、Xw[1]、Yw[2]、Vx[3]、Vy[4]、heading[5]
        Dr = sqrt(pow(radar_output[i,1], 2) + pow(radar_output[i,2],2))  # 雷达距离原点的距离
        # Vr = sqrt(pow(radar_output[i,3], 2) + pow(radar_output[i,4],2))  # 雷达测得的速度
        Kd = 1 + tanh(Dr/10000)/4.    # Kv = 1 + tanhf(Vr/1000)/4.
        Kx, Ky = long(X_scale*Kd), long(Y_scale*Kd)
        # print (Kx, Ky)
        RC_bias = np.zeros((camera_output.shape[0],2), np.int32)  # 第i个雷达和剩下所有camera目标的位置差
        RC_d = 1000*np.ones(RC_bias.shape[0], np.int32) # 第i个雷达和剩下所有camera目标的距离
        for j in range(RC_bias.shape[0]): # RC_bias: Xw, Yw
            RC_bias[j,0] = abs(radar_output[i,1] - camera_output[j,1])
            RC_bias[j,1] = abs(radar_output[i,2] - camera_output[j,2])
            if RC_bias[j,0] < Kx and RC_bias[j,1] < Ky: # 若Xw, Yw都小于尺度值
                RC_d[j] = long(sqrt(pow(RC_bias[j,0], 2) + pow(RC_bias[j,1], 2)))
        if RC_d.shape[0] > 0 and np.min(RC_d) < 1000: # 同时被毫米波雷达和摄像头检测出来
            idex = long(np.argmin(RC_d)) # RC_d中最小值的索引, 即第i雷达目标和第idex个摄像头目标的距离最小
            Prop = 0.6*Kd  # radar所占比例值，0.4~1(Kd最大值为1.25, 实际达不到)
            tmp_state[1] = camera_output[idex, 0]  # class:类型都来自camera
            tmp_state[2] = long((1-Prop)*camera_output[idex,1] + Prop*tmp_state[2])  # X进行融合
            tmp_state[3] = long((1-Prop)*camera_output[idex,2] + Prop*tmp_state[3])  # Y进行融合
            tmp_state[4] = long((1-Prop)*camera_output[idex,3] + Prop*tmp_state[4])  # Vx进行融合
            tmp_state[5] = long((1-Prop)*camera_output[idex,4] + Prop*tmp_state[5])  # Vy进行融合
            tmp_state[7] = 1 # camera_identify, 表示目标是由摄像头识别
            camera_output = np.delete(camera_output, idex, axis=0)  # 把找到和radar匹配目标的camera目标删除                    
        else: # 被radar检出而未被camera检出, tmp_state: class, Xw(cm), Yw(cm)
            tmp_state[1:6] = radar_output[i, 0:5]  # 根据雷达状态信息独自进行目标类型判断
            
        # 对于这个目标，以下考虑它是否在上一时刻出现，通过跟踪得到 ID, 以及确定 target_class
        trace_bias = np.zeros((target_predict.shape[0],2), np.int32) # 当前目标和所有预测结果的X,Y偏差
        trace_d = 1000*np.ones(trace_bias.shape[0], np.int32) # 当前目标与所有预测结果的距离
        for j in range(trace_bias.shape[0]): # trace_bias: Xw,Yw
            trace_bias[j,0] = abs(tmp_state[2] - target_predict[j,0])
            trace_bias[j,1] = abs(tmp_state[3] - target_predict[j,1])
            if trace_bias[j,0] < Kx and trace_bias[j,1] < Ky: # 若Xw, Yw都小于尺度值
                trace_d[j] = long(sqrt(pow(trace_bias[j,0], 2) + pow(trace_bias[j,1], 2)))
        if trace_d.shape[0] > 0 and np.min(trace_d) < 1000: # 若上一时刻目标存在
            idex = long(np.argmin(trace_d)) # 即当前状态信息与第idex个预测值相差最小
            tmp_state[0] = last_target_state[idex,0] # 沿用上一时刻的目标ID
            tmp_state[6] = heading(np.float64(tmp_state[4:6])) # 计算当前航向角
            if tmp_state[6] == 3600:  # 若算出角度为无效值（vx=0, vy=0）
                tmp_state[6] = last_target_state[idex,6]  # 沿用上一时刻的角度
            elif abs(last_target_state[idex,6] - tmp_state[6]) >= 100:  # 大于10°... 十倍角
                tmp_state[6] = long(0.2*tmp_state[6] + 0.8*last_target_state[idex,6])  # 滤波处理
            if sqrt(pow(tmp_state[4], 2.) + pow(tmp_state[5], 2.)) < 250:
                tmp_state[6] = radar_output[i, 5]
            if (not tmp_state[7]) and (last_target_state[idex,7]):  # 若目标本次不是由图像识别，但上一帧是被图像识别
                tmp_state[1] = last_target_state[idex,1]  # 完全沿用上一帧得到的目标类型
            tmp_state[7] = max(tmp_state[7], last_target_state[idex,7]) # camera_identify 更新
            last_target_state = np.delete(last_target_state, idex, axis=0)  # 消除已经被继承的项
            target_predict = np.delete(target_predict, idex, axis=0)  # 消除已经被继承的项
        else: # 如果当前目标在上一帧不存在，即为新目标
            ID_list, target_ID = np.hstack((target_state[:lt,0], last_target_state[:,0])), 1  # 已经存在的ID, 初始赋值
            while target_ID in ID_list: 
                target_ID += 1
            tmp_state[0] = target_ID
            tmp_state[6] = radar_output[i, 5] # 初始化航向角
        target_state[lt], lt = tmp_state, lt+1
    for i in range(camera_output.shape[0]): # 被camera检出而未被radar检出（被两者均检出的项已在和雷达融合后剔除）
        Dc = sqrt(pow(camera_output[i,1], 2) + pow(camera_output[i,2],2))  # 视频距离原点的距离
        Kd = 1 + tanh(Dc/6000)/4.  # 随着距离增加，比例大小用tanh函数来约束
        Kx, Ky = long(X_scale*Kd), long(Y_scale*Kd)
        tmp_state = np.zeros(target_state.shape[1], np.int32)  # tmp_state: ID(0), class(1), Xw(2), Yw(3), Vx(4), Vy(5), camera_identify(6)
        tmp_state[1:6] = camera_output[i, 0:5]  # 类型都来自camera
        tmp_state[7] = 1  # camera_identify 置1
        trace_bias = np.zeros((target_predict.shape[0],2), np.int32) # 当前目标和所有预测结果的X,Y偏差
        trace_d = 1000*np.ones(trace_bias.shape[0], np.int32) # 当前目标与所有预测结果的距离
        for j in range(trace_bias.shape[0]): # trace_bias: Xw,Yw
            trace_bias[j,0] = abs(tmp_state[2] - target_predict[j,0])
            trace_bias[j,1] = abs(tmp_state[3] - target_predict[j,1])
            if trace_bias[j,0] < Kx and trace_bias[j,1] < Ky: # 若Xw, Yw都小于尺度值
                trace_d[j] = long(sqrt(pow(trace_bias[j,0], 2) + pow(trace_bias[j,1], 2)))
        if trace_d.shape[0] > 0 and np.min(trace_d) < 1000: # 若上一目标存在
            idex = long(np.argmin(trace_d)) # 即当前状态信息与第idex个预测值相差最小
            tmp_state[0] = last_target_state[idex,0] # 沿用上一时刻的目标ID
            tmp_state[6] = heading(np.float64(tmp_state[4:6])) # 计算当前航向角
            if tmp_state[6] == 3600:  # 若算出角度为无效值（vx=0, vy=0）
                tmp_state[6] = last_target_state[idex,6]  # 沿用上一时刻的角度
            elif abs(last_target_state[idex,6] - tmp_state[6]) >= 100:  # 大于10°... 十倍角
                tmp_state[6] = long(0.2*tmp_state[6] + 0.8*last_target_state[idex,6])  # 滤波处理
            if sqrt(pow(tmp_state[4], 2.) + pow(tmp_state[5], 2.)) < 250:
                tmp_state[6] = camera_output[i, 5]
            last_target_state = np.delete(last_target_state, idex, axis=0)  # 消除已经被继承的项
            target_predict = np.delete(target_predict, idex, axis=0)  # 消除已经被继承的项
        else: # 如果当前目标在上一帧不存在，即为新目标
            ID_list, target_ID = np.hstack((target_state[:lt,0], last_target_state[:,0])), 1  # 已经存在的ID, 初始赋值
            while target_ID in ID_list: 
                target_ID += 1
            tmp_state[0] = target_ID
            tmp_state[6] = camera_output[i, 5] # 计算当前航向角
        target_state[lt], lt = tmp_state, lt+1
    # for i in range(last_target_state.shape[0]): # 对于 last_target_state 中剩下的任一目标
    #     target_position[:, last_target_state[i,0], :] = np.nan # 不再进行跟踪
    # for i in range(lt):
    #     target_ID = target_state[i,0]
    #     for j in range(target_position.shape[2] + np.int32(1)): 
    #         if j == target_position.shape[2] or np.isnan(target_position[X, target_ID, j]):
    #             break  # j表示还未被赋值的列数
    #     if j < target_position.shape[2]: # position 数量不足，直接在后面赋值
    #         target_position[X, target_ID, j] = np.copy(target_state[i,2])
    #         target_position[Y, target_ID, j] = np.copy(target_state[i,3])
    #     else: # j == target_position.shape[2]
    #         target_position[:, target_ID, 0:j-1] = target_position[:, target_ID, 1:j]
    #         target_position[X, target_ID, j-1] = np.copy(target_state[i,2]) # X
    #         target_position[Y, target_ID, j-1] = np.copy(target_state[i,3]) # Y   
    #         target_state[i,4] = long(0.8*(target_position[X, target_ID, j-1]-target_position[X, target_ID, 0])/delta_t +  \
    #                            0.2*target_state[i,4]) # Vx
    #         target_state[i,5] = long(0.2*(target_position[Y, target_ID, j-1]-target_position[Y, target_ID, 0])/delta_t +  \
    #                            0.8*target_state[i,5]) # Vy修正受到限制    
    return target_state[:lt]


@cython.boundscheck(False)
@cython.wraparound(False)
def radar_movement(np.ndarray[np.int32_t, ndim=2] radar_data, np.ndarray[np.int32_t, ndim=2] last_radar_state, 
                   np.ndarray[np.float64_t, ndim=3] radar_position, str usage, long ini_life=-2, long full_life=2, double dt=0.05):
    # radar_data: X, Y, Vy, Pv
    # last_radar_state: ID(0), class(1), Xr(2), Yr(3), Vx(4), Vy(5), Pv(6), life(7)
    def radar_class(np.ndarray[np.int32_t, ndim=1] state, str usage): # class, Xw, Yw, Vx, Vy, Pv, life
        if state[6] < 100 and 'ped' in usage: return 1
        else: return 3
    cdef long i, j, idex, lr = 0, Kx, Ky, Kv, X=0, Y=1, radar_ID
    cdef double Dr, Kd, X_scale = 300, Y_scale = 350, Vy_scale = 200, delta_t
    cdef np.ndarray[np.int32_t, ndim=2] radar_state, trace_bias, radar_predict = np.copy(last_radar_state[:,2:7])
    cdef np.ndarray[np.int32_t, ndim=1] trace_d, tmp_state, ID_list
    # 初始化雷达状态，行：当前检测目标数+历史检测目标数, 列数=8
    radar_state = np.zeros((last_radar_state.shape[0] + radar_data.shape[0], last_radar_state.shape[1]), np.int32)
    delta_t = dt*(radar_position.shape[2]-np.int32(1)) # 用距离计算速度所需要的时间
    tmp_state = np.zeros(radar_state.shape[1], np.int32) # ID, class, X, Y, Vx, Vy, Pv, life
    for i in range(radar_predict.shape[0]): # target_predict: Xr, Yr, Vx, Vy, Pv
        radar_predict[i,0] = long(last_radar_state[i,2] + dt*last_radar_state[i,4])
        radar_predict[i,1] = long(last_radar_state[i,3] + dt*last_radar_state[i,5])
    for i in range(radar_data.shape[0]): # 对radar中的每一个目标: X[0]、Y[1]、Vy[2]、Pv[3]
        Dr = sqrt(pow(radar_data[i,0],2.) + pow(radar_data[i,1],2.))  # 雷达目标距离原点的距离
        Kd = 1 + tanh(Dr/10000)/4. 
        Kx, Ky, Kv = long(X_scale*Kd), long(Y_scale*Kd), long(Vy_scale*Kd)
        trace_bias = np.zeros((radar_predict.shape[0],3), np.int32)  # X, Y, Vy
        trace_d = 1000*np.ones(trace_bias.shape[0], np.int32) # 当前目标与所有预测结果的距离
        for j in range(trace_bias.shape[0]): # X, Y, Vy
            trace_bias[j,0] = abs(radar_data[i,0]-radar_predict[j,0])  # X
            trace_bias[j,1] = abs(radar_data[i,1]-radar_predict[j,1])  # Y
            trace_bias[j,2] = abs(radar_data[i,2]-radar_predict[j,3])  # Vy...radar_data没有Vx
            if trace_bias[j,0] < Kx and trace_bias[j,1] < Ky and trace_bias[j,2] < Kv: # 若X, T, V都小于尺度值
                trace_d[j] = long(sqrt(pow(trace_bias[j,0], 2) + pow(trace_bias[j,1], 2)))
        if trace_d.shape[0] > 0 and np.min(trace_d) < 1000: # tmp_state: ID(0), class(1), X(2), Y(3), Vx(4), Vy(5), Pv(6), life(7)
            idex = long(np.argmin(trace_d)) # 即当前状态信息与第idex个预测值相差最小
            tmp_state[0] = last_radar_state[idex,0]  # ID
            tmp_state[2:4] = radar_data[i,0:2] # X, Y
            tmp_state[4] = long(0.2*(radar_data[i,0]-last_radar_state[idex,2])/dt + 0.8*last_radar_state[idex,4]) # Vx
            tmp_state[5] = radar_data[i,2] # Vy
            tmp_state[6] = long(0.2*radar_data[i,3] + 0.8*last_radar_state[idex,6])  # Pv
            tmp_state[7] = min(full_life, last_radar_state[idex,7]+1) # life update
            last_radar_state = np.delete(last_radar_state, idex, axis=0)
            radar_predict = np.delete(radar_predict, idex, axis=0)
        else: # 如果当前目标在上一帧不存在，即为新目标
            ID_list, radar_ID = np.hstack((radar_state[:lr,0], last_radar_state[:,0])), 1  # 已经存在的ID, 初始赋值
            while radar_ID in ID_list: 
                radar_ID += 1
            tmp_state[0] = radar_ID
            tmp_state[2:4] = radar_data[i,0:2] # Xw, Yw, 
            tmp_state[4], tmp_state[5] = np.int32(0), radar_data[i,2] # Vx, Vy
            tmp_state[6], tmp_state[7] = radar_data[i,3], ini_life # Pv, life              
        tmp_state[1] = radar_class(tmp_state, usage) # class
        if 'ped' in usage and tmp_state[1] > 1: continue
        radar_state[lr,:], lr = tmp_state, lr+1
    for i in range(last_radar_state.shape[0]): # 对于 last_radar_state 中剩下的任一目标
        tmp_state[7] = last_radar_state[i,7] - 1  # 生命值 -1
        if tmp_state[7] <= -1: 
            radar_position[:, last_radar_state[i,0], :] = np.nan
            continue  # 不再进行跟踪更新
        tmp_state[0:2] = last_radar_state[i,0:2]  # ID, class
        tmp_state[2:7] = radar_predict[i] # Xw, Yw, Vx, Vy, Pv        
        radar_state[lr,:], lr = tmp_state, lr+1
    for i in range(lr): # 对于检测出的任一目标
        radar_ID = radar_state[i,0]
        for j in range(radar_position.shape[2] + np.int32(1)): 
            if j == radar_position.shape[2] or np.isnan(radar_position[X, radar_ID, j]):
                break  # j表示还未被赋值的列数
        if j < radar_position.shape[2]: # 此时radar_position未集齐足够position, 直接跟在后面赋值
            radar_position[X, radar_ID, j] = radar_state[i,2]
            radar_position[Y, radar_ID, j] = radar_state[i,3]
        else: # j = radar_position.shape[2]
            radar_position[:, radar_ID, 0:j-1] = radar_position[:, radar_ID, 1:j]
            radar_position[X, radar_ID, j-1] = radar_state[i,2] # X
            radar_position[Y, radar_ID, j-1] = radar_state[i,3] # Y
            movement_obj = sqrt(pow(radar_position[X, radar_ID, j-1] - radar_position[X, radar_ID, 0], 2.) +  \
                           pow(radar_position[Y, radar_ID, j-1] - radar_position[Y, radar_ID, 0], 2.))
            if movement_obj < 200:
                radar_state[i, 4] = radar_state[i, 5] = 0
            else:
                radar_state[i,4] = long(0.8*(radar_position[X, radar_ID, j-1]-radar_position[X, radar_ID, 0])/delta_t +  \
                                   0.2*radar_state[i,4]) # Vx
                radar_state[i,5] = long(0.2*(radar_position[Y, radar_ID, j-1]-radar_position[Y, radar_ID, 0])/delta_t +  \
                                   0.8*radar_state[i,5]) # Vy修正受到限制

    return radar_state[:lr]


@cython.boundscheck(False)
@cython.wraparound(False)
def camera_movement(np.ndarray[np.int32_t, ndim=2] camera_data, np.ndarray[np.int32_t, ndim=2] last_camera_state, 
                    np.ndarray[np.float64_t, ndim=3] camera_position, long ini_life=0, long full_life=2, double dt=0.04):
    # camera_data: class Xw Yw m_B m_G m_R
    # last_camera_state: ID[0], class[1], Xw[2], Yw[3], Vx[4], Vy[5], m_B[6], m_G[7], m_R[8], life[9]
    cdef long i, j, idex, lc = 0, Kx, Ky, K_color = 25, X=0, Y=1, camera_ID
    cdef double Dr, Kd, X_scale = 300, Y_scale = 300, delta_t
    cdef np.ndarray[np.int32_t, ndim=2] camera_state, trace_bias, camera_predict=np.copy(last_camera_state[:,2:9])
    cdef np.ndarray[np.int32_t, ndim=1] tmp_state, trace_d
    # 初始化摄像头状态，行：当前检测目标数+历史检测目标数, 列数=10                         
    camera_state = np.zeros((camera_data.shape[0] + last_camera_state.shape[0], last_camera_state.shape[1]), np.int32)
    delta_t = dt*(camera_position.shape[2]-np.int32(1)) # 用距离计算速度所需要的时间
    tmp_state = np.zeros(camera_state.shape[1], np.int32) # ID(0), class(1), Xw(2), Yw(3), Vx(4), Vy(5), m_B(6), m_G(7), m_R(8), life(9)
    for i in range(camera_predict.shape[0]): # camera_predict: Xw[0], Yw[1], Vx[2], Vy[3], m_B[4], m_G[5], m_R[6]
        camera_predict[i,0] = long(last_camera_state[i,2] + dt*last_camera_state[i,4])
        camera_predict[i,1] = long(last_camera_state[i,3] + dt*last_camera_state[i,5])
    for i in range(camera_data.shape[0]): # 对摄像头中的每一个目标：class Xw Yw m_B m_G m_R
        Dr = sqrt(pow(camera_data[i,1], 2.) + pow(camera_data[i,2],2.))  # 视频目标距离原点的距离
        Kd = 1 + tanh(Dr/6000)/4. 
        Kx, Ky = long(X_scale*Kd), long(Y_scale*Kd)
        trace_bias = np.zeros((camera_predict.shape[0],5), np.int32)  # X, Y, B, G, R
        trace_d = 1000*np.ones(trace_bias.shape[0], np.int32) # 当前目标与所有预测结果的距离
        for j in range(trace_bias.shape[0]):  # X, Y, B, G, R
            trace_bias[j,0] = abs(camera_data[i,1] - camera_predict[j,0]) # X
            trace_bias[j,1] = abs(camera_data[i,2] - camera_predict[j,1]) # Y
            trace_bias[j,2] = abs(camera_data[i,3] - camera_predict[j,4]) # R
            trace_bias[j,3] = abs(camera_data[i,4] - camera_predict[j,5]) # G
            trace_bias[j,4] = abs(camera_data[i,5] - camera_predict[j,6]) # B
            if (trace_bias[j,0] < Kx and trace_bias[j,1] < Ky and trace_bias[j,2] < K_color and
                trace_bias[j,3] < K_color and trace_bias[j,4] < K_color): # 所有的均小于尺度值
                trace_d[j] = long(sqrt(pow(trace_bias[j,0], 2) + pow(trace_bias[j,1], 2)))
        if trace_d.shape[0] > 0 and np.min(trace_d) < 1000: # tmp_state: ID(0), class(1), Xw(2), Yw(3), Vx(4), Vy(5), m_B(6), m_G(7), m_R(8), life(9)
            idex = long(np.argmin(trace_d)) # 即当前状态信息与第idx个预测值相差最小
            tmp_state[0:2] = last_camera_state[idex,0:2] # 继承目标ID, class
            if last_camera_state[idex,1] == 2: # 若上一帧识别目标为bike
                tmp_state[1] = max(last_camera_state[idex,1], tmp_state[1]) # 被识别为bike的目标不会再被识别为行人
            tmp_state[2:4] = camera_data[i,1:3] # Xw, Yw
            tmp_state[4] = long(0.2*(camera_data[i,1]-last_camera_state[idex,2])/dt + 0.8*last_camera_state[idex,4]) # Vx
            tmp_state[5] = long(0.2*(camera_data[i,2]-last_camera_state[idex,3])/dt + 0.8*last_camera_state[idex,5]) # Vy
            tmp_state[6:9] = camera_data[i,3:6] # B, G, R
            tmp_state[9] = min(full_life, last_camera_state[idex,9]+1) if tmp_state[1] != (1 and 2) else min(10, last_camera_state[idex,9]+1)# life
            last_camera_state = np.delete(last_camera_state, idex, axis=0)
            camera_predict = np.delete(camera_predict, idex, axis=0)
        else: # 如果当前目标在上一帧不存在，即为新目标
            ID_list, camera_ID = np.hstack((camera_state[:lc,0], last_camera_state[:,0])), 1  # 已经存在的ID, 初始赋值
            while camera_ID in ID_list: 
                camera_ID += 1
            tmp_state[0] = camera_ID
            tmp_state[1:4] = camera_data[i,0:3] # class, Xw, Yw
            tmp_state[4:6] = np.int32([0,0]) # Vx, Vy
            tmp_state[6:9] = camera_data[i,3:6] # B, G, R
            tmp_state[9] = ini_life      
        camera_state[lc,:], lc = tmp_state, lc+1
    for i in range(last_camera_state.shape[0]): # 对于 last_camera_state 中剩下的任一目标
        tmp_state[9] = last_camera_state[i,9] - 1  # 生命值 -1
        if tmp_state[9] <= -1: 
            camera_position[:, last_camera_state[i,0], :] = np.nan
            continue
        tmp_state[0:2] = last_camera_state[i,0:2]
        tmp_state[2:9] = camera_predict[i] # Xw[0], Yw[1], Vx[2], Vy[3], m_B[4], m_G[5], m_R[6]
        camera_state[lc,:], lc = tmp_state, lc+1
    for i in range(lc): # 对于检测出的任一目标
        camera_ID = camera_state[i,0]
        for j in range(camera_position.shape[2] + np.int32(1)): 
            if j == camera_position.shape[2] or np.isnan(camera_position[X, camera_ID, j]):
                break  # j表示还未被赋值的列数
        if j < camera_position.shape[2]: # 此时target_XY未集齐足够position, 直接跟在后面赋值
            camera_position[X, camera_ID, j] = camera_state[i,2]
            camera_position[Y, camera_ID, j] = camera_state[i,3]
        else: # j = camera_position.shape[2]
            camera_position[:, camera_ID, 0:j-1] = camera_position[:, camera_ID, 1:j]
            camera_position[X, camera_ID, j-1] = camera_state[i,2] # X
            camera_position[Y, camera_ID, j-1] = camera_state[i,3] # Y
            movement_obj = sqrt(pow(camera_position[X, camera_ID, j-1] - camera_position[X, camera_ID, 0], 2.) +  \
                                pow(camera_position[Y, camera_ID, j-1] - camera_position[Y, camera_ID, 0], 2.))
            if movement_obj < 200:
                camera_state[i, 4] = camera_state[i, 5] = 0
            else:
                camera_state[i,4] = long(0.8*(camera_position[X, camera_ID, j-1]-camera_position[X, camera_ID, 0])/delta_t +  \
                                    0.2*camera_state[i,4]) # Vx
                camera_state[i,5] = long(0.8*(camera_position[Y, camera_ID, j-1]-camera_position[Y, camera_ID, 0])/delta_t +  \
                                    0.2*camera_state[i,5]) # Vy                
    return camera_state[:lc], camera_position
