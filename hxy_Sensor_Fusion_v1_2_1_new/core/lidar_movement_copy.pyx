# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Spyder Editor

Modified on 2021-8-5, v1.1.0

Modified by Houxueyuan
"""

import numpy as np
cimport numpy as np
import cython
cimport cython
import time, sys

from cython.view cimport array as cvarray
from cpython.array cimport array as pvarray
from array import array as pvarray 

i32 = "l" if 'win' in sys.platform else "i"

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

cdef extern from "float.h":  # float.h
    double LDBL_MAX # 继承最大值

@cython.boundscheck(False)
@cython.wraparound(False)
def is_overlap(np.ndarray[np.int32_t, ndim=1] box, np.ndarray[np.int32_t, ndim=2] bbox, double threshold=0.75): # xmin, xmax, ymin, ymax
    # 求bbox中每行相对于b的覆盖比例... xmin, xmax, ymin, ymax
    cdef long i, j = 0, max_idex, xmin, xmax, ymin, ymax
    cdef double overlap_area, area, overlap_ratio, max_ratio = 0
    cdef np.ndarray[np.int32_t, ndim=2] overlap
    cdef np.ndarray[np.int32_t, ndim=1] overlap_index
    
    overlap = np.zeros((bbox.shape[0],2), np.int32)
    overlap_index = np.zeros(bbox.shape[0], np.int32)
    area = (box[1]-box[0])*(box[3]-box[2]) # 目标target的面积
    for i in range(bbox.shape[0]):
        xmin, ymin = max(box[0], bbox[i,0]), max(box[2], bbox[i,2])
        xmax, ymax = min(box[1], bbox[i,1]), min(box[3], bbox[i,3])
        overlap[i,:] = np.int32([xmax-xmin, ymax-ymin]) # 有交叠的x，y必然是>0q
    for i in range(overlap.shape[0]):
        if overlap[i,0] > 0 and overlap[i,1] > 0:
            overlap_index[j], j = i, j+1
    overlap_index = overlap_index[:j] # overlap_index包含了bbox中所有与box有相交的索引
    overlap = overlap[overlap_index]
    for i in range(overlap.shape[0]):
        overlap_area = overlap[i,0]*overlap[i,1]
        overlap_ratio = overlap_area/area
        if overlap_ratio > max_ratio: 
            max_idex, max_area,  = i, overlap_area 
            max_ratio = overlap_ratio # box与bbox中的第overlap_index[max_idex]个交比最大
    if (max_ratio > threshold) and (max_area > 20000): # 满足一定条件
        return overlap_index[max_idex] # 返回索引

@cython.boundscheck(False)
@cython.wraparound(False)
def cal_overlap_ratio(np.ndarray[np.int32_t, ndim=1] big_box, np.ndarray[np.int32_t, ndim=1] small_box):
    # 求big_box相对于small_box的覆盖比例... xmin, xmax, ymin, ymax
    cdef long xmin, xmax, ymin, ymax, overlap_x, overlap_y
    cdef double overlap_area, area, overlap_ratio
    
    area = (small_box[1]-small_box[0])*(small_box[3]-small_box[2]) # samll_box的面积
    xmin, ymin = max(big_box[0], small_box[0]), max(big_box[2], small_box[2])
    xmax, ymax = min(big_box[1], small_box[1]), min(big_box[3], small_box[3])
    overlap_x = xmax-xmin
    overlap_y = ymax-ymin
    if overlap_x > 0 and overlap_y > 0:
        overlap_area = overlap_x * overlap_y
    else:
        overlap_area = 0
    overlap_ratio = overlap_area/area
    return overlap_ratio

@cython.boundscheck(False)
@cython.wraparound(False)
def target_fusion(np.ndarray[np.int32_t, ndim=2] target): # 把target中索引为idex的合为一体
    # target：xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
    cdef long target_width, i, target_num
    cdef np.ndarray[np.int32_t, ndim=1] target_new = np.copy(target[0]) 
    target_num = 1
    for i in range(1, target.shape[0]):
        target_new[2] = min(target_new[2], target[i,2])
        target_new[3] = max(target_new[3], target[i,3])
        target_new[4] = min(target_new[4], target[i,4])
        target_new[5] = max(target_new[5], target[i,5])
        target_new[7] = max(target_new[7], target[i,7])
        target_new[8] = target_new[8] + target[i,8]
        target_num = target_num + 1
    target_new[8] = target_new[8]//target_num
    target_new[0] = (target_new[2] + target_new[3])//2
    target_new[1] = (target_new[4] + target_new[5])//2    
    target_width = max(target_new[3]-target_new[2], target_new[5]-target_new[4])
    if target_new[7]/target_width >= 4.5:
        target_new = None # 若目标及其细长（误检可能性大）
    return target_new

@cython.boundscheck(False)
@cython.wraparound(False)
#Edited by HOUXUEYUAN at 20210630
def lidar_class_new(np.ndarray[np.int32_t, ndim=1] target):
        # target: xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
        cdef long area = 0
        cdef double aspect_ratio = 0
        cdef double delta_x, delta_y = 0
        delta_x = (target[3]-target[2]) * 1.0
        delta_y = (target[5]-target[4]) * 1.0
        area = (target[3]-target[2])*(target[5]-target[4])
        aspect_ratio = delta_x/delta_y
        #纵横比小于2且xy平面投影面积小于0.5平米，认为是行人
        if (1800  <= target[0] <=2600) or (1300 <= target[1] <= 2050):
            if (area < 4000) and (aspect_ratio >= 0.6) and (aspect_ratio <= 1.7):
                return 1 #行人
            elif ( 4000 <= area <= 10000):
                return 2 #非机动车
            else:
                return 3 #车辆
        else:
            return 3

@cython.boundscheck(False)
@cython.wraparound(False)
#功能：找到两个box，五点（中心点和四个交点）中距离最近的点
#五个点的顺序为：中心点(xc,yc)(0),(xmax,ymax)左上(1),(xmax,ymin)右上(2),(xmin,ymin)右下(3),(xmin,ymax)左下(4)
#输入：
#   box1: xmin(0), xmax(1), ymin(2), ymax(3)
#   box2: xmin(0), xmax(1), ymin(2), ymax(3)
#输出：
#   point_idex:最近点的索引
#   dx_nearest:最近点的x坐标差值
#   dy_nearest:最近点的y坐标差值
#   distance_nearest:最近点之间的2范数距离
#Edited by HOUXUEYUAN at 20210530
def get_nearest_point(np.ndarray[np.int32_t, ndim=1] box1, np.ndarray[np.int32_t, ndim=1] box2):
    cdef long i, point_idex = 0
    cdef long xc1, yc1, xc2, yc2 = 0
    cdef long dx_nearest, dy_nearest, distance_nearest = 0
    cdef np.ndarray[np.int32_t, ndim=1] dx, dy, distance

    dx = np.zeros(5, np.int32)
    dy = np.zeros(5, np.int32)
    distance = np.zeros(5,np.int32)
    xc1, yc1 = (box1[1] + box1[0])//2, (box1[3] + box1[2])//2
    xc2, yc2 = (box2[1] + box2[0])//2, (box2[3] + box2[2])//2

    dx[0], dy[0] = xc1 - xc2, yc1 - yc2   #xc,yc
    distance[0] = long(sqrt(pow(dx[0], 2.) + pow(dy[0], 2.)))   #xc,yc
    dx[1], dy[1] = box1[1] - box2[1], box1[3] - box2[3]   #xmax,ymax
    distance[1] = long(sqrt(pow(dx[1], 2.) + pow(dy[1], 2.)))   #xmax,ymax
    dx[2], dy[2] = box1[1] - box2[1], box1[2] - box2[2]   #xmax,ymin     
    distance[2] = long(sqrt(pow(dx[2], 2.) + pow(dy[2], 2.)))   #xmax,ymin
    dx[3], dy[3] = box1[0] - box2[0], box1[2] - box2[2]   #xmin,ymin 
    distance[3] = long(sqrt(pow(dx[3], 2.) + pow(dy[3], 2.)))   #xmin,ymin
    dx[4], dy[4] = box1[0] - box2[0], box1[3] - box2[3]   #xmin,ymax
    distance[4] = long(sqrt(pow(dx[4], 2.) + pow(dy[4], 2.)))   #xmin,ymax
    
    point_idex = np.argmin(distance)
    dx_nearest = abs(dx[point_idex])
    dy_nearest = abs(dy[point_idex])
    distance_nearest = distance[point_idex]

    return point_idex, dx_nearest, dy_nearest, distance_nearest

@cython.boundscheck(False)
@cython.wraparound(False)
#Edited by HOUXUEYUAN at 20210630
def get_fusioned_target(np.ndarray[np.int32_t, ndim=2] lidar_data):
    cdef long Kd_fusion = 80
    cdef np.ndarray[np.int32_t, ndim=1] selected, dr, target
    
    dr = np.zeros(lidar_data.shape[0], np.int32)
    target = np.zeros(lidar_data.shape[1], np.int32) # lidar_data.shape[1] = 9

    for i in range(lidar_data.shape[0]):
        #每一行与第一行的2范数距离
        l2_distance = long(sqrt(pow(lidar_data[0,0]-lidar_data[i,0], 2.) + pow(lidar_data[0,1]-lidar_data[i,1], 2.)))
        #第一行相对于每一行的覆盖比例
        overlap_ratio = cal_overlap_ratio(lidar_data[0,2:6], lidar_data[i, 2:6])
        if (overlap_ratio > 0.5) or (l2_distance < Kd_fusion):
            dr[i] = 100
        selected = np.int32(np.where(dr > 0)[0]) #距离值小于Kd_fusion或者overlap大于0.5的索引
        if selected.shape[0] == 1:
            target[:] = lidar_data[0, :]
        elif  selected.shape[0] > 1:
            target = target_fusion(lidar_data[selected])    #把符合条件的目标合为一体
        else:
            print("selected is wrong")
            continue
    return target, selected

@cython.boundscheck(False)
@cython.wraparound(False)
#功能：设定跟踪门，匹配当前帧目标和历史目标
#输入：
#    lidar_data:排序后的lidar_data,即当前帧grid_cluster后的所有target
#    lidar_predict:历史信息（last_lidar_state）中，所有目标的预测位置
#输出：
#    target_3Dlist：历史目标和当前目标的对应关系（目标对应关系）
#    target_3Dlist_num:历史目标和当前目标的对应关系（目标的对应数量关系）
#    new_targets:新目标
#Edited by HOUXUEYUAN at 20210630
def track_process(np.ndarray[np.int32_t, ndim=2] lidar_data, np.ndarray[np.int32_t, ndim=2] lidar_predict):
    cdef double aspect_ratio = 0
    cdef long i, j, dx_nearest, dy_nearest, dx_thresh, dy_thresh, distance_thresh, distance_nearest, lidar_predict_area = 0
    cdef np.ndarray[np.int32_t, ndim=1] selected, target, trace_d, target_3Dlist_num
    cdef np.ndarray[np.int32_t, ndim=2] new_targets
    cdef np.ndarray[np.int32_t, ndim=3] target_3Dlist

    target = np.zeros(lidar_data.shape[1], np.int32) # lidar_data.shape[1] = 9
    new_targets = np.zeros((lidar_data.shape[0], lidar_data.shape[1]), np.int32)
    target_3Dlist = np.zeros((lidar_predict.shape[0], lidar_data.shape[0], lidar_data.shape[1]), np.int32)
    target_3Dlist_num = np.zeros(lidar_predict.shape[0], np.int32) 
    
    while (lidar_data.shape[0]) > 0:
        target, selected = get_fusioned_target(lidar_data)
        lidar_data = np.delete(lidar_data, selected, axis=0)  # 从lidar_data中移除已经选中的
        if target is None: 
            continue 
        # target: xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
        #lidar_predict: class(0), xc(1), yc(2), vx(3), vy(4), xmin(5), xmax(6), ymin(7), ymax(8)
        overlap_index = is_overlap(target[2:6], lidar_predict[:,5:]) # 查看该目标是否与上一帧预测的目标有足够的交集
        trace_d = 1000*np.ones(lidar_predict.shape[0], np.int32)
        for i in range(lidar_predict.shape[0]):
            point_idex, dx_nearest, dy_nearest, distance_nearest = get_nearest_point(target[2:6], lidar_predict[i,5:])
            #设定行人的跟踪门大小(x,y方向均为1.5m)
            if lidar_predict[i, 0] == 1:
                dx_thresh = 150
                dy_thresh = 150                   
            #根据目标位置设定车辆的跟踪门大小
            else:
                if 100 < target[0] < 1500:
                    dx_thresh = 180
                    dy_thresh = 500
                elif target[0] >= 1500 and -950 <= target[1] <= 600:
                    dx_thresh = 500
                    dy_thresh = 300
                else:
                    dx_thresh = 300
                    dy_thresh = 300
            if (dx_nearest < dx_thresh) and (dy_nearest < dy_thresh):
                trace_d[i] = distance_nearest
        if trace_d.shape[0] > 0 and np.min(trace_d) < 1000: # 小于1000表示有满足条件的trace_d
            idex = long(np.argmin(trace_d))  # 当前帧和上一帧第idex个目标可能是同一个
            if lidar_predict[idex, 0] == 1:
                distance_thresh = 100
            else:
                distance_thresh = 500
            # 将target放入对应的target_3Dlist中, target_3Dlist_num[idex]记录了target_3Dlist[idex]中存放了多少目标
            if trace_d[idex] < distance_thresh:
                target_3Dlist[idex, target_3Dlist_num[idex]] = target
                target_3Dlist_num[idex] += 1
        elif overlap_index is not None: # 另一种被认为是同一目标的情况
            if lidar_predict[overlap_index, 0] == 1:
                distance_thresh = 100
            else:
                distance_thresh = 500
            if trace_d[overlap_index] < 1.5*distance_thresh:
                # 如果和一次预测的target有足够的交集, 且最近点在一定范围之内 
                # 将target放入对应的target_3Dlist中...记录lidar_predict中的每个目标可能对应的所有lidar_data
                target_3Dlist[overlap_index, target_3Dlist_num[overlap_index]] = target
                target_3Dlist_num[overlap_index] += 1 # lidar_predict中的第overlap_index增加1
        else: # 认为是一个新目标 
            new_targets[j, :] = target
            j = j + 1
    return new_targets, target_3Dlist, target_3Dlist_num

@cython.boundscheck(False)
@cython.wraparound(False)
#xy:xmin,xmax,ymin,ymax
def box_modify(np.ndarray[np.int32_t, ndim=1] xy, np.ndarray[np.int32_t, ndim=1] lxy):
        # 根据l_xy修正 xy: xmin, ymin, xmax, ymax
        cdef np.ndarray[np.int32_t, ndim=2] box=np.zeros((5,2),np.int32), lbox=np.zeros((5,2),np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] cxy, lcxy, 
        cdef long idex, w, l, xmin, ymin, xmax, ymax 
        xy[:] = xy[[0,2,1,3]]
        lxy[:] = lxy[[0,2,1,3]]
        w = long(0.2*(xy[2]-xy[0]) + 0.8*(lxy[2]-lxy[0]))
        l = long(0.2*(xy[3]-xy[1]) + 0.8*(lxy[3]-lxy[1]))
        box[1,:], lbox[1,:] = xy[[0,1]], lxy[[0,1]]   # xmin, ymin
        box[2,:], lbox[2,:] = xy[[0,3]], lxy[[0,3]]   # xmin, ymax
        box[3,:], lbox[3,:] = xy[[2,3]], lxy[[2,3]]   # xmax, ymax
        box[4,:], lbox[4,:] = xy[[2,1]], lxy[[2,1]]   # xmax, ymin
        box[0,:], lbox[0,:] = (box[3,:] + box[1,:]) // 2, (lbox[3,:] + lbox[1,:]) // 2 # center
        idex = np.argmin(np.linalg.norm((lbox - box), axis=1)) # 距离最小的索引
        if idex == 0: # 中心点最接近
            xmin, ymin, xmax, ymax = box[0,0] - w//2, box[0,1] - l//2, box[0,0] + w//2, box[0,1] + l//2
        elif idex == 1:
            xmin, ymin, xmax, ymax = box[idex,0], box[idex,1], box[idex,0] + w, box[idex,1] + l
        elif idex == 2:
            xmin, ymin, xmax, ymax = box[idex,0], box[idex,1] - l, box[idex,0] + w, box[idex,1]
        elif idex == 3:
            xmin, ymin, xmax, ymax = box[idex,0] - w, box[idex,1] - l, box[idex,0], box[idex,1]
        elif idex == 4:
            xmin, ymin, xmax, ymax = box[idex,0] - w, box[idex,1], box[idex,0], box[idex,1] + l
        return np.asarray([xmin, xmax, ymin, ymax], np.int32)

@cython.boundscheck(False)
@cython.wraparound(False)
def heading(np.ndarray[np.float64_t, ndim=1] vxy, long thresh=300):
    # 根据速度向量求航向角
    cdef double vx = vxy[0], vy = vxy[1], heading
    if sqrt(pow(vx,2.) + pow(vy,2.)) < thresh: #
        return np.int32(3600)  # 速度小于thresh，输出一个无效值
    elif vy > 0: heading = atan(vx / vy)
    elif vy < 0: heading = atan(vx / vy) + M_PI
    else:
        if vx > 0: heading = M_PI / 2.
        elif vx < 0: heading = M_PI * (3 / 2.)
        else: heading = 2 * M_PI
    heading = np.degrees(heading) + 360 if -M_PI/2 < heading < 0 else np.degrees(heading)
    return long(heading*10) # 返回10倍角度值

@cython.boundscheck(False)
@cython.wraparound(False)
#ini_heading_array = [3309,1509,1959,2409,609,1059]
def get_ini_heading(np.ndarray[np.int32_t, ndim=1] target_position, np.ndarray[np.int32_t, ndim=1] ini_headings):
    cdef long px, py, ini_heading = 0
    px = target_position[0]
    py = target_position[1]
    if 100< px < 800:
        ini_heading = ini_headings[0]
    if (800 <= px <= 1400) or (px >=800 and py < -2200):
        ini_heading = ini_headings[1]
    if px > 1400 and -2200 <= py <= -950:
        ini_heading = ini_headings[2]
    if px > 1400 and -950 < py <= -200:
        ini_heading = ini_headings[3]
    if px > 1400 and -200 < py <= 600:
        ini_heading = ini_headings[4]
    if px > 1400 and 600 < py <= 2500:
        ini_heading = ini_headings[5]
    return ini_heading

@cython.boundscheck(False)
@cython.wraparound(False)
def lidar2UTM(np.ndarray[np.int32_t, ndim=1] coordinate, np.ndarray[np.float64_t, ndim=2] R_c2w, np.ndarray[np.int32_t, ndim=1] P0_UTM):
    coordinate_rotation = np.copy(coordinate)
    coordinate_rotation = np.dot(R_c2w, coordinate.T).transpose()
    if P0_UTM.any():
        coordinate_rotation = coordinate_rotation + P0_UTM
    return np.int32(coordinate_rotation)

@cython.boundscheck(False)
@cython.wraparound(False)
def LidarPoints2UTM(np.ndarray[np.int32_t, ndim=2] coordinate, np.ndarray[np.float64_t, ndim=2] R_c2w, np.ndarray[np.int32_t, ndim=1] P0_UTM):
    clouds_rotation = np.copy(coordinate)
    clouds_rotation[:,0:2] = np.dot(R_c2w, coordinate[:,0:2].T).transpose()
    if P0_UTM.any():
        clouds_rotation[:,0:2] = clouds_rotation[:,0:2] + P0_UTM
    return np.int32(clouds_rotation)

@cython.boundscheck(False)
@cython.wraparound(False)
#功能：对跟踪到的目标进行处理
#输入：
#    target_3Dlist:历史目标和当前目标的对应关系（目标对应关系）
#    target_3Dlist_num:历史目标和当前目标的对应关系（目标的对应数量关系）
#    lidar_state：状态信息（ID[0], class[1], xc[2], yc[3], vx[4], vy[5], xmin[6], xmax[7], ymin[8], ymax[9], 
#              zmin[10], zmax[11], I[12], life[13], heading[14]）
#    lidar_predict:class(0), xc(1), yc(2), vx(3), vy(4), xmin(5), xmax(6), ymin(7), ymax(8)
#输出：
#    selected：未被跟踪到的历史目标的索引
#    lidar_state:更新后的状态信息
#Edited by HOUXUEYUAN at 20210701
def tracked_targets_process(np.ndarray[np.int32_t, ndim=3] target_3Dlist, np.ndarray[np.int32_t, ndim=1] target_3Dlist_num, 
                    np.ndarray[np.int32_t, ndim=2] lidar_state, np.ndarray[np.int32_t, ndim=2] last_lidar_state, 
                    np.ndarray[np.int32_t, ndim=2] lidar_predict, long full_life, np.ndarray[np.float64_t, ndim=2] R_c2w, 
                    np.ndarray[np.int32_t, ndim=1] P0_UTM, double dt=0.1):
    cdef long i, idex, target_num, ll = 0
    cdef long area, area_predict = 0
    cdef long xmin, xmax, ymin, ymax, zmin, zmax, I = 0
    cdef np.ndarray[np.int32_t, ndim=1] selected, fusioned_target, fusioned_target_box, tmp_state, coordinate_rotation, last_coordinate_rotation
    cdef double last_speed = 0

    selected = -np.ones(target_3Dlist.shape[0], np.int32)
    # fusioned_target: xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
    fusioned_target = np.zeros(9, np.int32) 
    tmp_state = np.zeros(15, np.int32) # tmp_state.shape[0] = 15
    fusioned_target_box = np.zeros(4, np.int32)  #xmin, xmax, ymin, ymax
    
    cdef long static_thresh = 50
    ini_headings = np.asarray([3309,1509,1959,2409,609,1059], dtype='int32')

    for idex in range(target_3Dlist.shape[0]):
        targetx = target_3Dlist[idex, :target_3Dlist_num[idex]]
        if targetx.shape[0]>0:
            #多个目标匹配到同一个历史目标时，对多个目标进行融合
            xmin, ymin, zmin = targetx[0,2], targetx[0,4], 0
            xmax, ymax, zmax = targetx[0,3], targetx[0,5], targetx[0,7]
            I = targetx[0, 8]
            target_num = 1
            for i in range(1, targetx.shape[0]):
                xmin, xmax = min(xmin, targetx[i,2]), max(xmax, targetx[i,3])
                ymin, ymax = min(ymin, targetx[i,4]), max(ymax, targetx[i,5])
                zmax = max(zmax, targetx[i,7])
                I = I + targetx[i,7]
                target_num = target_num + 1
            I = I//target_num
            fusioned_target_box[0], fusioned_target_box[1] = xmin, xmax
            fusioned_target_box[2], fusioned_target_box[3] = ymin, ymax 
            #对前后帧变化较大的目标，根据历史信息，对其进行修正
            fusioned_target_box = box_modify(fusioned_target_box, lidar_predict[idex, 5:])
            fusioned_target[0] = (fusioned_target_box[0] + fusioned_target_box[1])//2 #xc
            fusioned_target[1] = (fusioned_target_box[2] + fusioned_target_box[3])//2 #yc
            fusioned_target[2], fusioned_target[3] = fusioned_target_box[0], fusioned_target_box[1] #xmin, xmax
            fusioned_target[4], fusioned_target[5] = fusioned_target_box[2], fusioned_target_box[3]
            
            #在人行道区域，class用最新的检测特征进行判断
            #1.设定人行道区域
            #2.判定是否在人行道区域
            #3.是，tmp_state[1]=lidar_class_new(fusioned_target)
            #4.否，tmp_state[1] = max(last_lidar_state[idex,1], lidar_class_new(fusioned_target))
            tmp_class = lidar_class_new(fusioned_target)
            last_speed = sqrt(pow(last_lidar_state[idex,4], 2.) + pow(last_lidar_state[idex,5], 2.))
            if (1800  <= fusioned_target[0] <=2500) or (1300 <= fusioned_target[1] <= 1950):
                if last_speed >= 300:
                    tmp_state[1] = 3
                else:
                    tmp_state[1] = tmp_class
            else:
                tmp_state[1] = 3
            tmp_state[0] = last_lidar_state[idex,0] #ID                
            tmp_state[2] = (fusioned_target_box[0] + fusioned_target_box[1])//2 #xc
            tmp_state[3] = (fusioned_target_box[2] + fusioned_target_box[3])//2 #yc
            coordinate_rotation = lidar2UTM(tmp_state[2:4], R_c2w, P0_UTM)
            last_coordinate_rotation = lidar2UTM(last_lidar_state[idex,2:4], R_c2w, P0_UTM)
            tmp_state[4] = long(0.5*last_lidar_state[idex,4] + 0.5*(coordinate_rotation[0]-last_coordinate_rotation[0])/dt) # Vx
            tmp_state[5] = long(0.5*last_lidar_state[idex,5] + 0.5*(coordinate_rotation[1]-last_coordinate_rotation[1])/dt) # Vy 
            tmp_state[6], tmp_state[7] = fusioned_target_box[0], fusioned_target_box[1] #xmin, xmax
            tmp_state[8], tmp_state[9] = fusioned_target_box[2], fusioned_target_box[3]            
            tmp_state[10], tmp_state[11], tmp_state[12]  = zmin, zmax, I  #反射强度平均值
            #车辆驶出检测区域后，生命周期清零
            xc = tmp_state[2]
            yc = tmp_state[3]
            if (xc < 800 and yc < -3600) or (xc > 800 and yc > 2300) or (xc > 2300 and (-900 < yc < -200)):
                tmp_state[13] = 0
            elif (xc < 800 and 0 < yc < 900):
                tmp_state[13] = 1
            else:
                tmp_state[13] = min(full_life, last_lidar_state[idex,13]+1)
            # 添加heading
            head = heading(np.float64(tmp_state[4:6]))
            lhead = last_lidar_state[idex, 14]
            if sqrt(pow(tmp_state[4],2.) + pow(tmp_state[5],2.)) <= static_thresh:
                tmp_state[14] = get_ini_heading(tmp_state[2:4], ini_headings)
            else:
                if head < 0:
                    pass
                elif head ==3600:
                    tmp_state[14] = lhead
                elif abs(lhead - head) < 1800:
                    tmp_state[14] = min(lhead+50, head) if (lhead < head) else max(lhead-50, head) # 连续两帧航向偏转不超过10°
                else:
                    tmp_state[14] = min(lhead+50, 3600+head) if (lhead > head) else max(3600+lhead-50, head)
            if tmp_state[14] >= 3600: 
                tmp_state[14] -= 3600             
            lidar_state[ll], ll = tmp_state, ll+1  # 将当前目标置入target_state中
            selected[idex] = idex
    selected = np.int32(np.where(selected < 0)[0]) # <0的都是没有找到被对应的
    return selected, lidar_state, ll

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class LidarMovement(object):
    cdef public double dt
    cdef public long next_ID, init_heading, init_life, full_life
    cdef public np.ndarray R_c2w, P0_UTM, last_lidar_state, lidar_position
    cdef public object track_process, tracked_targets_process, LidarPoints2UTM

    def __init__(self, np.ndarray[np.float64_t, ndim=2] R_c2w, np.ndarray[np.int32_t, ndim=1] P0_UTM,
                long ini_life, long full_life): 
        self.dt = 0.1
        self.next_ID = 1
        self.R_c2w = R_c2w
        self.P0_UTM = P0_UTM
        self.ini_life = ini_life
        self.full_life = full_life 
        self.last_lidar_state = np.empty((0,15), dtype=np.int32)
        self.lidar_position = np.zeros((2,4096,6))*np.nan
        self.lidar2UTM = lidar2UTM
        self.track_process = track_process
        self.tracked_targets_process = tracked_targets_process
        self.LidarPoints2UTM = LidarPoints2UTM

    def __call__(self, np.ndarray[np.int32_t, ndim=2] lidar_data):
        cdef np.ndarray[np.int32_t, ndim=2] lidar_state, lidar_UTM_state
        lidar_state = self.lidar_mv(lidar_data, self.last_lidar_state, self.lidar_position)
        self.last_lidar_state = np.copy(lidar_state)
        lidar_UTM_state = self.LidarState2UTM(lidar_state)
        return lidar_UTM_state

    def LidarState2UTM(self, np.ndarray[np.int32_t, ndim=2] lidar_state):
        cdef np.ndarray[np.int32_t, ndim=2] tmp_lidar_UTM_state
        tmp_lidar_UTM_state = np.copy(lidar_state[:,0:9])        
        tmp_lidar_UTM_state[:, 6] = lidar_state[:, 14] - 3600
        tmp_lidar_UTM_state[:, 7] = lidar_state[:, 7] - lidar_state[:, 6]
        tmp_lidar_UTM_state[:, 8] = lidar_state[:, 9] - lidar_state[:, 8]             
        tmp_lidar_UTM_state[:, 2:4] = self.LidarPoints2UTM(tmp_lidar_UTM_state[:, 2:4], self.R_c2w, self.P0_UTM)
        return tmp_lidar_UTM_state

    def lidar_mv(self, np.ndarray[np.int32_t, ndim=2] lidar_data, np.ndarray[np.int32_t, ndim=2] last_lidar_state, 
                np.ndarray[np.float64_t, ndim=3] lidar_position):
        cdef long i, j, ll = 0, dx, dy, idex
        cdef long xmin, xmax, ymin, ymax, zmin, zmax, I, X = 0, Y = 1
        cdef np.ndarray[np.int32_t, ndim=1] selected, tmp_state, targets_square, target, target_3Dlist_num, ini_headings
        cdef np.ndarray[np.int32_t, ndim=2] lidar_state, lidar_predict = np.copy(last_lidar_state[:,1:10])
        cdef np.ndarray[np.int32_t, ndim=3] target_3Dlist
        
        target = np.zeros(lidar_data.shape[1], np.int32) # lidar_data.shape[1] = 9
        lidar_state = np.zeros((lidar_data.shape[0] + last_lidar_state.shape[0], last_lidar_state.shape[1]), np.int32) # 14列
        tmp_state = np.zeros(last_lidar_state.shape[1], np.int32) # tmp_state.shape[0] = 15
        targets_square = np.zeros(lidar_data.shape[0], np.int32) # 雷达检测目标的占地面积
        target_3Dlist = np.zeros((last_lidar_state.shape[0], lidar_data.shape[0], lidar_data.shape[1]), np.int32)
        target_3Dlist_num = np.zeros(last_lidar_state.shape[0], np.int32)
        delta_t = self.dt*(lidar_position.shape[2]-np.int32(1)) # 用距离计算速度所需要的时间

        # 预测历史上每个目标的位置
        # lidar_predict: class(0), xc(1), yc(2), vx(3), vy(4), xmin(5), xmax(6), ymin(7), ymax(8)
        # 此处速度为UTM坐标系下的速度，位置为激光雷达坐标系下的位置，需要统一到UTM坐标系下
        for i in range(lidar_predict.shape[0]):  
            dx, dy = long(last_lidar_state[i,4]*self.dt), long(last_lidar_state[i,5]*self.dt)
            lidar_predict[i,1], lidar_predict[i,2] = lidar_predict[i,1]+dx, lidar_predict[i,2]+dy # xc, yc
            lidar_predict[i,5], lidar_predict[i,7] = lidar_predict[i,5]+dx, lidar_predict[i,7]+dy # xmin, ymin
            lidar_predict[i,6], lidar_predict[i,8] = lidar_predict[i,6]+dx, lidar_predict[i,8]+dy # xmax, ymax
        
        for i in range(lidar_data.shape[0]):  # 计算当前时刻被测物占地面积
            targets_square[i] = (lidar_data[i,3]-lidar_data[i,2])*(lidar_data[i,5]-lidar_data[i,4])
        lidar_data = lidar_data[np.argsort(-targets_square)] # 被测物面积从大到小排序

        #区分lidar_data中的新目标和跟踪成功的目标，并获取跟踪成功的目标与历史目标的对应关系
        new_targets, target_3Dlist, target_3Dlist_num = self.track_process(lidar_data, lidar_predict)
        
        #分别对新目标、跟踪到的目标、未跟踪到的目标进行处理，组建lidar_state    
        #1.处理跟踪到的目标
        selected, lidar_state, tracked_num = self.tracked_targets_process(target_3Dlist, target_3Dlist_num, lidar_state, 
                                                            last_lidar_state, lidar_predict, self.full_life, self.R_c2w, self.P0_UTM, self.dt)
        
        #2.处理未跟踪到的目标
        ll = tracked_num
        last_lidar_state = last_lidar_state[selected]
        lidar_predict = lidar_predict[selected]    
        for i in range(last_lidar_state.shape[0]):      
            tmp_state[13] = last_lidar_state[i,13] - 1
            if tmp_state[13] <= 0: 
                lidar_position[:, last_lidar_state[i,0], :] = np.nan
                continue  # 生命值终结，放弃跟踪
            tmp_state[0] = last_lidar_state[i,0] # ID, class
            tmp_state[1:10] = lidar_predict[i] # lidar_predict: class(0), xc(1), yc(2), vx(3), vy(4), xmin(5), xmax(6), ymin(7), ymax(8)
            tmp_state[10:13] = last_lidar_state[i,10:13] # zmin, zmax, I
            tmp_state[14] = last_lidar_state[i, 14]
            lidar_state[ll], ll = tmp_state, ll+1  # 将当前目标置入target_state中
        #3.处理新目标
        ini_headings = np.asarray([3309,1509,1959,2409,609,1059], dtype='int32')
        for i in range(new_targets.shape[0]):
            if((new_targets[i] == 0).all()):
                continue
            else:
                tmp_state[0] = -1  # 初始化
                for j in range(999): # 999次后，next_ID又回到原样                
                    if self.next_ID in last_lidar_state[:,0] or self.next_ID in lidar_state[:ll,0]: # 查看当前ID有无被占用
                        self.next_ID += 1 if self.next_ID < 999 else - 998
                    else:
                        tmp_state[0] = self.next_ID
                        break
                if tmp_state[0] == -1 : continue    
                tmp_state[1] = self.lidar_class_new(new_targets[i])  # 判断目标物类型
                tmp_state[2:4], tmp_state[6:13], tmp_state[13]= new_targets[i, 0:2], new_targets[i, 2:], self.ini_life  # 新目标假设速度为0            
                tmp_state[14] = self.get_ini_heading(tmp_state[2:4], ini_headings)
                lidar_state[ll], ll = tmp_state, ll+1  # 将当前目标置入target_state中
        
        for i in range(ll):
            lidar_ID = lidar_state[i,0]
            coordinate_rotation = self.lidar2UTM(lidar_state[i, 2:4], self.R_c2w, self.P0_UTM)
            for j in range(lidar_position.shape[2] + np.int32(1)):
                # j表示还未被赋值的列数 
                if j == lidar_position.shape[2] or np.isnan(lidar_position[X, lidar_ID, j]):
                    break  
            # 此时lidar_position未集齐足够position, 直接跟在后面赋值
            if j < lidar_position.shape[2]: 
                lidar_position[X, lidar_ID, j] = coordinate_rotation[0]
                lidar_position[Y, lidar_ID, j] = coordinate_rotation[1]
            else: 
                lidar_position[:, lidar_ID, 0:j-1] = lidar_position[:, lidar_ID, 1:j]
                lidar_position[X, lidar_ID, j-1] = coordinate_rotation[0] # xc
                lidar_position[Y, lidar_ID, j-1] = coordinate_rotation[1] # yc
                lidar_state[i,4] = long(0.8*(lidar_position[X, lidar_ID, j-1]-lidar_position[X, lidar_ID, 0])/delta_t +  \
                                        0.2*lidar_state[i,4]) # Vx
                lidar_state[i,5] = long(0.8*(lidar_position[Y, lidar_ID, j-1]-lidar_position[Y, lidar_ID, 0])/delta_t +  \
                                        0.2*lidar_state[i,5]) # Vy
        return lidar_state[:ll]
