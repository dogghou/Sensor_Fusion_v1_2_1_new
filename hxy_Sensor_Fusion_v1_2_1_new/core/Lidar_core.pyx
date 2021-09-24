# -*- coding: utf-8 -*-
# cython: cdivision=True
"""
Spyder Editor

This is a temporary script file.
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
    double round(double)
    double fabs(double)
    double M_PI  # 继承 pi

@cython.boundscheck(False)
@cython.wraparound(False)
def isinPolygon(int x, int y, np.float32_t[:,:] polygon_areas):
    # 判断是否在外包矩形内，如果不在，直接返回false
    cdef double xmax, ymax, xmin, ymin = 0
    cdef long i = 0
    xmax = polygon_areas[0][0]
    ymax = polygon_areas[0][1]
    xmin = polygon_areas[0][0]
    ymin = polygon_areas[0][1]
    
    for i in range(1, (polygon_areas.shape[0]-1)):
        if polygon_areas[i][0] > xmax:
            xmax = polygon_areas[i][0]
        if polygon_areas[i][0] < xmin:
            xmin = polygon_areas[i][0]
        if polygon_areas[i][1] > ymax:
            ymax = polygon_areas[i][1]
        if polygon_areas[i][1] < ymin:
            ymin = polygon_areas[i][1]

    # print(xmax , xmin, ymax, ymin)
    if (x > xmax or x < xmin or y > ymax or y < ymin):
        return False
    cdef long count = 0
    cdef long j = 0
    for j in range(polygon_areas.shape[0]):
        point1 = [polygon_areas[j][0], polygon_areas[j][1]]
        point2 = [polygon_areas[j][2], polygon_areas[j][3]]
        # 点与多边形顶点重合
        if (x == point1[0] and y == point1[1]) or (x == point2[0] and y == point2[1]):
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < y and point2[1] >= y) or (point1[1] >= y and point2[1] < y):
            # 求线段与射线交点 再和lat比较
            point12lng = polygon_areas[j][4] * y + polygon_areas[j][5]
            # 点在多边形边上
            if (point12lng == x):
                return False
            if (point12lng < x):
                count +=1
    if count%2 == 0:
        return False
    else:
        return True

@cython.boundscheck(False)
@cython.wraparound(False)
# range and ground filt
def clouds_range_no_ground(int transform_x, int transform_y, np.int32_t[:,:] clouds, np.int16_t[:,:] range_ground_z):
    #print(sizeof(short))
    #print(sizeof(int))
    #print(sizeof(long))
    #print(sizeof(long long))
    #print(sizeof(np.int16_t))
    #print(sizeof(np.int32_t))
    cdef long l, lp = 0, z_ground, x, y, z
    cdef np.int32_t[:,:] clouds_range_no_ground
    if clouds.shape[0] > 0:
        clouds_range_no_ground = cvarray(shape=(clouds.shape[0], 4), itemsize=sizeof(np.int32_t), format=i32)
    if clouds.shape[0] == 0:
        clouds_range_no_ground = cvarray(shape=(2, 4), itemsize=sizeof(np.int32_t), format=i32)
    for l in range(clouds.shape[0]): # 对clouds中的每一点
        if (clouds[l, 0] > -100) & (clouds[l, 0] < 4000) & (clouds[l, 1] > -4000) & (clouds[l, 1] < 3000) & (clouds[l, 2] < 1000):
            x = (clouds[l, 0]/1 + transform_x)//10
            y = (clouds[l, 1] + transform_y)//10
            z = clouds[l, 2]
            z_ground = range_ground_z[x][y]
            if (z_ground != 10000) & (z > (z_ground + 30)) & (z < (z_ground + 330)):
                clouds_range_no_ground[lp:lp+1,:], lp = clouds[l:l+1,:], lp+1
    return np.ctypeslib.as_array(clouds_range_no_ground[:lp])

@cython.boundscheck(False)
@cython.wraparound(False)
def downsample_circle(np.int16_t[:,:] xel, np.int32_t[:] idex): # 依附于降采样下的循环操作
    cdef long i, j, k, lp = 0
    cdef np.int16_t[:,:]  tmp, points_downsample = np.empty((xel.shape[0], 4), np.int16) # mx, my, mz, sumI
    cdef np.int32_t[:] tsum = cvarray(shape=(4,), itemsize=sizeof(np.int32_t), format=i32)
     
    for i in range(idex.shape[0]):
        if i+1 < <int>idex.shape[0]: tmp = xel[idex[i]:idex[i+1]]
        else: tmp = xel[idex[i]:]
        tsum[...] = 0
        for j in range(tmp.shape[0]):
            for k in range(4): tsum[k] += tmp[j,k] # x, y, z, I取平均
        for k in range(4):
            points_downsample[lp,k] = tsum[k]//(j+1)  # (j+1) = tmp.shape[0]            
        lp = lp+1
    return np.ctypeslib.as_array(points_downsample[:lp])


@cython.boundscheck(False)
@cython.wraparound(False)    
def grid_circle(np.int16_t[:,:] voxel, np.int32_t[:] idex): # 依附于栅格化下的循环操作
    cdef long i, j, m, n, lp=0, zmax, zmin
    cdef np.int16_t[:,:] tmp, points_grid = np.empty((voxel.shape[0],7), np.int16) # 'sx','sy','x','y','z','zmax','I'  
 
    for i in range(idex.shape[0]):
        zmax, zmin = -1000, 1000  # 赋予zmax, zmin初始值
        if i+1 < idex.shape[0]: tmp = voxel[idex[i]:idex[i+1]]  # sx, sy, x, y, z, I
        else: tmp = voxel[idex[i]:]
        for j in range(tmp.shape[0]):   
            zmax = max(zmax, tmp[j,4])
            zmin = min(zmin, tmp[j,4])
        # if (zmax-zmin < 40) or (zmax < 80): continue  # 则把tmp赋值给points_update
        for m in range(tmp.shape[0]):
            for n in range(5): points_grid[lp+m,n] = tmp[m,n]
            points_grid[lp+m,5] = zmax
            points_grid[lp+m,6] = tmp[m,5]
        lp += <long>(tmp.shape[0])            
    return np.ctypeslib.as_array(points_grid[:lp])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int is_in(long x, np.int32_t[:] X):
    cdef int i
    for i in range(X.shape[0]):
        if x == X[i]: return 1
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.int32_t[:] delete_1d(np.int32_t[:] X, np.int32_t[:] idex, int axis=0):
    cdef int i, lx = <int>X.shape[0], lo=0
    cdef np.int32_t[:] output = X # cvarray(shape=(lx,), itemsize=sizeof(long), format='l')
    if axis == 0:
        for i in range(lx):
            if is_in(i,idex): continue
            output[lo:lo+1] = X[i:i+1]; lo += 1    
    return output[:lo]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.int32_t[:] cluster_circle2(np.int16_t[:,:,:] points_slice, np.int16_t[:,:] grid_slice, np.int16_t[:,:] target_points, 
                    np.int32_t[:] idex, np.int32_t[:] idex2, np.int32_t[:] split, np.int32_t[:] num):
    # points_slice: sx[0], sy[1], x[2], y[3], z[4], zmax[5], I[6]
    # grid_slice: sx, sy, zmax
    # target_points: sx[0], sy[1], x[2], y[3], z[4], zmax[5], I[6], 被认为是同一目标的点云集合
    cdef long i, j, lp2, lr2 # lps = <long>points_slice.shape[1]
    cdef np.int16_t[:] seed_grid2 = cvarray(shape=(3,), itemsize=sizeof(short), format='h') # 次级栅格种子
    cdef np.int32_t[:] idex_round2 = cvarray(shape=(8,), itemsize=sizeof(np.int32_t), format=i32) # 次级栅格的邻域索引
    cdef np.int32_t[:] idex2_2 = cvarray(shape=(idex.shape[0],), itemsize=sizeof(np.int32_t), format=i32) # seed_grid2的邻域slice索引
    
    for i in range(idex2.shape[0]): # 对每一个领域种子索引
        seed_grid2[:] = grid_slice[idex2[i]:idex2[i]+1]  # 得到次一级种子栅格
        lp2 = split[idex2[0]+1] - split[idex2[0]]
        # for j in range(lps): # 找到target_points中当前存入点的数目
        #     if points_slice[idex2[i], j, 2] == 0 and points_slice[idex2[i], j, 3] == 0:  # x, y 方向为零 
        #         break
        #     lp2 += 1
        # tmp2 = points_slice[idex2[i]:idex2[i]+1] # 该邻域栅格所包含的点云
        target_points[num[0]:num[0]+lp2] = points_slice[idex2[i]:idex2[i]+1,0:lp2]; num[0] += lp2 # 把次级种子栅格的点云加入目标
        idex_round2[...], idex2_2[...], lr2 = 0, 0, 0 # 用于存放被选中的邻近栅格的的索引, lr2是idex_round2的索引

        for j in range(idex.shape[0]):
            if ((abs(grid_slice[idex[j],0]-seed_grid2[0]) <= 1) & (abs(grid_slice[idex[j],1]-seed_grid2[1]) <= 1) & # 如果属于邻近栅格
                (abs(grid_slice[idex[j],2]-seed_grid2[2]) <= 160)): # 且zmax相差小于100
                idex_round2[lr2], idex2_2[lr2], lr2 = j, idex[j], lr2+1     # 将idex2_2中的索引作为种子栅格，进行下一轮栅格筛选

        idex = delete_1d(idex, idex_round2[:lr2], axis=0) # 从idex中 删除 idex_round2的内容
        idex = cluster_circle2(points_slice, grid_slice, target_points, idex, idex2_2[:lr2], split, num) if idex.shape[0] else idex
    return idex

@cython.boundscheck(False)
@cython.wraparound(False)
def cluster_circle(np.int16_t[:,:,:] points_slice, np.int16_t[:,:] grid_slice, np.int32_t[:] idex, np.int32_t[:] split, long pn):
    # 对栅格化点云的父级聚类
    # points_slice: sx[0], sy[1], x[2], y[3], z[4], zmax[5], I[6]
    # grid_slice: sx, sy, zmax
    cdef long i, lr, halfcell=25, ln = 0, lp
    # cdef long lps = <long>points_slice.shape[1] # pn = <long>idex.shape[0]*lps 
    cdef long xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
    cdef np.int16_t[:,:] target_points = np.empty((pn,7), np.int16)
    cdef np.int32_t[:,:] target_output = np.empty((idex.shape[0],9), np.int32) # xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I 
    cdef np.int16_t[:] seed_grid = cvarray(shape=(3,), itemsize=sizeof(short), format='h')
    cdef np.int32_t[:] idex_round = cvarray(shape=(8,), itemsize=sizeof(np.int32_t), format=i32)
    cdef np.int32_t[:] idex2 = cvarray(shape=(idex.shape[0],), itemsize=sizeof(np.int32_t), format=i32) # seed_grid的邻域slice索引
    cdef np.int32_t[:] num = cvarray(shape=(1,), itemsize=sizeof(np.int32_t), format=i32) # 记录形成聚类的点云数目
    
    # t1 = time.time()
    while idex.shape[0]:  # 只要栅格数量 > 0    
        num[0] = 0 # 初始化'sx','sy','x','y','z','zmax','I'...
        seed_grid[:] = grid_slice[idex[0]:idex[0]+1]  # 找到一个种子栅格... sx, sy, zmax
        lp = split[idex[0]+1] - split[idex[0]]
        # for i in range(lps): # 每个slice有多少点云
        #     if points_slice[idex[0], i, 2] == 0 & points_slice[idex[0], i, 3] == 0:
        #         break
        #     lp += 1 # 每个slice的点云数量        
        target_points[:lp] = points_slice[idex[0]:idex[0]+1, :lp]; num[0] += lp # 栅格内点云赋给target_points,目标中点云数量加tmp.shape[0]        
        idex = idex[1:]; # idex2 = idex2[:idex.shape[0]]  # 索引中删除当前种子, 栅格数量减1
        idex_round[...], idex2[...], lr = 0, 0, 0 # 用于存放被选中的邻近栅格的的索引, lr是idex_round的索引
        
        for i in range(idex.shape[0]):  # 对剩下的idex中的每一个索引
            if ((abs(grid_slice[idex[i],0]-seed_grid[0]) <= 1) & (abs(grid_slice[idex[i],1]-seed_grid[1]) <= 1) & # 如果属于邻近栅格
                (abs(grid_slice[idex[i],2]-seed_grid[2]) <= 160)): # 且zmax相差小于100
                idex_round[lr], idex2[lr], lr = i, idex[i], lr+1  # 可以作为邻域种子栅格的索引, 由邻域形成的种子索引
        idex = delete_1d(idex, idex_round[:lr], axis=0)  # 从原索引中删除已经形成种子的索引 -> 剩下还没聚类的slice索引        
        idex = cluster_circle2(points_slice, grid_slice, target_points, idex, idex2[:lr], split, num) if idex.shape[0] else idex
        
        # print(np.ctypeslib.as_array(target_points))
        # target_points: sx[0], sy[1], x[2], y[3], z[4], zmax[5], I[6]
        xmin, ymin, zmin = target_points[0,2], target_points[0,3], target_points[0,4]
        xmax, ymax, zmax = target_points[0,2], target_points[0,3], target_points[0,4]
        I = target_points[0,6]        
        for i in range(1, num[0]):
            if target_points[i,2] == 0 and target_points[i,3] == 0: break
            xmin, xmax = min(xmin, target_points[i,2]), max(xmax, target_points[i,2])
            ymin, ymax = min(ymin, target_points[i,3]), max(ymax, target_points[i,3])
            zmin, zmax = min(zmin, target_points[i,4]), max(zmax, target_points[i,4])
            I += target_points[i,6]  # 反射强度求和
            # print ('I: ',I)  
        xc, yc = (xmin+xmax)//2, (ymin+ymax)//2
        xmin, xmax = min(xmin, xc - halfcell), max(xmax, xc + halfcell)
        ymin, ymax = min(ymin, yc - halfcell), max(ymax, yc + halfcell)
        zmin, zmax = 0, zmax
        target_output[ln,0], target_output[ln,1] = xc, yc
        target_output[ln,2], target_output[ln,3] = xmin, xmax
        target_output[ln,4], target_output[ln,5] = ymin, ymax
        target_output[ln,6], target_output[ln,7] = zmin, zmax
        target_output[ln,8] = I  # 每个聚类被检测出的总强度
        ln += 1  # 行组合, 确认的目标数+1
        # print (time.time()-t1)
    return np.ctypeslib.as_array(target_output[:ln])

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
    #print("big_box, smallbox",big_box, small_box)
    
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
    #print("overlap_ratio:", overlap_ratio)
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
    #print("target_fusion--target_new[8]:", target_new[8])    
    target_new[0] = (target_new[2] + target_new[3])//2
    target_new[1] = (target_new[4] + target_new[5])//2    
    target_width = max(target_new[3]-target_new[2], target_new[5]-target_new[4])
    if target_new[7]/target_width >= 4.5:
        target_new = None # 若目标及其细长（误检可能性大）
    return target_new

@cython.boundscheck(False)
@cython.wraparound(False)
def Lidar_class(np.ndarray[np.int32_t, ndim=1] state):
    # state: ID, class, xc, yc, vx, vy, xmin(6), xmax(7), ymin(8), ymax(9), zmin, zmax, I
    cdef long area
    area = (state[7]-state[6])*(state[9]-state[8])
    if area <= 10000: return 1
    elif area <= 40000: return 2
    else: return 3

@cython.boundscheck(False)
@cython.wraparound(False)
#Edited by HOUXUEYUAN at 20210630
def lidar_class_new(np.ndarray[np.int32_t, ndim=1] target):
        # target: xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
        cdef long area = 0
        cdef double aspect_ratio = 0
        cdef double delta_x, delta_y = 0
        #print("lidar_class_target", target)
        delta_x = (target[3]-target[2]) * 1.0
        delta_y = (target[5]-target[4]) * 1.0
        area = (target[3]-target[2])*(target[5]-target[4])
        aspect_ratio = delta_x/delta_y
        #print("target:", target)
        #print("area:", area)
        #print("aspect_ratio:", aspect_ratio)
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
    cdef long i, j, k, dx_nearest, dy_nearest, dx_thresh, dy_thresh, distance_thresh, distance_nearest, lidar_predict_area = 0
    cdef np.ndarray[np.int32_t, ndim=1] selected, target, trace_d, target_3Dlist_num, distance_list
    cdef np.ndarray[np.int32_t, ndim=2] new_targets
    cdef np.ndarray[np.int32_t, ndim=3] target_3Dlist

    target = np.zeros(lidar_data.shape[1], np.int32) # lidar_data.shape[1] = 9
    new_targets = np.zeros((lidar_data.shape[0], lidar_data.shape[1]), np.int32)
    target_3Dlist = np.zeros((lidar_predict.shape[0], lidar_data.shape[0], lidar_data.shape[1]), np.int32)
    target_3Dlist_num = np.zeros(lidar_predict.shape[0], np.int32) 
    
    #print("new_targets.shape[0], new_targets.shape[1]:", new_targets.shape[0], new_targets.shape[1])   
    j = 0
    while (lidar_data.shape[0]) > 0:
        target, selected = get_fusioned_target(lidar_data)
        lidar_data = np.delete(lidar_data, selected, axis=0)  # 从lidar_data中移除已经选中的
        if target is None: 
            continue 
        # target: xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
        #lidar_predict: class(0), xc(1), yc(2), vx(3), vy(4), xmin(5), xmax(6), ymin(7), ymax(8)
        overlap_index = is_overlap(target[2:6], lidar_predict[:,5:]) # 查看该目标是否与上一帧预测的目标有足够的交集
        trace_d = 1000*np.ones(lidar_predict.shape[0], np.int32)
        distance_list = np.zeros(lidar_predict.shape[0], np.int32)
        if lidar_predict.shape[0] > 0:
            for k in range(lidar_predict.shape[0]):
                point_idex, dx_nearest, dy_nearest, distance_list[k] = get_nearest_point(target[2:6], lidar_predict[k,5:])
            distance_list = distance_list[np.argsort(distance_list)]
        for i in range(lidar_predict.shape[0]):
            point_idex, dx_nearest, dy_nearest, distance_nearest = get_nearest_point(target[2:6], lidar_predict[i,5:])

            #设定行人的跟踪门大小(x,y方向均为1.5m)
            if lidar_predict[i, 0] == 1:
                dx_thresh = 100
                dy_thresh = 100                   
            #根据目标位置设定车辆的跟踪门大小
            else:
                if (distance_list.shape[0] == 1) or (distance_list[1] > 600):
                    dx_thresh = 300
                    dy_thresh = 500
                elif 100 < target[0] < 1500:
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
                distance_thresh = 400
            # 将target放入对应的target_3Dlist中, target_3Dlist_num[idex]记录了target_3Dlist[idex]中存放了多少目标
            if trace_d[idex] < distance_thresh:
                target_3Dlist[idex, target_3Dlist_num[idex]] = target
                target_3Dlist_num[idex] += 1
        elif overlap_index is not None: # 另一种被认为是同一目标的情况
            if lidar_predict[overlap_index, 0] == 1:
                distance_thresh = 100
            else:
                distance_thresh = 400
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
#功能：根据历史信息修正当前目标
#输入：
#    box:xmin,xmax,ymin,ymax
#    predict_box:xmin,xmax,ymin,ymax
#输出：
#    corrected_box：修正后的xmin,xmax,ymin,ymax
def correct(np.ndarray[np.int32_t, ndim=1] box, np.ndarray[np.int32_t, ndim=1] predict_box):
    # area_ratio:xy平面，面积比 box_area/predict_box_area
    cdef long box_area, predict_box_area = 0
    cdef double area_ratio = 0
    cdef np.ndarray[np.int32_t, ndim=1] corrected_box

    corrected_box = np.zeros(4,np.int32)

    box_area = (box[1] - box[0]) * (box[3] - box[2])
    predict_box_area = (predict_box[1] - predict_box[0]) * (predict_box[3] - predict_box[2])
    if (predict_box_area != 0) and (box_area != 0):
        area_ratio = box_area/predict_box_area
    else:
        print("predict_box or box is empty!")
        return box
    
    if (area_ratio < 0.5) or (area_ratio > 2):
        for i in range(4):
            corrected_box[i] = long(0.8*box[i] + 0.2*predict_box[i])
        return corrected_box
    else:
        return box

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
    # 6月25日改：输出无效值的阈值改为输入参数，默认值提高到800
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
#[[100,-4000],[100,2500],[1500,2500],[2000,1500],[3000,600],[3000,-950],[2000,-1700],[1500,-4000],[100, -4000]]
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
    cdef long heading_1 = 0
    cdef long area, area_predict = 0
    cdef long xmin, xmax, ymin, ymax, zmin, zmax, I = 0
    cdef np.ndarray[np.int32_t, ndim=1] selected, fusioned_target, fusioned_target_box, tmp_state, coordinate_rotation, last_coordinate_rotation
    cdef double last_speed = 0

    selected = -np.ones(target_3Dlist.shape[0], np.int32)
    # fusioned_target: xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
    fusioned_target = np.zeros(9, np.int32) 
    tmp_state = np.zeros(15, np.int32) # tmp_state.shape[0] = 15
    fusioned_target_box = np.zeros(4, np.int32)  #xmin, xmax, ymin, ymax
    #coordinate_rotation, last_coordinate_rotation = np.zeros(2, np.int32) 

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
            #fusioned_target_box = correct(fusioned_target_box, lidar_predict[idex, 5:])
            fusioned_target_box = box_modify(fusioned_target_box, lidar_predict[idex, 5:])

            fusioned_target[0] = (fusioned_target_box[0] + fusioned_target_box[1])//2 #xc
            fusioned_target[1] = (fusioned_target_box[2] + fusioned_target_box[3])//2 #yc
            fusioned_target[2], fusioned_target[3] = fusioned_target_box[0], fusioned_target_box[1] #xmin, xmax
            fusioned_target[4], fusioned_target[5] = fusioned_target_box[2], fusioned_target_box[3]

            tmp_state[0] = last_lidar_state[idex,0] #ID
            #在人行道区域，class用最新的检测特征进行判断
            #1.设定人行道区域
            #2.判定是否在人行道区域
            #3.是，tmp_state[1]=lidar_class_new(fusioned_target)
            #4.否，tmp_state[1] = max(last_lidar_state[idex,1], lidar_class_new(fusioned_target))
            tmp_class = lidar_class_new(fusioned_target)
            last_speed = sqrt(pow(last_lidar_state[idex,4], 2.) + pow(last_lidar_state[idex,5], 2.))
            #if last_speed >= 300:
            #    tmp_state[1] = max(last_lidar_state[idex,1], tmp_class)
            #elif (1900  <= fusioned_target[0] <=2500) or (1400 <= fusioned_target[1] <= 1950):
            #    tmp_state[1] = tmp_class
            #else:
            #    tmp_state[1] = max(last_lidar_state[idex,1], tmp_class) #class
            if (1800  <= fusioned_target[0] <=2500) or (1300 <= fusioned_target[1] <= 1950):
                if last_speed >= 300:
                    tmp_state[1] = 3
                else:
                    tmp_state[1] = tmp_class
            else:
                tmp_state[1] = 3
                            
            tmp_state[2] = (fusioned_target_box[0] + fusioned_target_box[1])//2 #xc
            tmp_state[3] = (fusioned_target_box[2] + fusioned_target_box[3])//2 #yc
            #print("tmp_state[2:4]:" ,tmp_state[2:4])
            #print("last_lidar_state[idex,2:4]:", last_lidar_state[idex,2:4])
            coordinate_rotation = lidar2UTM(tmp_state[2:4], R_c2w, P0_UTM)
            last_coordinate_rotation = lidar2UTM(last_lidar_state[idex,2:4], R_c2w, P0_UTM)
            #print("coordinate_rotation:", coordinate_rotation)
            #print("last_coordinate_rotation:", last_coordinate_rotation)
            tmp_state[4] = long(0.5*last_lidar_state[idex,4] + 0.5*(coordinate_rotation[0]-last_coordinate_rotation[0])/dt) # Vx
            tmp_state[5] = long(0.5*last_lidar_state[idex,5] + 0.5*(coordinate_rotation[1]-last_coordinate_rotation[1])/dt) # Vy 
            #tmp_state[4] = long(0.2*last_lidar_state[idex,4] + 0.8*(tmp_state[2]-last_lidar_state[idex,2])/dt) # Vx
            #tmp_state[5] = long(0.2*last_lidar_state[idex,5] + 0.8*(tmp_state[3]-last_lidar_state[idex,3])/dt) # Vy 
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
            elif (1800  <= xc <=2500) or (1300 <= yc <= 1950):
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
            if tmp_state[14] >= 3600: tmp_state[14] -= 3600 #
            
            #if tmp_state[0] == 4:
            #    print("heading_1:",heading_1)
            #    print("tmp_state:\n", tmp_state)
            #    print("heading:",tmp_state[14])
            lidar_state[ll], ll = tmp_state, ll+1  # 将当前目标置入target_state中
            selected[idex] = idex
    selected = np.int32(np.where(selected < 0)[0]) # <0的都是没有找到被对应的
    return selected, lidar_state, ll

@cython.boundscheck(False)
@cython.wraparound(False)
# lidar_data: xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
# lidar_state: ID[0], class[1], xc[2], yc[3], vx[4], vy[5], xmin[6], xmax[7], ymin[8], ymax[9], 
#              zmin[10], zmax[11], I[12], life[13], heading[14]
def lidar_movement(np.ndarray[np.int32_t, ndim=2] lidar_data, np.ndarray[np.int32_t, ndim=2] last_lidar_state, 
                   np.ndarray[np.float64_t, ndim=3] lidar_position, long ini_life, long full_life, long next_ID, 
                   np.ndarray[np.float64_t, ndim=2] R_c2w, np.ndarray[np.int32_t, ndim=1] P0_UTM, double dt=0.1): 
    #print("last_lidar_state.shape[1]:", last_lidar_state.shape[1])
    ini_life = -2
    full_life = 5
    cdef long lidar_ID = 0
    #print("next_ID:", next_ID)
    # next_ID = next_ID + 1
    #print("ini_life, full_life", ini_life, full_life)
    cdef long ini_heading = 1200
    cdef long i, j, ll = 0, dx, dy, idex, Kd_fusion = 50, Kd = 300
    cdef long xmin, xmax, ymin, ymax, zmin, zmax, I, X = 0, Y = 1
    cdef double area_predict, area 
    cdef np.ndarray[np.int32_t, ndim=1] selected, dr, tmp_state, targets_square, target, trace_d, target_3Dlist_num, ini_headings
    cdef np.ndarray[np.int32_t, ndim=2] fusion_target, lidar_state, targetx, trace_bias, lidar_predict = np.copy(last_lidar_state[:,1:10])
    cdef np.ndarray[np.int32_t, ndim=3] target_3Dlist
    
    target = np.zeros(lidar_data.shape[1], np.int32) # lidar_data.shape[1] = 9
    fusion_target = np.zeros((lidar_data.shape[0], lidar_data.shape[1]), np.int32)
    lidar_state = np.zeros((lidar_data.shape[0] + last_lidar_state.shape[0], last_lidar_state.shape[1]), np.int32) # 14列
    tmp_state = np.zeros(last_lidar_state.shape[1], np.int32) # tmp_state.shape[0] = 15
    targets_square = np.zeros(lidar_data.shape[0], np.int32) # 雷达检测目标的占地面积
    target_3Dlist = np.zeros((last_lidar_state.shape[0], lidar_data.shape[0], lidar_data.shape[1]), np.int32)
    target_3Dlist_num = np.zeros(last_lidar_state.shape[0], np.int32)
    delta_t = dt*(lidar_position.shape[2]-np.int32(1)) # 用距离计算速度所需要的时间

    # 预测历史上每个目标的位置
    # lidar_predict:id(0) class(1), xc(2), yc(3), vx(4), vy(5), xmin(6), xmax(7), ymin(8), ymax(9)
    for i in range(lidar_predict.shape[0]):  
        dx, dy = long(last_lidar_state[i,4]*dt), long(last_lidar_state[i,5]*dt)
        lidar_predict[i,1], lidar_predict[i,2] = lidar_predict[i,1]+dx, lidar_predict[i,2]+dy # xc, yc
        lidar_predict[i,5], lidar_predict[i,7] = lidar_predict[i,5]+dx, lidar_predict[i,7]+dy # xmin, ymin
        lidar_predict[i,6], lidar_predict[i,8] = lidar_predict[i,6]+dx, lidar_predict[i,8]+dy # xmax, ymax
    
    for i in range(lidar_data.shape[0]):  # 计算当前时刻被测物占地面积
        targets_square[i] = (lidar_data[i,3]-lidar_data[i,2])*(lidar_data[i,5]-lidar_data[i,4])
    lidar_data = lidar_data[np.argsort(-targets_square)] # 被测物面积从大到小排序

    #区分lidar_data中的新目标和跟踪成功的目标，并获取跟踪成功的目标与历史目标的对应关系
    new_targets, target_3Dlist, target_3Dlist_num = track_process(lidar_data, lidar_predict)
    
    #分别对新目标、跟踪到的目标、未跟踪到的目标进行处理，组建lidar_state    
    #1.处理跟踪到的目标
    #print("111111 ll:", ll)
    selected, lidar_state, tracked_num = tracked_targets_process(target_3Dlist, target_3Dlist_num, lidar_state, 
                                                        last_lidar_state, lidar_predict, full_life, R_c2w, P0_UTM, dt)
    
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
    #print("3333333 ll:", ll)

    #3.处理新目标
    #print("lidar_movement new_targets.shape[0]:", new_targets.shape[0])
    ini_headings = np.asarray([3309,1509,1959,2409,609,1059], dtype='int32')
    for i in range(new_targets.shape[0]):
        #print((new_targets[i] == 0).all())
        if((new_targets[i] == 0).all()):
            continue
        else:
            # zmin(10), zmax(11), I(12), life(13)
            #ID_list, lidar_ID = np.hstack((lidar_state[:ll,0], last_lidar_state[:,0])), 1  # 已经存在的ID, 初始赋值
            #while lidar_ID in ID_list: 
            #    lidar_ID += 1
            tmp_state[0] = -1
            for j in range(999): # 999次后，next_ID又回到原样
                if next_ID in last_lidar_state[:,0] or next_ID in lidar_state[:ll,0]: # 查看当前ID有无被占用
                    next_ID += 1 if next_ID < 999 else - 998
                else:
                    tmp_state[0] = next_ID
                    break
            # tmp_state[0] = lidar_ID 
            if tmp_state[0] == -1: continue   
            tmp_state[1] = lidar_class_new(new_targets[i])  # 判断目标物类型
            tmp_state[2:4], tmp_state[6:13], tmp_state[13]= new_targets[i, 0:2], new_targets[i, 2:], ini_life  # 新目标假设速度为0            
            # tmp_state[14] = ini_heading
            tmp_state[14] = get_ini_heading(tmp_state[2:4], ini_headings)
            # print(tmp_state[0],tmp_state[2:4], tmp_state[14])
            lidar_state[ll], ll = tmp_state, ll+1  # 将当前目标置入target_state中
            if next_ID < 999:
                next_ID = next_ID + 1
            else:
                next_ID = next_ID - 998
    
    for i in range(ll):
        lidar_ID = lidar_state[i,0]
        coordinate_rotation = lidar2UTM(lidar_state[i, 2:4], R_c2w, P0_UTM)
        for j in range(lidar_position.shape[2] + np.int32(1)): 
            if j == lidar_position.shape[2] or np.isnan(lidar_position[X, lidar_ID, j]):
                break  # j表示还未被赋值的列数
        if j < lidar_position.shape[2]: # 此时lidar_position未集齐足够position, 直接跟在后面赋值
            lidar_position[X, lidar_ID, j] = coordinate_rotation[0]
            lidar_position[Y, lidar_ID, j] = coordinate_rotation[1]
        else: # j = lidar_position.shape[2]
            lidar_position[:, lidar_ID, 0:j-1] = lidar_position[:, lidar_ID, 1:j]
            lidar_position[X, lidar_ID, j-1] = coordinate_rotation[0] # xc
            lidar_position[Y, lidar_ID, j-1] = coordinate_rotation[1] # yc
            lidar_state[i,4] = long(0.8*(lidar_position[X, lidar_ID, j-1]-lidar_position[X, lidar_ID, 0])/delta_t +  \
                                    0.2*lidar_state[i,4]) # Vx
            lidar_state[i,5] = long(0.8*(lidar_position[Y, lidar_ID, j-1]-lidar_position[Y, lidar_ID, 0])/delta_t +  \
                                    0.2*lidar_state[i,5]) # Vy
    return lidar_state[:ll], next_ID
