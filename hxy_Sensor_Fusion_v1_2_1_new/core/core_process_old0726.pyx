# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:12:36 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

Modified on 2021-5-18, v1.1.0

Modified by Lixiaohui
"""

import numpy as np
cimport numpy as np
import cython
cimport cython

from scipy.optimize import linear_sum_assignment


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
def isinpolygon(np.ndarray[np.int32_t, ndim=1] point, np.ndarray areas=np.empty(0)):
    # areas: xmin,ymin,xmax,ymax,k,b  
    def creat_areas(np.ndarray[np.float64_t, ndim=2] area):
        cdef np.ndarray[np.float64_t, ndim=2] areas
        cdef long la = long(area.shape[0])        
        if (area[la-1]-area[0]).any():
            area = np.vstack((area, area[0]))
        areas = np.zeros((la-1,6), np.float64)
        areas[:,:2], areas[:,2:4] = area[:la-1], area[1:]
        areas[:,4] = (areas[:,2]-areas[:,0])/(areas[:,3]-areas[:,1])
        areas[:,5] = areas[:,0] - areas[:,4]*areas[:,1]
        return areas   
    cdef long x0, x=point[0], y=point[1], i, j, cross=0            
    
    if areas.shape[0] < 3: return 1  # 区域不构成多边形
    elif areas.shape[1] == 2: # 只包含多边形顶点
        areas = creat_areas(areas)    
    
    for i in range(areas.shape[0]): # 判断是否在顶点上
        if x == areas[i,0] and y == areas[i,1]: return 1    
    for i in range(areas.shape[0]): # 对于不在顶点上的情形
        if (areas[i,0] < x) and (areas[i,2] < x): continue # 两个点皆该点左面
        elif (areas[i,1] > y) and (areas[i,3] > y): continue # 两个点皆在该点上面
        elif (areas[i,1] < y) and (areas[i,3] < y): continue # 两个点皆在下面
        elif abs(areas[i,4]) == np.inf: # 斜率无限大，线段两端y相等，线段与x轴平行且与该点射线重合于一条直线
            if areas[i,0] <= x <= areas[i,2]: 
                return 1  # 该点在多边型的边上
            # 其它情况下穿过的端点数必然为偶数
        else:
            x0 = long(round(areas[i,4]*y + areas[i,5]))
            if x0 == x: return 1
            elif x0 > x:
                if y == areas[i,1]: # 线段起点在射线上
                    j = i-1 if i > 0 else long(areas.shape[0])-1 # 寻找上一截的y
                    if (areas[i,3]-y) * (areas[j,1]-y) < 0: 
                        continue
                cross += 1      
    return cross%2


@cython.boundscheck(False)
@cython.wraparound(False)
def get_target_location(np.ndarray[np.float64_t, ndim=2]H, np.ndarray[np.int32_t, ndim=1] Pxy):
    # 利用单应性变换矩阵做坐标转移
    cdef np.ndarray[np.float64_t, ndim=1] Pw = np.zeros(3, np.float64)
    cdef long i, j
    Ps = np.float64([[Pxy[0]], [Pxy[1]], [1]])
    for i in range(3):
        for j in range(3):
            Pw[i] += H[i,j]*Ps[j]
    Pw[0] = round(Pw[0]/Pw[2])
    Pw[1] = round(Pw[1]/Pw[2])
    return Pw[:2].astype(np.int32)


@cython.boundscheck(False)
@cython.wraparound(False)
def same_type_fusion(np.ndarray[np.int32_t, ndim=2] target_output):
    # 将同类传感器测出的目标进行融合
    # target_output: ID(0), class(1), Xw(2), Yx(3), Vx(4), Vy(5), ini_heading(6)
    cdef np.ndarray[np.int32_t, ndim=2] final_output=np.zeros_like(target_output)
    cdef long i, j, lf=0, select
    for i in range(target_output.shape[0]):
        for j in range(lf):
            select = 1
            if target_output[i,0]//1000 == final_output[j,0]//1000: break
            if sqrt(pow(target_output[i,2]-final_output[j,2], 2) + pow(target_output[i,3]- \
                    final_output[j,3], 2)) <= 350: 
                select = 0
                break
        if lf==0 or select: 
            final_output[lf] = target_output[i]
            lf = lf + 1
    return final_output[:lf]


@cython.boundscheck(False)
@cython.wraparound(False)
def heading(np.ndarray[np.float64_t, ndim=1] vxy, long thresh=750):
    # 根据速度向量求航向角
    # 6月16日改：输出无效值的阈值改为输入参数，默认值提高到450
    cdef double vx = vxy[0], vy = vxy[1], heading
    if sqrt(pow(vx,2) + pow(vy,2)) < thresh: #
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
def speed_filter(np.ndarray[np.int32_t, ndim=2] state, np.ndarray[np.float64_t, ndim=3] position, 
                 double dt_base, long len_frame, str st = 'XY'):
    # 根据一定时间内的位置关系变化对速度进行滤波修正，state: ID, class, X, Y, Vx, Vy, heading
    # 6月15日改：修正了position赋值中的一个小错误
    cdef long i, j, lc = long(state.shape[0]), move, idex
    cdef np.ndarray[np.float64_t, ndim=1] tmp = np.zeros(2,np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] id_list
    cdef double delta_t, prop = 0  # 平均时间和比例
        
    for i in range(lc):
        id_list = np.where(np.isnan(position[0, state[i,0], :]))[0].astype(np.int32)
        idex = long(id_list[0]) if id_list.size else long(position.shape[2]) # 当前第一个np.nan的位置
        if idex == 0: # position中还未赋值过
            position[0, state[i,0], idex] = state[i,2]  # X
            position[1, state[i,0], idex] = state[i,3]  # Y
            continue
            
        if idex+len_frame-1 >= long(position.shape[2]): # 此时超出position范围                                   
            move = idex + len_frame - long(position.shape[2]) # 整体向左平移
            idex = idex - move
            for j in range(idex):  # 整体向左平移
                position[0, state[i,0], j] = position[0, state[i,0], move+j]
                position[1, state[i,0], j] = position[1, state[i,0], move+j]
                
        tmp[0] = (state[i,2]-position[0, state[i,0], idex-1])/len_frame  # 当前时刻何上一时刻的距离差
        tmp[1] = (state[i,3]-position[1, state[i,0], idex-1])/len_frame  # 除以间隔帧数
        for j in range(len_frame): # 依次赋值
            position[0, state[i,0], idex+j] = position[0, state[i,0], idex+j-1] + tmp[0]
            position[1, state[i,0], idex+j] = position[1, state[i,0], idex+j-1] + tmp[1]
        idex = idex + j
        delta_t = dt_base * idex # 用距离计算速度所需要的时间
        if delta_t < 0.5: prop = 0  # 由position计算出的速度比重为0
        elif 0.5 <= delta_t < 1: prop = 1   # 由position计算出的速度比重为1
        else: prop = 0.8

        state[i,4] = long(prop*(position[0, state[i,0], idex] - position[0, state[i,0], 0])/delta_t + \
                                                                (1-prop)*state[i,4]) if 'X' in st else state[i,4]
        state[i,5] = long(prop*(position[1, state[i,0], idex] - position[1, state[i,0], 0])/delta_t + \
                                                                (1-prop)*state[i,5]) if 'Y' in st else state[i,5]                
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DataFusion(object):
    # 6_11日改：建立DataFusion类，将融合和重新计算航向角分开
    cdef public np.ndarray ID_form
    cdef public object heading
    cdef public long next_idex
    
    def __init__(self):
        self.ID_form = -np.ones((1000, 6), np.int32) # ID, cameraID(s1), radar/lidarID(s2), class, heading, counter
        self.ID_form[:,0] = np.arange(1, long(self.ID_form.shape[0])+1)
        self.heading = heading
        self.next_idex = 0
    
    def __call__(self, np.ndarray[np.int32_t, ndim=2] s1_output, np.ndarray[np.int32_t, ndim=2] s2_output):
        # 将两个不同类别的传感器进行融合，其中s1的优先级高于s2，ID_from保留s1识别出的目标类型
        cdef np.ndarray[np.int32_t, ndim=2] target_state # ID, class, Xw, Yw, Vx, Vy, heading
        target_state = self.data_fusion(self.ID_form, s1_output, s2_output)
        self.heading_update(target_state, self.ID_form)
        return target_state
       
    def data_fusion(self, np.ndarray[np.int32_t, ndim=2] ID_form, np.ndarray[np.int32_t, ndim=2] s1_output,
                    np.ndarray[np.int32_t, ndim=2] s2_output):
        # s1_output/s2_output: ID, class, Xw, Yw, Vx, Vy, iniheading... iniheading表示道路航向
        # target_state: ID, class, Xw, Yw, Vx, Vy, heading
        # ID_form: ID[0], cameraID(s1)[1], radar/lidarID(s2)[2], class[3], heading[4], counter[5] 
        
        cdef np.ndarray[np.int32_t, ndim=2] target_state, S12_bias
        cdef np.ndarray[np.int32_t, ndim=1] tmp_state, S12_d
        cdef np.ndarray[np.float64_t, ndim=2] rM
        cdef double theta, sin_theta, cos_theta, Prop=0.6
        cdef long i, j, idex, lt = 0, counter=100
    
        # 初始化，行数为雷达、摄像头及历史检测目标之和，列数为7...不需要再进行生命跟踪
        target_state = np.zeros((ID_form.shape[0],7), np.int32) # ID(0), class(1), Xw(2), Yw(3), Vx(4), Vy(5), heading(6)
        tmp_state = np.zeros(target_state.shape[1], np.int32)  # tmp_state: ID(0), class(1), Xw(2), Yw(3), Vx(4), Vy(5), heading(6)
    
        for i in range(ID_form.shape[0]): # ID, s1, s2, class, heading, counter 
            ID_form[i,5] = max(0, ID_form[i,5]-1)  # 先让计时周期减1... 最小到0, 到0了就可以新分配ID了
        
        for i in range(s1_output.shape[0]): # 对s1中的每一个目标: ID[0]、class[1]、Xw[2]、Yw[3]、Vx[4]、Vy[5]、iniheading[6]
            tmp_state[0], idex = 0, -1  # 初始化匹配图像的索引， -1表示未匹配
            theta = np.deg2rad(s1_output[i,6]/10.)  # 道路的预设方向角（角度转弧度）
            sin_theta, cos_theta = sin(-theta), cos(-theta) # 顺时针是反方向, -theta... 或是求逆
            rM = np.array([[cos_theta, -sin_theta],  # 左乘矩阵
                           [sin_theta, cos_theta]]);
            S12_bias = s1_output[i,2:4] - s2_output[:,2:4]
            S12_bias = np.dot(S12_bias, rM).astype(np.int32) # (x',y')... y'方向是车辆行驶方向
            S12_d = 1000*np.ones(S12_bias.shape[0], np.int32) # 雷达与图像的绝对距离... 初始值1000cm
            for j in range(S12_bias.shape[0]):            
                if ((s1_output[i,1] >= 2 and (-200 <= S12_bias[j,0] <= 200) and (-450 <= S12_bias[j,1] <= 450)) or # 横向2.5m 纵向5m... 疑似同一目标
                   (s1_output[i,1] == 1 and (-150 <= S12_bias[j,0] <= 150) and (-150 <= S12_bias[j,1] <= 150))):  # 对人1.5m
                    S12_d[j] = long(sqrt(pow(S12_bias[j,0], 2) + pow(S12_bias[j,1], 2)))  # 计算两个传感器检测出的绝对距离
                
            if S12_d.size and min(S12_d) < 1000: # 同时被毫米波雷达和摄像头检测出来
                idex = long(np.argmin(S12_d)) # RC_d中最小值的索引, 即第i雷达目标和第idex个摄像头目标是同一个
                tmp_state[1] = s2_output[idex,1]  # class:类型都来自camera
                tmp_state[2] = long((1-Prop)*s2_output[idex,2] + Prop*s1_output[i,2])  # X进行融合
                tmp_state[3] = long((1-Prop)*s2_output[idex,3] + Prop*s1_output[i,3])  # Y进行融合
                tmp_state[4] = long((1-Prop)*s2_output[idex,4] + Prop*s1_output[i,4])  # Vx进行融合
                tmp_state[5] = long((1-Prop)*s2_output[idex,5] + Prop*s1_output[i,5])  # Vy进行融合
                tmp_state[6] = s1_output[i,6]  # 角度暂时设置为初始化... 即道路方向角
            else: # 
                for j in range(1,tmp_state.size):
                    tmp_state[j] = s1_output[i,j]  # 根据摄像头状态信息独自进行目标类型判断
                    
            # 对于这个目标，考虑上一时刻是否被检出
            for j in range(ID_form.shape[0]): # 该目标是否能基于图像进行跟踪？
                if tmp_state[0] > 0: break  # 已经跟踪到了，跳出循环
                elif s1_output[i,0] == ID_form[j,1]: # 若找到了基于图像的对应ID
                    tmp_state[0] = ID_form[j,0]  # 继承跟踪到的ID
                    ID_form[j,2] = -1 if idex < 0 else s2_output[idex,0]  # 重新匹配雷达ID... 若此时没匹配到雷达则置零
                    ID_form[j,3] = tmp_state[1] # 更新由图像识别到的类型
                    ID_form[j,5] = counter # 将计数器置为full
            for j in range(ID_form.shape[0]):  # 若对图像是一个新目标，则该目标是否能基于雷达进行跟踪？
                if tmp_state[0] > 0 or idex < 0: break  # 已经跟踪到了或当前帧没有匹配上雷达故不能基于雷达进行跟踪，跳出循环
                elif s2_output[idex,0] == ID_form[j,2]: # 若当前匹配的雷达ID在上一刻找到对应，则可以跟踪
                    tmp_state[0] = ID_form[j,0]  # 继承跟踪到的ID
                    ID_form[j,1] = s1_output[i,0] # 在当前将图像ID保存至ID列表                
                    ID_form[j,3] = tmp_state[1] # 更新由图像识别的类型
                    ID_form[j,5] = counter # 同时把计数置满
        
            for j in range(ID_form.shape[0]):  # 若对图像是一个新目标，又没能被雷达跟踪到，说明是一个新目标
                if tmp_state[0] > 0: break  # 分配了ID就跳出循环
                """
                elif ID_form[j,5] <= 0:  # 找到一个conter已经归零的ID号，进行ID分配
                    tmp_state[0] = ID_form[j,0] # 分配新ID
                    ID_form[j,1] = s1_output[i,0]  # 记录下图像的ID
                    ID_form[j,2] = -1 if idex < 0 else s2_output[idex,0]
                    ID_form[j,3] = tmp_state[1] # 更新由图像识别的类型
                    ID_form[j,5] = counter # 同时把计数置满
                """
                if ID_form[self.next_idex,5] > 0:
                    self.next_idex += 1 if self.next_idex < ID_form.shape[0] else - (ID_form.shape[0]-1)
                else:
                    tmp_state[0] = ID_form[self.next_idex,0] # 分配新ID
                    ID_form[self.next_idex,1] = s1_output[i,0]  # 记录下图像的ID
                    ID_form[self.next_idex,2] = -1 if idex < 0 else s2_output[idex,0]
                    ID_form[self.next_idex,3] = tmp_state[1] # 更新由图像识别的类型
                    ID_form[self.next_idex,5] = counter # 同时把计数置满
                
        
            if idex > -1: s2_output = np.delete(s2_output, idex, axis=0)
            for j in range(target_state.shape[1]):
                target_state[lt,j] = tmp_state[j]
            lt = lt + 1
        
        for i in range(s2_output.shape[0]): # 被s2检出而未被s1检出（被两者均检出的项已剔除）
            for j in range(tmp_state.size):
                if j == 0: tmp_state[j] = 0
                else: tmp_state[j] = s2_output[i,j]
        
            # 对于这个目标，考虑上一时刻是否被检出
            for j in range(ID_form.shape[0]):
                if tmp_state[0] > 0: break  # 跟踪到ID了就退出
                elif s2_output[i,0] == ID_form[j,2]:
                    tmp_state[0] = ID_form[j,0]
                    ID_form[j,1] = -1 # 没被图像识别到，所以置-1 
                    tmp_state[1] = ID_form[j,3] if ID_form[j,3] > 0 else tmp_state[1]  # 若目标曾被图像识别，则持续保留
                    ID_form[j,5] = counter # 同时把计数置满
                    
            for j in range(ID_form.shape[0]):
                if tmp_state[0] > 0: break
                """
                elif ID_form[j,5] <= 0: # 计数已经归零，表示可以使用该ID
                    tmp_state[0] = ID_form[j,0] # 分配ID
                    ID_form[j,1] = -1 # 没被图像识别到，所以置0
                    ID_form[j,2] = s2_output[i,0]  # 记录下雷达的ID
                    ID_form[j,5] = counter # 同时把计数置满
                """
                if ID_form[self.next_idex,5] > 0:
                    self.next_idex += 1 if self.next_idex < ID_form.shape[0] else - (ID_form.shape[0]-1)
                else:
                    tmp_state[0] = ID_form[self.next_idex,0] # 分配ID
                    ID_form[self.next_idex,1] = -1 # 没被图像识别到，所以置0
                    ID_form[self.next_idex,2] = s2_output[i,0]  # 记录下雷达的ID
                    ID_form[self.next_idex,5] = counter # 同时把计数置满
            
            for j in range(target_state.shape[1]):
                target_state[lt,j] = tmp_state[j]
            lt = lt + 1
        # print(ID_form[ID_form[:,1]>0, 0:2])
        return target_state[:lt]
    
    def heading_update(self, np.ndarray[np.int32_t, ndim=2] target_state, np.ndarray[np.int32_t, ndim=2] ID_form):
        # 根据速度重新计算航向角
        cdef long i, j, head, idex, lhead
    
        for i in range(target_state.shape[0]): # 对每个目标计算航向角
            if target_state[i,6] < 0:
                target_state[i,6] += 3600
                continue

            head = self.heading(np.float64(target_state[i,4:6]))  # 当前根据速度计算出的航向角
            idex = np.where(ID_form[:,0] == target_state[i,0])[0]
            lhead = long(ID_form[idex,4]) # 上一时刻的航向角

            if lhead < 0: pass  # 新检测到的目标，采用初始航向角
            elif head == 3600: # 如果算出来的角度是3600，无效航向，继承上一时刻的角度
                target_state[i,6] = lhead                    
            elif abs(lhead - head) < 1800:
                target_state[i,6] = min(lhead+100, head) if (lhead < head) else max(lhead-100, head) # 连续两帧航向偏转不超过10°
                # if target_state[i,6] >= 3600: target_state[i,6] -= 3600 # 
            else:
                target_state[i,6] = min(lhead+100, 3600+head) if (lhead > head) else max(3600+lhead-100, head)
                if target_state[i,6] >= 3600: target_state[i,6] -= 3600 #
            ID_form[idex,4] = target_state[i,6]  # 更新记录表里面的航向角
                
        for i in range(ID_form.shape[0]):
            if ID_form[i,5] == 3: # 大于该值时都保留ID,图像若临时遮挡不会跟丢 
                for j in range(1,5): ID_form[i,j] = -1  # 全部初始化            
        return None

        
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RadarMovement(object):
    # 6-10日改：将速度滤波部分单独提取出来
    # 6-11日改：radar_mv写入init_heading
    
    cdef public str usage
    cdef public long next_ID, init_heading, init_life, full_life
    cdef public np.ndarray P0_UTM, P0_radar, Homography, radar_position, last_radar_state, radar_areas, ID_form
    cdef public object isinpolygon, get_target_location, speed_filter
    
    def __init__(self, np.ndarray[np.int32_t, ndim=1] P0_UTM, np.ndarray[np.int32_t, ndim=1] P0_radar, np.ndarray[np.float64_t, ndim=2] Homography, \
                 long init_heading, np.ndarray areas, long init_life, long full_life, str usage='lane'):
        self.usage = usage
        self.next_ID = 1
        self.init_life, self.full_life = init_life, full_life
        self.P0_UTM, self.P0_radar = P0_UTM, P0_radar
        self.Homography = Homography
        self.init_heading = init_heading
        self.radar_position = np.zeros((2, 1000, 21)) * np.nan
        self.radar_areas = areas # x1,y1,x2,y2,k,b... x=k*y+b np.float64
        self.last_radar_state = np.empty((0, 9), dtype=np.int32)
        self.isinpolygon = isinpolygon
        self.get_target_location = get_target_location
        self.speed_filter = speed_filter
        self.ID_form = -np.ones(1000, np.int32)

    def __call__(self, np.ndarray[np.int32_t, ndim=2] radar_data, str supplier, double dt_base=0.05, long len_frame=1):
        # 6月22日改：对于前期赋予了ID的雷达，重新投影一遍ID的区间
        cdef np.ndarray[np.int32_t, ndim=2] radar_state, radar_UTM_state
        if 'yh' in supplier:
            radar_data = radar_data[:,[2,3,5,6]]  # X, Y, Vy, Pv(RCS)
            radar_state = self.radar_mv2(radar_data, self.last_radar_state, dt_base, len_frame)
        elif 'hes' in supplier: # ID, class, X, Y, Vx, Vy, (RCS)
            radar_data = self.id_reassign(radar_data) # ID, class, Xr, Yr, Vx, Vy, RCS, head, life
            radar_state = self.radar_mv1(radar_data, self.last_radar_state, dt_base, len_frame)
        self.speed_filter(radar_state, self.radar_position, dt_base, len_frame, st='X') # 仅对X方向速度滤波
        self.last_radar_state = np.copy(radar_state)
        radar_UTM_state = self.radar2UTM(radar_state)
        return radar_UTM_state

    def id_reassign(self, np.ndarray[np.int32_t, ndim=2] radar_state):
        # 将5位ID号缩减至1~999，若有重复，则保留life值更大的
        cdef np.ndarray[np.int32_t, ndim=2] radar_state_update = np.zeros_like(radar_state)
        cdef np.ndarray[np.int32_t, ndim=1] idex
        cdef long ID, i, j, lr = 0

        for i in range(radar_state.shape[0]):
            idex = np.where(self.ID_form == radar_state[i,0])[0].astype(np.int32)
            if idex.size : radar_state_update[lr,0] = idex[0]
            else: # 表示是一个新目标
                for j in range(999): # 999次后，做了一个1~999个ID的筛选
                    if self.ID_form[self.next_ID] > 0: # 若当前ID被占用
                        self.next_ID += 1 if self.next_ID < 999 else - 998 # ID + 1
                    else:
                        self.ID_form[self.next_ID] = radar_state[i,0] # 对 ID_form进行分派
                        radar_state_update[lr,0] = self.next_ID
                        break

            if radar_state_update[lr,0] > 0:
                for j in range(1, radar_state.shape[1]):
                    radar_state_update[lr,j] = radar_state[i,j]
                lr += 1

        for i in range(self.ID_form.shape[0]): # 先将不在radar_state中的置为-1
            self.ID_form[i] = -1 if (self.ID_form[i] > -1 and self.ID_form[i] not in radar_state[:,0]) \
                               else self.ID_form[i]

        return radar_state_update[:lr]

    def radar2UTM(self, np.ndarray[np.int32_t, ndim=2] radar_state):
        cdef np.ndarray[np.int32_t, ndim=2] radar_UTM_state = np.zeros_like(radar_state)  # ID, class, Xr, Yr, Vx, Vy, RCS, head, life
        cdef long i, lr=0
        for i in range(radar_state.shape[0]):
            Wxy = self.get_target_location(self.Homography, radar_state[i,2:4]) + self.P0_UTM
            if not self.isinpolygon(Wxy, self.radar_areas): continue  # 若不在筛选范围内（包括范围非封闭）
            radar_UTM_state[lr,0:2] = radar_state[i,0:2]
            radar_UTM_state[lr,2:4] = Wxy
            radar_UTM_state[lr,4:6] = get_target_location(self.Homography, radar_state[i,4:6]+self.P0_radar)
            radar_UTM_state[lr,6:] = radar_state[i,6:]
            lr += 1
        return radar_UTM_state[:lr]
                              
    
    def radar_mv1(self, np.ndarray[np.int32_t, ndim=2] radar_data, np.ndarray[np.int32_t, ndim=2] last_radar_state, \
                  double dt_base=0.05, long len_frame=1):
        # radar_data: ID, class, X, Y, Vx, Vy, (RCS)
        # last_radar_state/radar_state: ID(0), class(1), X(2), Y(3), Vx(4), Vy(5), RCS(6), head(7), life(8)
        cdef np.ndarray[np.int32_t, ndim=2] radar_state
        cdef np.ndarray[np.int32_t, ndim=1] tmp_state, idex
        cdef long i, j, lr=0
        cdef double dt = dt_base*len_frame
        
        radar_state = np.zeros((last_radar_state.shape[0] + radar_data.shape[0], \
                                last_radar_state.shape[1]), np.int32)
        tmp_state = np.zeros(radar_state.shape[1], np.int32) # ID, class, X, Y, Vx, Vy, rcs, head, life
        
        for i in range(radar_data.shape[0]): # 对雷达中的每一个目标
            idex = np.where(last_radar_state[:,0] == radar_data[i,0])[0].astype(np.int32)
            for j in range(6):  # ID(0), class(1), X(2), Y(3), Vx(4), Vy(5)
                tmp_state[j] = radar_data[i,j]
            tmp_state[6] = -1 if radar_data.shape[1] < 7 else radar_data[i,6]
            tmp_state[7] = self.init_heading
            tmp_state[8] = min(self.full_life, last_radar_state[idex,8]+1) if idex.size else self.init_life
            last_radar_state = np.delete(last_radar_state, idex, axis=0)
            for j in range(radar_state.shape[1]):
                radar_state[lr,j] = tmp_state[j]
            lr = lr+1        
        for i in range(last_radar_state.shape[0]): # 对于 last_radar_state 中剩下的任一目标
            last_radar_state[i,8] -= len_frame  # 生命值 -1
            if last_radar_state[i,8] <= 0: 
                for j in range(self.radar_position.shape[2]):
                    self.radar_position[0, last_radar_state[i,0], j] = np.nan
                    self.radar_position[1, last_radar_state[i,0], j] = np.nan
                continue  # 不再进行跟踪更新
            for j in range(radar_state.shape[1]):
                radar_state[lr,j] = last_radar_state[i,j]
            radar_state[lr,2] += long(radar_state[lr,4] * dt)
            radar_state[lr,3] += long(radar_state[lr,5] * dt)
            lr = lr+1
        return radar_state[:lr]
  
    def radar_mv2(self, np.ndarray[np.int32_t, ndim=2] radar_data, np.ndarray[np.int32_t, ndim=2] last_radar_state, \
                  double dt_base=0.05, long len_frame=1):
        # radar_data: X, Y, Vy, Pv
        # last_radar_state/radar_state: ID(0), class(1), Xr(2), Yr(3), Vx(4), Vy(5), Pv(6), head(7), life(8)
        def radar_class(np.ndarray[np.int32_t, ndim=1] state, str usage): # class, Xw, Yw, Vx, Vy, Pv, life
            if state[6] < 100 and 'ped' in usage: return 1
            else: return 3
        cdef np.ndarray[np.int32_t, ndim=2] radar_state, trace_bias, radar_predict= np.copy(last_radar_state[:,2:7]) # Xr, Yr, Vx, Vy, Pv
        cdef np.ndarray[np.int32_t, ndim=1] trace_d, tmp_state, idex
        cdef long i, j, lr = 0
        cdef long X_scale = 150, Y_scale = 300, Vy_scale = 200
        cdef double dt = dt_base*len_frame
    
        # 初始化雷达状态，行：当前检测目标数+历史检测目标数, 列数=8
        radar_state = np.zeros((last_radar_state.shape[0] + radar_data.shape[0], last_radar_state.shape[1]), np.int32)                
        for i in range(radar_predict.shape[0]): # target_predict: Xr, Yr, Vx, Vy, Pv
            radar_predict[i,0] += long(radar_predict[i,2] * dt)
            radar_predict[i,1] += long(radar_predict[i,3] * dt)
        tmp_state = np.zeros(radar_state.shape[1], np.int32) # ID, class, X, Y, Vx, Vy, Pv, head, life    
        # print ('predict:\n', radar_predict)   
        for i in range(radar_data.shape[0]): # 对radar中的每一个目标: X[0]、Y[1]、Vy[2]、Pv[3]
            trace_bias = np.zeros((radar_predict.shape[0],3), np.int32)  # X, Y, Vy
            trace_d = 1000*np.ones(trace_bias.shape[0], np.int32) # 当前目标与所有预测结果的距离
            for j in range(trace_bias.shape[0]): # X, Y, Vy
                trace_bias[j,0] = abs(radar_data[i,0]-radar_predict[j,0])  # X
                trace_bias[j,1] = abs(radar_data[i,1]-radar_predict[j,1])  # Y
                trace_bias[j,2] = abs(radar_data[i,2]-radar_predict[j,3])  # Vy...radar_data没有Vx
                if trace_bias[j,0] < X_scale and trace_bias[j,1] < Y_scale and trace_bias[j,2] < Vy_scale: # 若X, Y, V都小于尺度值
                    trace_d[j] = long(sqrt(pow(trace_bias[j,0], 2) + pow(trace_bias[j,1], 2)))
            if trace_d.shape[0] > 0 and np.min(trace_d) < 1000: # tmp_state: ID(0), class(1), X(2), Y(3), Vx(4), Vy(5), Pv(6), life(7)
                idex = np.int32([np.argmin(trace_d)])  # 即当前状态信息与第idex个预测值相差最小
                tmp_state[0] = last_radar_state[idex,0]  # 继承ID
                tmp_state[2] = radar_data[i,0] # X
                tmp_state[3] = radar_data[i,1] # Y            
                tmp_state[4] = long(0.2*(radar_data[i,0]-last_radar_state[idex,2])/dt + 0.8*last_radar_state[idex,4]) # Vx
                tmp_state[5] = radar_data[i,2] # Vy
                tmp_state[6] = long(0.2*radar_data[i,3] + 0.8*last_radar_state[idex,6])  # Pv
                tmp_state[8] = min(self.full_life, last_radar_state[idex,8]+1) # life update
                last_radar_state = np.delete(last_radar_state, idex, axis=0)
                radar_predict = np.delete(radar_predict, idex, axis=0)
            else: # 如果当前目标在上一帧不存在，即为新目标
                tmp_state[0] = -1  # 初始化
                for j in range(999): # 999次后，next_ID又回到原样
                    if self.next_ID in last_radar_state[:,0] or self.next_ID in radar_state[:lr,0]: # 查看当前ID有无被占用
                        self.next_ID += 1 if self.next_ID < 999 else - 998
                    else:
                        tmp_state[0] = self.next_ID
                        break
                if tmp_state[0] == -1 : continue # 说明当前已经同时存在999个目标, 不再加入新目标
                for j in range(2,7): # X(2), Y(3), Vx(4), Vy(5), Pv(6)    
                    if j < 4: tmp_state[j] = radar_data[i,j-2]
                    elif j==4: tmp_state[j] = 0 # 新目标x方向为0
                    else: tmp_state[j] = radar_data[i,j-3]            
                tmp_state[8] = self.init_life # life            
            tmp_state[1] = radar_class(tmp_state, self.usage) # class
            tmp_state[7] = self.init_heading
            # if 'ped' in self.usage and tmp_state[1] > 1: continue # 人行道雷达检测到车
            for j in range(radar_state.shape[1]):
                radar_state[lr,j] = tmp_state[j]
            lr = lr+1
    
        for i in range(last_radar_state.shape[0]): # 对于 last_radar_state 中剩下的任一目标
            last_radar_state[i,8] -= len_frame  # 生命值 -1
            if last_radar_state[i,8] < 1: 
                for j in range(self.radar_position.shape[2]):
                    self.radar_position[0, last_radar_state[i,0], j] = np.nan
                    self.radar_position[1, last_radar_state[i,0], j] = np.nan
                continue  # 不再进行跟踪更新
            for j in range(radar_state.shape[1]):
                radar_state[lr,j] = last_radar_state[i,j]
            radar_state[lr,2] = radar_predict[i,0]
            radar_state[lr,3] = radar_predict[i,1]
            lr = lr+1
        return radar_state[:lr]



@cython.boundscheck(False)
@cython.wraparound(False)
cdef class CameraMovement(object): # 以类的方式定义camera_movement
    cdef public long next_ID, init_heading, init_life, full_life
    cdef public np.ndarray P0_UTM, P0_camera, Homography, camera_position, last_camera_state, area_human, area_vehicle
    cdef public object isinpolygon, get_target_location, speed_filter

    def __init__(self, np.ndarray[np.int32_t, ndim=1] P0_UTM, np.ndarray[np.float64_t, ndim=2] Homography, long init_heading,
                 np.ndarray area_human, np.ndarray area_vehicle, long init_life, long full_life):
        self.init_life, self.full_life = init_life, full_life
        self.P0_UTM = P0_UTM
        self.Homography = Homography
        self.init_heading = init_heading
        self.camera_position = np.zeros((2, 1000, 36)) * np.nan
        self.area_human = area_human # x1,y1,x2,y2,k,b... x=k*y+b np.float64
        self.area_vehicle = area_vehicle
        self.last_camera_state = np.empty((0,8), dtype=np.int32) # ID(0), class(1), X(2), Y(3), Vx(4), Vy(5), head(6), life(7)
        self.isinpolygon = isinpolygon
        self.get_target_location = get_target_location
        self.speed_filter = speed_filter
        
    def __call__(self, np.ndarray[np.int32_t, ndim=2] camera_data, double dt_base=0.04, long len_frame=1):
        cdef np.ndarray[np.int32_t, ndim=2] camera_UTM, camera_UTM_state
        # camera_data = self.camera_select(camera_data)
        camera_UTM = self.camera2UTM(camera_data)
        # print('camera_UTM: ', camera_UTM)
        camera_UTM_state = self.camera_mv(camera_UTM, self.last_camera_state, dt_base, len_frame)
        self.speed_filter(camera_UTM_state, self.camera_position, dt_base, len_frame)
        self.last_camera_state = np.copy(camera_UTM_state)
        return camera_UTM_state
    
    def camera_select(self, np.ndarray[np.int32_t, ndim=2] camera_data):
    # 根据定义的范围筛选图像中的目标
        cdef long i, j, select, lc=0
        cdef np.ndarray[np.int32_t, ndim=2] camera_new_data = np.zeros_like(camera_data)
        for i in range(camera_data.shape[0]):
            if camera_data[i,1] == 0: continue
            select = self.isinpolygon(camera_data[i,2:], self.area_human) if camera_data[i,1] == 1 else \
                     self.isinpolygon(camera_data[i,2:], self.area_vehicle)
            if not select: continue
            for j in range(4):
                camera_new_data[lc,j] = camera_data[i,j]
            lc += 1        
        return(camera_new_data[:lc])
        
    
    def camera2UTM(self, np.ndarray[np.int32_t, ndim=2] camera_data): # ID, class, X, Y
        cdef np.ndarray[np.int32_t, ndim=2] camera_UTM = np.zeros_like(camera_data)  # ID, class, Xr, Yr
        cdef long i
        for i in range(camera_data.shape[0]):
            camera_UTM[i,0:2] = camera_data[i,0:2]
            camera_UTM[i,2:4] = self.get_target_location(self.Homography, camera_data[i,2:4])
        return camera_UTM

    
    def camera_mv(self, np.ndarray[np.int32_t, ndim=2] camera_UTM, np.ndarray[np.int32_t, ndim=2] last_camera_state, \
                  double dt_base=0.04, long len_frame=1):
        # camera_UTM: ID, class, X, Y
        # last_camera_state: ID(0), class(1), X(2), Y(3), Vx(4), Vy(5), head(6), life(7)
        cdef np.ndarray[np.int32_t, ndim=2] camera_state
        cdef np.ndarray[np.int32_t, ndim=1] tmp_state, idex
        cdef long i, j, lc = 0
        cdef double dt = dt_base*len_frame, delta_t
            
        camera_state = np.zeros((last_camera_state.shape[0] + camera_UTM.shape[0], last_camera_state.shape[1]), np.int32)
        tmp_state = np.zeros(camera_state.shape[1], np.int32) # ID(0), class(1), X(2), Y(3), Vx(4), Vy(5), head(6), life(7)
    
        for i in range(camera_UTM.shape[0]):
            for j in range(4): # ID, class, X, Y
                tmp_state[j] = camera_UTM[i,j] # tmp_state[:4] = camera_UTM[i]... ID, class, X, Y   
            tmp_state[6] = self.init_heading
            idex = np.where(last_camera_state[:,0] == camera_UTM[i,0])[0].astype(np.int32) # 和上一帧相同的ID的索引
            if idex.size: # 找到跟踪目标            
                tmp_state[4] = long(0.2*(camera_UTM[i,2]-last_camera_state[idex,2])/dt + 0.8*last_camera_state[idex,4]) # Vx
                tmp_state[5] = long(0.2*(camera_UTM[i,3]-last_camera_state[idex,3])/dt + 0.8*last_camera_state[idex,5]) # Vy
                tmp_state[7] = last_camera_state[idex,6]+1 if last_camera_state[idex,6] == -1 else self.full_life
                last_camera_state = np.delete(last_camera_state, idex, axis=0)
            else:
                tmp_state[7] = self.init_life # 本帧出现的新目标
            for j in range(camera_state.shape[1]):
                camera_state[lc,j] = tmp_state[j]
            lc = lc+1
        for i in range(last_camera_state.shape[0]):
            last_camera_state[i,7] -= len_frame  # life 生命值 - len_frame
            if last_camera_state[i,7] <= 0: # life为零时
                for j in range(self.camera_position.shape[2]): # 清空对应的position
                    self.camera_position[0, last_camera_state[i,0], j] = np.nan
                    self.camera_position[1, last_camera_state[i,0], j] = np.nan
                continue
            for j in range(camera_state.shape[1]):
                camera_state[lc,j] = last_camera_state[i,j]
            camera_state[lc,2] += long(camera_state[lc,4] * dt)
            camera_state[lc,3] += long(camera_state[lc,5] * dt)
            lc = lc+1

        return camera_state[:lc]
