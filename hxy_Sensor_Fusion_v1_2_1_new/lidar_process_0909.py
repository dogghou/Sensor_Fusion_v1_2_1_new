# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import time, sys, os, psutil, platform, math
import socket, struct
import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt

from funs import draw_process, tanh, send_message_process
from multiprocessing import Pool, Process, Manager, Value, Lock, RawArray
from threading import Thread
from pynput.keyboard import Key, Listener
from config_operate import load_config, load_sys_config
from core.Lidar_core import downsample_circle, lidar_movement, isinPolygon, clouds_range_no_ground

i32 = "l" if 'win' in sys.platform else "i"
i16 = "h"

def ini_pcl(system):
    global vss, viewer
    # elif 'linux' in system:
    import pcl
    import pcl.pcl_visualization as viewer
    vss = viewer.PCLVisualizering('cloud')
    vss.AddCoordinateSystem(1.0)
    points_cloud = pcl.PointCloud()
    return points_cloud, vss, viewer

def draw_lidar_range(system, points_cloud_range, show_area=False):
    if show_area == False:
        return
    points_cloud_range = np.float32(points_cloud_range)
    points_cloud_range[:,:3] = points_cloud_range[:,:3]*0.01
    # elif 'linux' in system: 
    import pcl
    cloud_range = pcl.PointCloud()
    color = viewer.PointCloudColorHandleringCustom(cloud_range, 127, 127, 127)
    cloud_range.from_array(points_cloud_range[:,:3])
    vss.AddPointCloud_ColorHandler(cloud_range, color, 'range')
    vss.SpinOnce(1)        

def draw_clouds(system, points_cloud, points_show, lidar_state_upload):
    global vss, viewer
    # elif 'linux' in system:
       
    color = viewer.PointCloudColorHandleringCustom(points_cloud, 200, 255, 0)
    points_cloud.from_array((points_show[:,:3]*0.01).astype(np.float32))
    vss.RemovePointCloud(b'cloud', 0)
    vss.RemoveAllShapes(0)    
    vss.AddPointCloud_ColorHandler(points_cloud, color, 'cloud')

    for i in range(lidar_state_upload.shape[0]):
        target = lidar_state_upload[i]/100.
        text = "ID:%d" % lidar_state_upload[i, 0] + "class:%d" % lidar_state_upload[i, 1]
        # text = "ID_%d" % lidar_state_upload[i, 0]
        text_id = "text_%d" % i
        xmin,xmax,ymin,ymax,zmin,zmax = target[6],target[7],target[8],target[9],target[10],target[11]
        vss.AddCube(xmin,xmax,ymin,ymax,zmin,zmax,1,0,0,str(i))  # 单位m
        vss.add_text3D(text,(xmin,ymin,zmin),1.0,255,0,0,id=text_id,viewport=0)
    vss.SpinOnce(1)
        
def lidarshow_process(Clouds_Show_Raw, Lidar_Box_Raw, lidarshow, cloud_range=False, show_area=False, last_frame=0):
    global vss, viewer
    p = psutil.Process(os.getppid())  # 获取父进程对象
    show_area_state = False
    system = sys.platform
    while p.is_running(): # 父进程要存在
        if lidarshow.value < 0:
            show_area_state = False
            if 'vss' in globals():
                vss.Close()
                del vss           
            continue
        if 'vss' not in globals():
            points_cloud, vss, viewer = ini_pcl(system)
        if show_area and (not show_area_state):  # 如果需要显示图像范围且当前未显示范围
            draw_lidar_range(system, cloud_range, show_area)
            show_area_state = True
        clouds_data = np.copy(np.ctypeslib.as_array(Clouds_Show_Raw)) # 用ctypeslib从内存中读取数据
        box_data = np.copy(np.ctypeslib.as_array(Lidar_Box_Raw))
        lidar_state_upload = box_data[2:box_data[1]+2].reshape((-1,14))
        points_show = clouds_data[2:clouds_data[1]+2].reshape((-1,4))
        draw_clouds(system, points_cloud, points_show, lidar_state_upload)
        # while time.time()-tick <= 0.05:
        #     time.sleep(0.001)
        
def Lidar_parameter(Lidar_type, block_num, hz=10):
    def V_theta_creat(V_theta): 
        V_theta_row = [V_theta[x] for x in V_theta]
        if len(V_theta_row)==16:
           V_theta_row = V_theta_row*2
        V_theta_row = np.array(V_theta_row, np.float64)
        V_theta_deg = np.expand_dims(V_theta_row, 0).repeat(block_num, axis=0)    
        return np.deg2rad(V_theta_deg)
    
    parameters = dict()
    parameters['lowbit'] = np.array([x for x in range(4,100) if x%3==1])  # 低位索引
    parameters['highbit'] = np.array([x for x in range(4,100) if x%3==2]) # 高位索引
    parameters['attenbit'] = np.array([x for x in range(4,100) if x%3==0]) # 反射位索引
    
    # 镭神激光雷达参数
    # 16线激光雷达
    C16_151B_qc = 1
    C16_151B_V_theta = {0:-15, 1:1, 2:-13, 3:3, 4:-11, 5:5, 6:-9, 7:7, 8:-7, 
                        9:9, 10:-5, 11:11, 12:-3, 13:13, 14:-1, 15:15}  # 镭神16线激光雷达的垂直角度分布
    C16_151B_A_modify = [0]*32
    # 32线激光雷达参数
    C32_151C_qc = 0.25  # 0.25
    C32_151C_V_theta = {0:-18, 1:-15, 2:-12, 3:-10, 4:-8, 5:-7, 6:-6, 7:-5, 8:-4,
                        9:-3.33, 10:-2.66, 11:-3, 12:-2.33, 13:-2, 14:-1.33, 15:-1.66,
                        16:-1, 17:-0.66, 18:0, 19:-0.33, 20:0.33, 21:0.66, 22:1.33, 23:1,
                        24:1.66, 25:2, 26:3, 27:4, 28:6, 29:8, 30:11, 31:14}    
    # A1, A2, A3, A4 = [1, 159], [3, 207], [2, 68], [3, 46]
    # A1, A2, A3, A4 = 0.01*(A1[0]*256+A1[1]), 0.01*(A2[0]*256+A2[1]), 0.01*(A3[0]*256+A3[1]), 0.01*(A4[0]*256+A4[1])
    # A_modify = [A2,A1,A2,A1,A2,A1,A2,A1,A2,A1,A4,A3,A2,A1,A4,A3,
    #              A2,A1,A4,A3,A2,A1,A4,A3,A2,A1,A2,A1,A2,A1,A2,A1]
    
    C32_151C_A_modify = [9.75, 4.15, 9.75, 4.15, 9.75, 4.15, 9.75, 4.15, 9.75, 4.15, 8.14,
                         5.8 , 9.75, 4.15, 8.14, 5.8 , 9.75, 4.15, 8.14, 5.8 , 9.75, 4.15,
                         8.14, 5.8 , 9.75, 4.15, 9.75, 4.15, 9.75, 4.15, 9.75, 4.15]
       
    # A1, A2 = [0, 120], [1, 34]
    # A1, A2 = 0.01*(A1[0]*256+A1[1]), 0.01*(A2[0]*256+A2[1])
    # A_modify = [+A2,-A2,+A2,-A2,+A2,-A2,+A2,-A2,+A2,-A2,+A1,-A1,+A2,-A2,+A1,-A1,
    #             +A2,-A2,+A1,-A1,+A2,-A2,+A1,-A1,+A2,-A2,+A2,-A2,+A2,-A2,+A2,-A2]
    """
    C32_151C_A_modify = [ 2.9, -2.9,  2.9, -2.9,  2.9, -2.9,  2.9, -2.9,  2.9, -2.9,  1.2,
                         -1.2,  2.9, -2.9,  1.2, -1.2,  2.9, -2.9,  1.2, -1.2,  2.9, -2.9,
                          1.2, -1.2,  2.9, -2.9,  2.9, -2.9,  2.9, -2.9,  2.9, -2.9]
    """
    
    # 北科天绘激光雷达参数
    rad_speed = {5:0.0018, 10:0.0036, 20:0.0072}
    # 16线激光雷达
    RFans16_qc = 0.25
    RFans16_V_theta = {0:-15, 1:-13, 2:-11, 3:-9, 4:-7, 5:-5, 6:-3, 7:-1, 8:1, 
                       9:3, 10:5, 11:7, 12:9, 13:11, 14:13, 15:15} # 北科16线激光雷达垂直角分布
    RFans16M_V_theta = {0:-12, 1:-15, 2:-7.5, 3:-9.5, 4:-5, 5:-6, 6:-3, 7:-4, 8:-1, 
                        9:-2, 10:1, 11:0, 12:5, 13:3, 14:11, 15:8}
    RFans16_H_BETA = [6.01, 3.377]
    RFans16M_H_BETA = [1.325, -1.325]
    RFans16_delta_T = {0:0, 1:13.32, 2:3.33, 3:16.65, 4:6.66, 5:19.98, 6:9.99, 7:23.31, 8:26.64, 
                       9:39.96, 10:29.97, 11:43.29, 12:33.3, 13:46.62, 14:36.63, 15:49.95}
    RFans16M_delta_T = {0:0, 1:13.32, 2:3.33, 3:16.65, 4:6.66, 5:19.98, 6:9.99, 7:23.31, 8:26.64, 
                        9:39.96, 10:29.97, 11:43.29, 12:33.3, 13:46.62, 14:36.63, 15:49.95}
    # 32线激光雷达
    # 缺
    if Lidar_type == 'C16_151B':
        parameters['qc'] = C16_151B_qc
        parameters['V_theta'] = V_theta_creat(C16_151B_V_theta)
        parameters['A_modify'] = np.array(C16_151B_A_modify)
    elif Lidar_type == 'C32_151C':
        parameters['qc'] = C32_151C_qc
        parameters['V_theta'] = V_theta_creat(C32_151C_V_theta)
        parameters['A_modify'] = np.array(C32_151C_A_modify)
    elif Lidar_type == 'RFans16':
        parameters['qc'] = RFans16_qc
        parameters['V_theta'] = V_theta_creat(RFans16_V_theta)
        parameters['H_BETA'] = np.array(RFans16_H_BETA * 16)
        delta_T = [RFans16_delta_T[x] for x in RFans16_delta_T] * 2
        parameters['Azimuth_wt'] = (rad_speed[hz] * np.float64(delta_T)).reshape((1,1,32)).repeat(block_num,1) # (?,12,32)
    elif Lidar_type == 'RFans16M':
        parameters['qc'] = RFans16_qc
        parameters['V_theta'] = V_theta_creat(RFans16M_V_theta)
        parameters['H_BETA'] = np.array(RFans16M_H_BETA * 16)
        delta_T = [RFans16M_delta_T[x] for x in RFans16M_delta_T] * 2
        parameters['Azimuth_wt'] = (rad_speed[hz] * np.float64(delta_T)).reshape((1,1,32)).repeat(block_num,1) # (?,12,32)
    return parameters


def get_segment(area, opt='auto', e=5): # 未验证输入多个激光雷达时的效果
    areas = [area] if not isinstance(area, list) else area
    dl = np.zeros((len(areas), 4), np.int32)
    for l, area in enumerate(areas):
        angle, dis= area[0:2], area[3]-area[2]
        for i in range(4):
            for j in range(90*i, 90*(i+1)):
                d = dis[np.product(angle-j, axis=0) <= 0]
                dl[l,i] = dl[l,i] + d[0] if d.size > 0 else dl[l,i]
    dl = np.sum(dl, axis=0)
    pn = min(opt, 4) if isinstance(opt, int) else int(np.ceil(sum(dl) / (4500*90)))
    if pn == 2:
        std = [np.std([dl[[0,1]].sum(), dl[[2,3]].sum()]), np.std([dl[[0,3]].sum(), dl[[1,2]].sum()])] # 比较两种组合的标准差
        if np.argmin(std) == 0: # 沿y轴分
            segmentx = np.array([[-e, np.inf], [-np.inf, e]])
            segmenty = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
        else: # 沿x轴分
            segmentx = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
            segmenty = np.array([[-e, np.inf], [-np.inf, e]])
    elif pn == 3:
        std = [np.std([dl[[0,1]].sum(), dl[2], dl[3]]), np.std([dl[[1,2]].sum(), dl[0], dl[3]]), 
               np.std([dl[[2,3]].sum(), dl[0], dl[1]]), np.std([dl[[3,0]].sum(), dl[1], dl[2]])]
        if np.argmin(std) == 0: # 0 & 1象限组合
            segmentx = np.array([[-e, np.inf], [-np.inf, e], [-np.inf, e]])
            segmentx = np.array([[-np.inf, np.inf], [-np.inf, e], [-e, np.inf]])
        elif np.argmin(std) == 1: # 1 & 2象限组合
            segmentx = np.array([[-e, np.inf], [-np.inf, np.inf], [-np.inf, e]])
            segmenty = np.array([[-e, np.inf], [-np.inf, e], [-e, np.inf]])
        elif np.argmin(std) == 2: # 2 & 3象限组合
            segmentx = np.array([[-e, np.inf], [-e, np.inf], [-np.inf, e]])
            segmenty = np.array([[-e, np.inf], [-np.inf, e], [-np.inf, np.inf]])
        elif np.argmin(std) == 3: # 3 & 0象限组合
            segmentx = np.array([[-np.inf, np.inf], [-e, np.inf], [-np.inf, e]])
            segmenty = np.array([[-e, np.inf], [-np.inf, e], [-np.inf, e]])
    elif pn == 4:
        segmentx = np.array([[-e, np.inf], [-e, np.inf], [-np.inf, e], [-np.inf, e]])
        segmenty = np.array([[-e, np.inf], [-np.inf, e], [-np.inf, e], [-e, np.inf]])
    else: # pn == 1
        segmentx = segmenty = np.array([[-np.inf, np.inf]])
    return [segmentx, segmenty]

def Lidar_points(Lidar_type, lidar_ori_data, Azimuth, base_scale, Lidar_para):
    packet_num = lidar_ori_data.shape[0]
    if Lidar_type in ['C16_151B', 'C32_151C']:  # 镭神的旋转雷达
        Azimuth_diff = np.diff(Azimuth)  # (?, 11)
        if Azimuth_diff.any(axis=0).all(): # Azimuth_diff中没有全为0列
            Azimuth_diff = np.hstack((Azimuth_diff, Azimuth_diff[:,-1:]))
        else: 
            Azimuth_diff = np.hstack((Azimuth_diff, Azimuth_diff[:,-2:-1]))  # 第12列等于倒数第二列
            idex = np.where(Azimuth_diff.any(axis=0)==False)[0]  # 
            Azimuth_diff[idex] = Azimuth_diff[idex+1]
        Azimuth_diff[np.where(Azimuth_diff<0)] += 360
        Azimuth_bias = np.expand_dims(Azimuth_diff/32,2).repeat(32,axis=2) * base_scale # (?, 12, 32)
        Azimuth_scale = Azimuth_bias + np.expand_dims(Azimuth,2) + Lidar_para['A_modify'] + 90# (?, 12, 32)
    elif Lidar_type in ['RFans16', 'RFans16M']:              
        Azimuth_bias = Lidar_para['Azimuth_wt'] + Lidar_para['H_BETA']
        Azimuth_scale = Azimuth_bias + np.expand_dims(Azimuth,2)
    Alpha = np.deg2rad(Azimuth_scale)
    V_theta = np.expand_dims(Lidar_para['V_theta'],0).repeat(packet_num,axis=0) # (?, 12, 32)    
    Distance = Lidar_para['qc']*(lidar_ori_data[:,:,Lidar_para['highbit']]*256+lidar_ori_data[:,:,Lidar_para['lowbit']]);
    Intensity = lidar_ori_data[:,:,Lidar_para['attenbit']]
    clouds, Azimuth_scale = get_points(Distance, Alpha, V_theta, Intensity, Azimuth_scale)
    return clouds, Azimuth_scale

def get_points(Distance, Alpha, V_theta, Intensity, Azimuth_scale):
    Distance = Distance.ravel()
    select = np.where(Distance > 0)[0]
    points = np.zeros((4,select.shape[0]), np.int16)    
    Distance = Distance[select]
    Alpha = Alpha.ravel()[select]
    V_theta = V_theta.ravel()[select]
    Azimuth_scale = Azimuth_scale.ravel()[select]
    points[0] = Distance*np.cos(V_theta)*np.sin(Alpha)
    points[1] = Distance*np.cos(V_theta)*np.cos(Alpha)
    points[2] = Distance*np.sin(V_theta)
    points[3] = Intensity.ravel()[select]
    return points.transpose(), Azimuth_scale
    
def points_downsample(clouds_range, cellsize=10):  # 降采样，减少数据量
    # print("downsample_cellsize:", cellsize)       
    clouds_downsample = np.zeros((0,4), np.int16)
    if clouds_range.shape[0] > 0:
        voxel = np.hstack([(np.ceil(clouds_range[:,0:3]/cellsize)).astype(np.int16), clouds_range]) # dx,dy,dz,x,y,z,I
        voxel = voxel[np.lexsort([voxel[:,2],voxel[:,1],voxel[:,0]])]  # 依次按dx,dy,dz进行排序
        _, idex = np.unique(voxel[:,:3], axis=0, return_index=True)
        idex = np.int32(idex) if idex.dtype != np.int32 else idex
        # voxel_diff_bool = np.insert(np.any(np.diff(voxel[:,0:3], axis=0), axis=1), 0, True)
        # idex = np.where(voxel_diff_bool)[0].astype(np.int32) # 按降采样尺度分组，每组第1个point的索引
        clouds_downsample = downsample_circle(voxel[:,3:], idex)  # 循环提取每一组的平均x,y,z,I得到降采样的结果
    return clouds_downsample

def points_grid(clouds_downsample, cellsize=50): # 栅格化
    # print("grid_cellsize:", cellsize) 
    from core.Lidar_core import grid_circle
    clouds_grid = np.zeros((0,7), np.int16)
    if clouds_downsample.shape[0] > 0:
        voxel = np.hstack((np.ceil(clouds_downsample[:,0:2]/cellsize).astype(np.int16), clouds_downsample)) # gx, gy, x,y,z,I
        voxel = voxel[np.lexsort([voxel[:,1],voxel[:,0]])]  # 将voxel按先x后y递增排序        
        _, idex = np.unique(voxel[:,:2], axis=0, return_index=True)
        idex = np.int32(idex) if idex.dtype != np.int32 else idex
        # voxel_diff_bool = np.insert(np.any(np.diff(voxel[:,[0,1]], axis=0), axis=1), 0, True)
        # idex = np.where(voxel_diff_bool)[0].astype(np.int32) # 按栅格化尺度分组，每组第1个point的索引
        clouds_grid = grid_circle(voxel, idex)  # 'sx','sy','x','y','z','zmax','I'
    return clouds_grid
    
def get_cluster(clouds_grid): 
    # clouds_grid: 'sx','sy','x','y','z','zmax','I'
    from core.Lidar_core import cluster_circle
    def np_fillna(data):
        lens =np.array([x.shape[0] for x in data])
        mask = np.arange(lens.max()) < lens[:,None]
        out = np.zeros(mask.shape+(7,), dtype=np.int16)
        out[mask] = np.concatenate(data)
        return out
    
    targets = np.zeros((0, 9), np.int32)
    if clouds_grid.shape[0] > 0:
        grid_slice, split = np.unique(clouds_grid[:,[0,1,5]], axis=0, return_index=True) # 栅格： sx, sy, (zmax?)
        points_slice = np_fillna(np.vsplit(clouds_grid, split[1:])) # 将clouds_grid按栅格切片
        idex = np.int32(np.arange(split.size))   # 一个idex为一个种子的索引
        # t1 = time.time()
        targets = cluster_circle(points_slice, grid_slice, idex, cellsize=50)  # xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
        # print(time.time()-t1)     
    return targets

def lidar2UTM(clouds_range, R_c2w, P0_UTM):
    clouds_rotation = np.copy(clouds_range)
    clouds_rotation[:,0:2] = np.dot(R_c2w, clouds_range[:,0:2].T).transpose()
    if P0_UTM.any():
        clouds_rotation[:,0:2] = clouds_rotation[:,0:2] + P0_UTM
    return np.int32(clouds_rotation)
        
def read_lidar_process(lock, addr, Lidar_Data_Raw, flag, online=True, lidar_connect_state = False, frame=0):    
    def is_nextcircle(a0, l_a0, Bool = False):        
        if l_a0!=None:  # 若上一时刻l_a0不为空
            return (18000<=l_a0<36000 and 0<=a0<18000)
    p = psutil.Process(os.getppid())  # 获取父进程对象
    data_list=list() # 存放一个周期的雷达data
    last_Azimuth_C0 = None
    Azimuth_idx = np.arange(12)*100
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # 创建 socket 对象, 设置端口
    
    tick = time.time()
    if not online:  # 如果不是在线读取数据
        addr = "/home/user/hxy/lidar_dataset/pcap/car_person_truck.pcap"
        fpcap = open(addr,'rb')
        string_data, i = fpcap.read(), 24
        pcap_packet_header = dict()  # 定义packet头
    while flag.value and time.time()-tick < 3 and p.is_running(): # 父进程要存在
        if not online: # 离线模式
            if i >= len(string_data): break
            pcap_packet_header['caplen'] = string_data[i+8:i+12]
            packet_len = struct.unpack('I',pcap_packet_header['caplen'])[0]
            packet_data, i = string_data[i+16:i+16+packet_len], i+16+packet_len
        elif online: # 在线模式
            if not lidar_connect_state:
                try: 
                    sock.bind(addr)
                    sock.setblocking(0)  # 非阻塞模式
                    lidar_connect_state = True # 若连接上了，则状态置为True
                    print ('{} is connected'.format(addr))   
                    tick = time.time()
                except:
                    time.sleep(0.2)
                    continue
            try: 
                packet_data, _ = sock.recvfrom(1248)
                #print("packet_data:", packet_data)
                if packet_data[1]: pass  # 防止packet为空
            except: 
                continue
                
        if len(packet_data) < 1206: # 一个数据包的长度
            # print ('miss one packet')
            continue
        data = np.frombuffer(packet_data, np.uint8)  # 按字节, 8位一个字节
        # data_tail = data[-6:] # 附加信息
        data_body = data[-1206:-6]  # 消息本体               
        Azimuth = data_body[Azimuth_idx+3]*256+data_body[Azimuth_idx+2] # 每个block第一个channel0的角度
        Azimuth[np.where(Azimuth>=36000)] -= 36000 
        Azimuth[np.where(Azimuth<Azimuth[0])] += 36000
        if np.std(Azimuth) > 512: continue  # 如果Azimuth出现了明显的波动，一般认为是坏帧
        Azimuth_C0 =  Azimuth[0] 
        # print (Azimuth_C0/100)
        is_next = is_nextcircle(Azimuth_C0, last_Azimuth_C0) # 判定当前packet是否已为下一个循环
        last_Azimuth_C0 = Azimuth_C0
        if is_next:  # 如果被判定为下一个循环
            tick = time.time()  # 正常解析出一帧, ，更新tick
            # print("tick", tick)
            tl, th = math.modf(tick)   # 分别取时间的小数位和整数位，单位s

            packet_body = np.int32(data_list).ravel()
            # packet_head = [frame+1, packet_body.size] # 将数据长度和当前帧数写进head
            packet_head = [frame+1, packet_body.size, int(th), int(tl*1000)] #添加时间戳
            packet = np.insert(packet_body, 0, packet_head)
            lock.acquire()  # 进程锁
            memoryview(Lidar_Data_Raw).cast('B').cast(i32)[:packet.size] = packet
            lock.release()  # 释放锁  
            frame = (frame+1) % 65535
            # print ('frame: ', frame)
            del data_list[:]  # 清空当前帧的数据列表
            while (not online) and (time.time()-tick < 0.099):
                time.sleep(0.001)
            tick = time.time() # 正常接收到一个周期，update tick
        data_list.append(data_body)
    if online:
        sock.close()
        del sock

def lidar_receive_process(lock, host_IP, lidar_conf, data_pickle, Cloud_Rotation_Raw, lidar_status, range_ground_z, flag, block_num=12, lidar_frame=0):
    Lidar_type = lidar_conf['usage']
    Lidar_IP = lidar_conf['IP']
    port = int(lidar_conf['port'])  #雷达发送数据至本机的端口
    addr = (host_IP, port)
    online = False    
    area = get_area(lidar_conf)     #雷达的点云筛选范围
    
    # try: P0_UTM = np.int32(data_pickle[Lidar_IP]['P0'][0])  # 一个方向上的原点
    # except: P0_UTM = np.int32([0,0])
    # try: R_c2w = data_pickle[Lidar_IP]['Calibration'][-1]['R_c2w']
    # except: R_c2w = np.eye(2)
    # print("P0_UTM:", P0_UTM)
    # print("R_c2w:", R_c2w)
    
    Lidar_para = Lidar_parameter(Lidar_type, block_num)
    base_scale = np.arange(32).astype(np.float32)  # 每行为1~32
    Lidar_Data_Raw = RawArray(i32, 1024*12*100)  # np.uint8...每个激光雷达原始数据的内存空间
    last_lidar_frame = 0   
    while flag.value:
        if 'p_read' not in vars():
            p_read = Process(target=read_lidar_process, args=(Lock(), addr, Lidar_Data_Raw, flag, online)) 
            p_read.start() # 读取雷达数据的进程
        if not p_read.is_alive():            
            del p_read
            continue
        
        data = np.copy(np.ctypeslib.as_array(Lidar_Data_Raw)) # 用ctypeslib从内存中读取数据
        if data[0] == last_lidar_frame:  # 如果当前帧数为上一帧...表示该帧已经读过
            continue
        last_lidar_frame = data[0]  # 重新赋值last_lidar_frame
        # lidar_ori_data = data[2:data[1]+2].reshape((-1,12,100))
        lidar_ori_data = data[4:data[1]+4].reshape((-1,12,100))
        # print("lidar_receive_process_time:",data[2], data[3])
        Azimuth = np.float64(lidar_ori_data[:,:,3]*256+lidar_ori_data[:,:,2])/100. # (?, 12)
        clouds, Azimuth_scale = Lidar_points(Lidar_type, lidar_ori_data, Azimuth, base_scale, Lidar_para)
        # clouds_rotation = np.copy(clouds)
        # clouds_rotation[:,0:2] = np.dot(R_c2w, clouds[:,0:2].T).transpose()
        # clouds_rotation[:,0:2] = clouds_rotation[:,0:2] + P0_UTM
        # clouds_rotation = lidar2UTM(clouds, R_c2w, P0_UTM)
        # print ('clouds_rotation: \n', clouds_rotation)
        packet_body = (np.int32(clouds)).ravel() # 旋转之后按P0_UTM进行了平移
        # packet_head = [lidar_frame+1, packet_body.size]
        packet_head = [lidar_frame+1, packet_body.size, data[2], data[3]]
        packet = np.insert(packet_body, 0, packet_head)
        lock.acquire()  # 进程锁
        memoryview(Cloud_Rotation_Raw).cast('B').cast(i32)[:packet.size] = packet
        #memoryview(Cloud_Rotation_Raw).cast('B').cast(i32)[:packet.size] = packet
        lock.release()  # 释放锁
        lidar_frame = (lidar_frame+1)%256
        # print ('lidar_frame: ', lidar_frame)
    
    if 'p_read' in vars():
        p_read.terminate()
        
def get_range_ground_z(transform_x, transform_y, rangelist, ground_z):
    # get polygon areas
    polygon_areas = np.zeros((rangelist.shape[0], 6), np.float32)
    for i in range (rangelist.shape[0]-1):
        x1, y1, x2, y2 = np.hstack((rangelist[i], rangelist[i+1]))
        if y2 == y1:
            y2 = y1 + 1
        k = (x2-x1)/(y2-y1)
        b = x1-k*y1
        polygon_areas[i] = [x1, y1, x2, y2, k, b]

    range_ground_z = np.ones((ground_z.shape[0], ground_z.shape[1]), np.int16)
    for j in range(ground_z.shape[0]):
        for k in range(ground_z.shape[1]):
            x = 10 * j - transform_x
            y = 10 * k - transform_y
            if isinPolygon(x, y, polygon_areas):
                range_ground_z[j][k] = ground_z[j][k]
            else:
                range_ground_z[j][k] = 10000
    return range_ground_z

def get_cloud_range(transform_x, transform_y, rangelist, ground_z): #for show area
    # get polygon areas
    polygon_areas = np.zeros((rangelist.shape[0], 6), np.float32)
    for i in range (rangelist.shape[0]-1):
        x1, y1, x2, y2 = np.hstack((rangelist[i], rangelist[i+1]))
        if y2 == y1:
            y2 = y1 + 1
        k = (x2-x1)/(y2-y1)
        b = x1-k*y1
        polygon_areas[i] = [x1, y1, x2, y2, k, b]
        
    cloud_range = np.zeros((ground_z.shape[0] * ground_z.shape[1],4), np.int16)
    
    count = 0
    for m in range(ground_z.shape[0]):
        for n in range(ground_z.shape[1]):
            x = 10 * m - transform_x
            y = 10 * n - transform_y
            if isinPolygon(x, y, polygon_areas):
                cloud_range[count][0] = 10 * m - transform_x
                cloud_range[count][1] = 10 * n - transform_y
                cloud_range[count][2] = ground_z[m][n]
                cloud_range[count][3] = 127
                count = count + 1
    
    return cloud_range  

def get_ground_z():
    f = open('./lidar_config/ground', 'rb')
    ground_z = pickle.load(f)
    return ground_z

def Lidar_detect(lock, lidar_IP_list, host_IP, location, Lidar_Target_Raw, flag, lidarshow, Lidar_status_dict, frame_num=0):
    show_area = False
    pickle_name = location.split('_')[0]
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data_pickle = pickle.load(pickle_file)
    Cloud_Raw_list = list()  # 存储激光雷达点云内存的列表
    #ID[0],class[1],xc[2],yc[3],vx[4],vy[5],xmin[6],xmax[7],ymin[8],ymax[9],zmin[10],zmax[11],I[12],life[13],heading[14
    last_lidar_state = np.empty((0,15), np.int32)
    lidar_position = np.zeros((2,4096,6))*np.nan    
    Clouds_Show_Raw = RawArray(i32, 1024*1024*5)  # 设置需要显示的激光雷达点云的内存空间
    Lidar_Box_Raw = RawArray(i32, 1024)  # 定义显示激光雷达目标瞄框内存

    Lidar_conf_0 = load_config(location, lidar_IP_list[0])
    Lidar_IP = Lidar_conf_0['IP']
    try: P0_UTM = np.int32(data_pickle[Lidar_IP]['P0'][0]*100)  # 一个方向上的原点
    except: P0_UTM = np.int32([0,0])
    try: R_c2w = data_pickle[Lidar_IP]['Calibration'][-1]['R_c2w']
    except: R_c2w = np.eye(2)
    print("P0_UTM", P0_UTM)

    xmin = -100
    ymin = -4000
    transform_x = -xmin
    transform_y = -ymin 
    rangelist = np.int16([[100,-4000],[100,2500],[1500,2500],[2000,1500],[3000,600],[3000,-950],[2000,-1700],[1500,-4000],[100, -4000]])

    ground_z = get_ground_z()
    cloud_range = get_cloud_range(transform_x, transform_y, rangelist, ground_z)
    range_ground_z = get_range_ground_z(transform_x, transform_y, rangelist, ground_z)

    lidar_conf = load_config(location, lidar_IP_list[0])
    area = get_area(lidar_conf)
    cell_grid = int(lidar_conf['cell_grid'])
    segment = get_segment(area.T, opt='auto', e=300//cell_grid)
    print('The clouds will be taken into {} process.'.format(segment[0].shape[0]))
    for i in range(len(lidar_IP_list)):
        Cloud_Raw_list.append(RawArray(i32, 1024*1024*10))  # the coloud points of lidar... int32
        lidar_conf = load_config(location, lidar_IP_list[i])
        p_recive = Process(target=lidar_receive_process, args=(Lock(), host_IP, lidar_conf, data_pickle, Cloud_Raw_list[i], 
                                                               Lidar_status_dict[lidar_conf['name']], range_ground_z, flag))
        p_recive.start()
    p_lidarshow = Process(target=lidarshow_process, args=(Clouds_Show_Raw, Lidar_Box_Raw, lidarshow, cloud_range, show_area))
    p_lidarshow.start()
    
    ini_life, full_life = (int(lidar_conf['ini_life']), int(lidar_conf['full_life'])) if 'ini_life' in lidar_conf.keys() \
                           and 'full_life' in lidar_conf.keys() else (-1, 2)
    next_ID = 1
    while flag.value:
        tick = time.time()
        clouds_update = np.empty((0,4), np.int32)  # 多个激光雷达的融合点云
        
        for i in range(len(lidar_IP_list)):            
            data = np.copy(np.ctypeslib.as_array(Cloud_Raw_list[i]))
            data_time1 = data[2]
            data_time2 = data[3]
            clouds_update = np.vstack((clouds_update, data[4:data[1]+4].reshape((-1,4))))  # X, Y, Z, I
        t1 = time.time()
        clouds_no_ground = clouds_range_no_ground(transform_x, transform_y, clouds_update, range_ground_z)
        clouds_downsample = points_downsample(np.int16(clouds_no_ground), cellsize=10)  # 以cellsize的尺度点云降采样
        
        clouds_grid = points_grid(clouds_downsample, cellsize=50)  # 以cellsize的尺度点云栅格化
        lidar_targets = get_cluster(clouds_grid)  # xc, yc, xmin, xmax, ymin, ymax, zmin, zmax
        # lidar_state:ID, class, xc, yc, vx, vy, xmin, xmax, ymin, ymax, zmin, zmax, I, life, heading
        lidar_state, next_ID = lidar_movement(lidar_targets, last_lidar_state, lidar_position, ini_life, full_life, next_ID, R_c2w, P0_UTM) 
        lidar_state_upload_box = lidar_state[np.where(lidar_state[:,13]>0)[0],0:14]  # 给出ID
        lidar_state_upload = lidar_state[np.where(lidar_state[:,13]>=0)[0],0:]  # 给出ID
        print(time.time()-tick)
        lidar_state_upload_1 = lidar_state[np.where(lidar_state[:,13]>=0)[0],0:]  # 给出ID
        last_lidar_state = lidar_state  # 更新上一帧
        lidar_id_area = lidar_state_upload_1[:, 0:5]
        lidar_id_area[:,4] = lidar_state_upload_1[:,13]
        # lidar_id_area[:,2:4] = lidar_state_upload_1[:, 4:6]
        # lidar_id_area[:,4] = (lidar_state_upload_1[:, 7]-lidar_state_upload_1[:, 6])*(lidar_state_upload_1[:, 9]-lidar_state_upload_1[:, 8])
        # print("lidar_id_area:\n", lidar_id_area)
        # # 激光雷达目标级输出
        # lidar_UTM_state:ID, class, xc, yc, vx, vy, heading, x_length, y_width 
        lidar_UTM_state = lidar_state_upload[:,0:9]        
        lidar_UTM_state[:, 6] = lidar_state_upload[:, 14] - 3600
        lidar_UTM_state[:, 7] = lidar_state_upload[:, 7] - lidar_state_upload[:, 6]
        lidar_UTM_state[:, 8] = lidar_state_upload[:, 9] - lidar_state_upload[:, 8]             
        lidar_UTM_state[:, 2:4] = lidar2UTM(lidar_UTM_state[:, 2:4], R_c2w, P0_UTM)
        #print("lidar_UTM_state:\n", lidar_UTM_state)        
        packet_body = lidar_UTM_state.ravel()
        packet_head = [frame_num+1, packet_body.size, data_time1, data_time1]
        packet = np.insert(packet_body, 0, packet_head)
        # 需要显示的点云
        # packet_clouds_body = clouds_no_ground.ravel()
        packet_clouds_body = clouds_update.ravel()
        packet_clouds_head = [frame_num+1, packet_clouds_body.size]
        packet_clouds = np.insert(packet_clouds_body, 0, packet_clouds_head)
        # 需要传递的目标瞄框数据
        packet_box_body = lidar_state_upload_box.ravel()
        packet_box_head = [frame_num+1, packet_box_body.size]
        packet_box = np.insert(packet_box_body, 0, packet_box_head)                
        lock.acquire()  # 进程锁
        memoryview(Clouds_Show_Raw).cast('B').cast(i32)[0:packet_clouds.size] = packet_clouds
        memoryview(Lidar_Box_Raw).cast('B').cast(i32)[:packet_box.size] = packet_box
        # Lidar_Target_Raw: frame_num+1, packet_body.size, class, xc, yc, vx, vy
        memoryview(Lidar_Target_Raw).cast('B').cast(i32)[:packet.size] = packet
        lock.release()  # 释放锁
        frame_num = (frame_num+1) % 65536
        while time.time() - tick <= 0.099:
            time.sleep (0.1 - (time.time() - tick))
    p_lidarshow.terminate()

def get_area(lidar_conf):
    xmin = [int(x.strip()) for x in lidar_conf['xmin'].split(',')]
    xmax = [int(x.strip()) for x in lidar_conf['xmax'].split(',')]
    ymin = [int(x.strip()) for x in lidar_conf['ymin'].split(',')]
    ymax = [int(x.strip()) for x in lidar_conf['ymax'].split(',')]
    area = np.int32([xmin,xmax,ymin,ymax]).transpose()
    return area

def start_keyboard_listener():
    from pynput.keyboard import Key, Listener
    def on_press(key):
        global flag, save
        if key == Key.esc:
            flag.value = 0
        if key not in Key and key.char == 'l':
            lidarshow.value *= -1
    def on_release(key):
        if key == Key.esc:
            return False
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()



if __name__ == "__main__":
    from threading import Thread
    system = sys.platform
    flag = Value('i', 1) # 0 指停止运行, 1指正常运行
    mapshow = Value('i',1) # -1 指不显示底图，1指正常显示
    lidarshow = Value('i', 1) # -1 指不显示点云图，1指正常显示

    edg_conf = load_sys_config('edg')
    host_IP = edg_conf['host_IP']
    Lidar_status_dict = dict()
    print("host_IP:", host_IP)
    edg_conf, gpu_conf, mqtt_conf = load_sys_config()
        
    listen_keyboard_thread = Thread(target=start_keyboard_listener, args=())
    listen_keyboard_thread.start()

    try: location = sys.argv[1]
    # except: location = 'lqy1'
    except: location = 'xgxx'
    Target_Send_Raw = RawArray('d', 1024)  # np.float64... 目标级数据
    lock = Lock()
    frame = 0

    p_mapshow = Process(target=draw_process, args=(location, Target_Send_Raw, mapshow))
    p_mapshow.start()


    pickle_name = location.split('_')[0]
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data_pickle = pickle.load(pickle_file)    
    Lidar_Target_Raw = RawArray(i32, 640)  # ID, class, Xw(cm), Yw(cm), Vx, Vym, Heading：定义显示激光雷达目标位置内存   
    lidar_IP_list = load_config(location)['lidar']
    for i in range(len(lidar_IP_list)):
        Lidar_conf = load_config(location, lidar_IP_list[i])
        Lidar_status_dict[Lidar_conf['name']] = Value('i', 1)
    # print("lidar_IP_list:", lidar_IP_list)

    Lidar_conf_0 = load_config(location, lidar_IP_list[0])
    Lidar_IP = Lidar_conf_0['IP']
    try: P0_UTM = np.int32(data_pickle[Lidar_IP]['P0'][0])  # 一个方向上的原点
    except: P0_UTM = np.int32([0,0])
    try: R_c2w = data_pickle[Lidar_IP]['Calibration'][-1]['R_c2w']
    except: R_c2w = np.eye(2)
    
    Lidar_detect_process = Process(target=Lidar_detect, args=(Lock(), lidar_IP_list, host_IP, location, Lidar_Target_Raw, 
                                                              flag, lidarshow, Lidar_status_dict)) 
    Lidar_detect_process.start()

    while flag.value:
        tick = time.time()
        raw = np.copy(np.ctypeslib.as_array(Lidar_Target_Raw))
        lidar_fnum = raw[0]
        # lidar_output:ID, class, xc, yc, vx, vy, heading
        lidar_output = raw[2:raw[1]+2].reshape((-1,9))        
        lidar_output[:, 2:4] = lidar2UTM(lidar_output[:, 2:4], R_c2w, P0_UTM)
        target_send = np.float64(lidar_output)
        packet_body = target_send.ravel()
        packet_head = [frame+1, packet_body.size]
        packet = np.insert(packet_body, 0, packet_head)
        lock.acquire()
        memoryview(Target_Send_Raw).cast('B').cast('d')[:packet.size] = packet
        lock.release()
        frame = (frame+1)%65536

        while time.time() - tick <= 0.099:
            continue



