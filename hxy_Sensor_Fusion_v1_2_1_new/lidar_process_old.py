# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import time
import os, re
import socket
import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt
from Lidar_core import get_points

from funs import draw_lane, init_client_App, init_client_RSU, send_fusion_data, send_status
from config_operate import load_config
from multiprocessing import Pool, Process, Manager, Value, Lock, RawArray
from threading import Thread
from pynput.keyboard import Key, Listener
from Lidar_core import points_downsample, points_grid, points_range, target_movement
from fusion_process import data_fusion
# from push_screen import lidar_push_screen


# from Lidar_core import get_points
def ini_pcl(system):
    global vss, viewer
    if 'linux' in system:
        import pcl
        import pcl.pcl_visualization as viewer
        vss = viewer.PCLVisualizering('cloud')
        vss.AddCoordinateSystem(1)
        points_cloud = pcl.PointCloud()
    elif 'win' in system:
        from pclpy import pcl  # pclpy 可视化库
        viewer = pcl.visualization
        vss = viewer.PCLVisualizer('cloud')
        vss.addCoordinateSystem(1)
        points_cloud = pcl.PointCloud.PointXYZRGB()
    return points_cloud, vss, viewer


def draw_lidar_range(system, points_cloud_range, show_area=False):
    if show_area == False:
        return
    points_cloud_range = np.float32(points_cloud_range)
    points_cloud_range[:, :2] = points_cloud_range[:, :2] * 0.01
    if 'win' in system:
        from pclpy import pcl
        cloud_range = pcl.PointCloud.PointXYZI()
        cloud_range = cloud_range.from_array(points_cloud_range)
        vss.addPointCloud(cloud_range, 'range')
        vss.spinOnce(1)
    elif 'linux' in system:
        import pcl
        cloud_range = pcl.PointCloud()
        color = viewer.PointCloudColorHandleringCustom(cloud_range, 127, 127, 127)
        cloud_range.from_array(points_cloud_range[:, :3])
        vss.AddPointCloud_ColorHandler(cloud_range, color, 'range')
        vss.SpinOnce(1)


def draw_points_cloud(system, points_cloud, points_show, lidar_state_upload):
    global vss, viewer
    if 'win' in system:
        Inten = points_show[:, -1]
        Inten = 200 / np.mean(Inten) * Inten
        Inten[np.where(Inten > 255)] = 255
        color = np.expand_dims([0, 255, 0], 0).repeat(len(points_show), 0)
        color[:, 0] = Inten
        points_cloud = points_cloud.from_array(points_show[:, :3] * 0.01, color)

        vss.removePointCloud('cloud')
        vss.removeAllShapes()
        vss.addPointCloud(points_cloud, 'cloud')
        for i in range(lidar_state_upload.shape[0]):
            target = lidar_state_upload[i] / 100.
            # xmin,xmax,ymin,ymax,zmin,zmax = target[2],target[3],target[4],target[5],target[6],target[7]
            xmin, xmax, ymin, ymax, zmin, zmax = target[5], target[6], target[7], target[8], target[9], target[10]
            vss.addCube(xmin, xmax, ymin, ymax, zmin, zmax, 1, 0, 0, str(i))  # 单位m
            vss.setShapeRenderingProperties(viewer.PCL_VISUALIZER_REPRESENTATION, 1, str(i))
        vss.spinOnce(1)
    elif 'linux' in system:
        color = viewer.PointCloudColorHandleringCustom(points_cloud, 200, 255, 0)
        points_cloud.from_array((points_show[:, :3] * 0.01).astype(np.float32))
        vss.RemovePointCloud(b'cloud', 0)
        vss.RemoveAllShapes(0)
        vss.AddPointCloud_ColorHandler(points_cloud, color, 'cloud')
        for i in range(lidar_state_upload.shape[0]):
            target = lidar_state_upload[i] / 100.
            # xmin,xmax,ymin,ymax,zmin,zmax = target[2],target[3],target[4],target[5],target[6],target[7]
            xmin, xmax, ymin, ymax, zmin, zmax = target[5], target[6], target[7], target[8], target[9], target[10]
            vss.AddCube(xmin, xmax, ymin, ymax, zmin, zmax, 1, 0, 0, str(i))  # 单位m
        vss.SpinOnce(1)


def Lidar_parameter(Lidar_type):
    parameters = dict()
    parameters['lowbit'] = [x for x in range(4, 100) if x % 3 == 1]  # 低位索引
    parameters['highbit'] = [x for x in range(4, 100) if x % 3 == 2]  # 高位索引
    parameters['attenbit'] = [x for x in range(4, 100) if x % 3 == 0]  # 反射位索引

    if 'C16' in Lidar_type:
        parameters['V_theta'] = {0: -15, 1: 1, 2: -13, 3: 3, 4: -11, 5: 5, 6: -9, 7: 7, 8: -7,
                                 9: 9, 10: -5, 11: 11, 12: -3, 13: 13, 14: -1, 15: 15}  # 镭神16线激光雷达的垂直角度分布
        parameters['A_modify'] = np.zeros([32], np.float32)
    elif 'C32C' in Lidar_type:
        parameters['V_theta'] = {0: -18, 1: -15, 2: -12, 3: -10, 4: -8, 5: -7, 6: -6, 7: -5, 8: -4,
                                 9: -3.33, 10: -2.66, 11: -3, 12: -2.33, 13: -2, 14: -1.33, 15: -1.66,
                                 16: -1, 17: -0.66, 18: 0, 19: -0.33, 20: 0.33, 21: 0.66, 22: 1.33, 23: 1,
                                 24: 1.66, 25: 2, 26: 3, 27: 4, 28: 6, 29: 8, 30: 11, 31: 14}
        """
        A1, A2, A3, A4 = [1, 159], [3, 207], [2, 68], [3, 46]
        A1 = 0.01*(A1[0]*256+A1[1])
        A2 = 0.01*(A2[0]*256+A2[1])
        A3 = 0.01*(A3[0]*256+A3[1])
        A4 = 0.01*(A4[0]*256+A4[1])
        parameters['A_modify'] = np.array([A2,A1,A2,A1,A2,A1,A2,A1,
                                         A2,A1,A4,A3,A2,A1,A4,A3,
                                         A2,A1,A4,A3,A2,A1,A4,A3,
                                         A2,A1,A2,A1,A2,A1,A2,A1], np.float32)
        """
        A1, A2 = [0, 120], [1, 34]
        A1 = 0.01 * (A1[0] * 256 + A1[1])
        A2 = 0.01 * (A2[0] * 256 + A2[1])
        parameters['A_modify'] = np.array(
            [+A2, -A2, +A2, -A2, +A2, -A2, +A2, -A2, +A2, -A2, +A1, -A1, +A2, -A2, +A1, -A1,
             +A2, -A2, +A1, -A1, +A2, -A2, +A1, -A1, +A2, -A2, +A2, -A2, +A2, -A2, +A2, -A2], np.float32)

    return parameters


def get_background_base(Distance, scale=100):
    D = np.arange(0, Distance, scale).reshape(-1, 1)
    Azimuth = np.arange(0, 360, 0.5).reshape(1, -1).astype(np.float32)
    x = np.dot(D, np.sin(np.deg2rad(Azimuth)))
    y = np.dot(D, np.cos(np.deg2rad(Azimuth)))
    A = -np.ones((x.size, 4), np.int16)
    A[:, 0] = x.ravel()
    A[:, 1] = y.ravel()
    A[:, 3] += 128
    Azimuth = np.repeat(Azimuth, D.shape[0], axis=0).ravel()
    return A, Azimuth


def Lidar_points(data_reshape, Azimuth, base_scale, A_modify, block_V_theta, highbit, lowbit, attenbit, qc=0.25):
    packet_num = len(data_reshape)

    Azimuth_diff = np.diff(Azimuth)  # (?, 11)
    if Azimuth_diff.any(axis=0).all():  # Azimuth_diff中没有全为0列
        Azimuth_diff = np.hstack((Azimuth_diff, Azimuth_diff[:, -1:]))
    else:
        Azimuth_diff = np.hstack((Azimuth_diff, Azimuth_diff[:, -2:-1]))  # 第12列等于倒数第二列
        idex = np.where(Azimuth_diff.any(axis=0) == False)[0]  #
        Azimuth_diff[idex] = Azimuth_diff[idex + 1]
    Azimuth_diff[np.where(Azimuth_diff < 0)] += 360
    # Azimuth_bias = np.expand_dims(block_Azimuth_bias,0).repeat(packet_num,axis=0)  # (?, 12, 32)
    Azimuth_bias = np.expand_dims(Azimuth_diff / 32, 2).repeat(32, axis=2) * base_scale  # (?, 12, 32)
    Azimuth_scale = Azimuth_bias + np.expand_dims(Azimuth, 2) + A_modify + 90  # (?, 12, 32)
    Azimuth_scale[np.where(Azimuth_scale > 360)[0]] -= 360

    H_alpha = np.deg2rad(Azimuth_scale)
    V_theta = np.expand_dims(block_V_theta, 0).repeat(packet_num, axis=0);  # (?, 12, 32)
    Intensity = data_reshape[:, :, attenbit].astype(np.float32)
    Distance = qc * ((data_reshape[:, :, highbit] * 256 + data_reshape[:, :, lowbit])).astype(
        np.float32);  # (?, 12, 32)
    points, Azimuth_scale = get_points(Distance, H_alpha, V_theta, Intensity, Azimuth_scale)
    return points, Azimuth_scale


def get_points(Distance, H_alpha, V_theta, Intensity, Azimuth_scale):
    Distance = Distance.ravel()
    select = np.where(Distance > 0)[0]
    points = np.zeros((4, select.shape[0]), np.int16)
    Distance = Distance[select]
    H_alpha = H_alpha.ravel()[select]
    V_theta = V_theta.ravel()[select]
    Azimuth_scale = Azimuth_scale.ravel()[select]
    points[0] = np.int16(Distance * np.cos(V_theta) * np.sin(H_alpha))
    points[1] = np.int16(Distance * np.cos(V_theta) * np.cos(H_alpha))
    points[2] = np.int16(Distance * np.sin(V_theta))
    points[3] = np.int16(Intensity.ravel()[select])
    return points.transpose(), Azimuth_scale


def get_cluster(points_update3):
    # points_update3: 'sx','sy','x','y','z','zmax','I'
    from Lidar_core import cluster_circle
    if points_update3.shape[0] > 0:
        grid_slice, idex = np.unique(points_update3[:, [0, 1, 5]], axis=0, return_index=True)
        points_slice = np.array(np.vsplit(points_update3, idex[1:]))  # 将points_update3按栅格切片
        idex = (np.arange(points_slice.shape[0])).astype(np.int32)  # 一个idex为一个种子的索引
        targets = cluster_circle(points_slice, grid_slice, idex, cellsize=100).astype(
            np.int32)  # xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
        # targets =  np.int32(target_fusion2(targets, dr=100)) # 将目标按dr距离进行融合 # xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
    else:
        targets = np.zeros((0, 9), np.int32)
    return targets


def is_overlap(x, bbox, threshold=0.75):  # xmin, xmax, ymin, ymax
    # 求bbox中每行相对于x的覆盖比例
    area = (x[1] - x[0]) * (x[3] - x[2])  # x的面积
    overlap = np.minimum(x[[1, 3]], bbox[:, [1, 3]]) - np.maximum(x[[0, 2]], bbox[:, [0, 2]])  # 有交叠的x，y必然是>0
    overlap_index = np.where(np.logical_and(overlap[:, 0] > 0, overlap[:, 1] > 0))[0]
    overlap_area = overlap[overlap_index, 0] * overlap[overlap_index, 1]
    overlap_ratio = overlap_area / area
    if (overlap_ratio.shape[0] > 0):
        max_idex = np.argmax(overlap_ratio)  # 最大值索引
        if (overlap_ratio[max_idex] > threshold) and overlap_area[max_idex] > 20000:
            return overlap_index[max_idex]  # 返回索引


def Lidar_class(state):
    # state: class, xc, yc, vx, vy, xmin(5), xmax(6), ymin(7), ymax(8), zmin, zmax, I
    area = (state[6] - state[5]) * (state[6] - state[5])
    if area <= 10000:
        return 1
    elif area <= 40000:
        return 2
    else:
        return 3


def target_fusion(target):  # 将target_group中的所有物体合成同一个
    # target_group：xc, yc, xmin, xmax, ymin, ymax, zmin, zmax, I
    if target.ndim <= 1:
        target_new = target
    else:
        target_new = np.zeros(target.shape[1], target.dtype)
        target_new[[2, 4]] = np.min(target[:, [2, 4]], axis=0)
        target_new[[3, 5]] = np.max(target[:, [3, 5]], axis=0)
        target_new[0] = np.mean(target_new[[2, 3]])
        target_new[1] = np.mean(target_new[[4, 5]])
        target_new[7] = np.max(target[:, 7], axis=0)
        target_new[8] = np.sum(target[:, 8], axis=0)
    target_zmax = target_new[7]
    target_width = max(target_new[3] - target_new[2], target_new[5] - target_new[4])
    if target_zmax / target_width >= 4.5:
        target_new = None
    return target_new


def lidar2UTM(points_org, P0_distance, R_c2w=np.eye(2)):
    points_rot = points_org
    P0_distance = P0_distance.reshape((1, 2))
    Pxy = points_org[:, 0:2].transpose()
    points_rot[:, 0:2] = np.int16(np.dot(R_c2w, Pxy)).transpose()
    if P0_distance.any():
        points_rot[:, 0:2] = points_rot[:, 0:2] - P0_distance
    return points_rot


def read_lidar(addr, Data, lock, flag):  # used for ÀØÉñ
    def is_nextcircle(a0, l_a0):
        Bool = False
        if l_a0 != None:  # ÈôÉÏÒ»Ê±¿Ìl_a0²»Îª¿Õ
            if 18000 <= l_a0 < 36000 and 0 <= a0 < 18000:
                Bool = True
        return Bool

    lidar_connect_state = False  # À×´ïÁ¬½Ó×´Ì¬ÎªFalse
    data_list = list()  # ´æ·ÅÒ»¸öÖÜÆÚµÄÀ×´ïdata
    last_Azimuth_C0 = None
    frame = 0
    Azimuth_idx = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])
    while flag.value:
        try:
            sock  # ²é¿´sockÊÇ·ñÒÑ¾­³õÊ¼»¯
        except:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # ´´½¨ socket ¶ÔÏó, ÉèÖÃ¶Ë¿Ú
        while lidar_connect_state is not True:
            try:
                sock.bind(addr)  # °ó¶¨¶Ë¿Ú
            except:
                continue  # ÈôÁ¬½Ó²»ÉÏ£¬ÔòÖØÐÂ¿ªÊ¼Ñ­»·£¬ÔÙ´ÎÁ¬½Ó
            lidar_connect_state = True  # ÈôÁ¬½ÓÉÏÁË£¬Ôò×´Ì¬ÖÃÎªTrue
            print(str(addr) + ' is connected')
        try:
            packet_data, _ = sock.recvfrom(1248)
        except:
            continue
        packet_len = len(packet_data)  # Ò»¸öÊý¾Ý°üµÄ³¤¶È
        if packet_len < 1206:
            print('miss one packet')
            continue
        data = np.frombuffer(packet_data, np.uint8)  # °´×Ö½Ú, 8Î»Ò»¸ö×Ö½Ú
        # data_tail = data[-6:] # ¸½¼ÓÐÅÏ¢
        data_body = data[-1206:-6]  # ÏûÏ¢±¾Ìå

        Azimuth = data_body[Azimuth_idx + 3] * 256 + data_body[Azimuth_idx + 2]  # Ã¿¸öblockµÚÒ»¸öchannel0µÄ½Ç¶È
        A_index = np.where(Azimuth < Azimuth[0])
        Azimuth[A_index] = Azimuth[A_index] + 36000
        # print (np.std(Azimuth))
        if np.std(Azimuth) > 512: continue  # Èç¹ûAzimuth³öÏÖÁËÃ÷ÏÔµÄ²¨¶¯£¬Ò»°ãÈÏÎªÊÇ»µÖ¡
        Azimuth_C0 = Azimuth[0]
        # print (Azimuth_C0/100)
        is_next = is_nextcircle(Azimuth_C0, last_Azimuth_C0)  # ÅÐ¶¨µ±Ç°packetÊÇ·ñÒÑÎªÏÂÒ»¸öÑ­»·
        last_Azimuth_C0 = Azimuth_C0
        if is_next:  # Èç¹û±»ÅÐ¶¨ÎªÏÂÒ»¸öÑ­»·
            # print (len(data_list))
            # print ('\n')
            data_packets = np.array(data_list).reshape(-1)
            data_bytes = len(data_packets)
            L1 = [data_bytes % 65536, data_bytes // 65536]
            L2 = [L1[0] % 256, L1[0] // 256]
            data_length = [frame, L1[1], L2[0], L2[1]]  # ½«Êý¾Ý³¤¶ÈºÍµ±Ç°Ö¡ÊýÐ´½ødata_length
            lock.acquire()  # ½ø³ÌËø
            memoryview(Data).cast('B')[:data_bytes + 4] = np.insert(data_packets, 0, data_length)
            lock.release()  # ÊÍ·ÅËø  
            frame = (frame + 1) % 256
            del data_list[:]  # Çå¿Õµ±Ç°Ö¡µÄÊý¾ÝÁÐ±í
            # print(time.time()-t1)
        data_list.append(data_body)
    sock.close()
    del sock


def lidar_receive(lock, host_IP, lidar_conf, Raw_cloud_points, Lidar_status_dict, flag):
    def V_theta_creat(Di, b):
        V_theta_row = [Di[x] for x in Di];
        if len(V_theta_row) == 16:  # ÈôÊÇ16ÏßÀ×´ï£¬´¹Ö±½Ç¶È¼ÆÁ½´Î
            V_theta_row = V_theta_row * 2
        V_theta_row = np.array(V_theta_row, np.float32)
        V_theta_deg = np.expand_dims(V_theta_row, 0).repeat(b, axis=0)
        return np.deg2rad(V_theta_deg)

    push_clouds = True  # 
    block_num = 12  # 1¸öpacketÀïÃæÓÐ12¸öblock
    Lidar_type = lidar_conf['usage']
    Lidar_para = Lidar_parameter(Lidar_type)
    base_scale = np.arange(32).astype(np.float32)  # Ã¿ÐÐÎª1~32
    block_V_theta = V_theta_creat(Lidar_para['V_theta'], block_num)  # ¹¹½¨12x32µÄ´¹Ö±»¡¶È½Ç£¨rad£©
    try:
        qc = float(lidar_conf['acc_coefficient'])
    except:
        qc = 1  # À×´ïµÄ¾«¶ÈÏµÊý

    # À×´ïµÄµãÔÆÉ¸Ñ¡·¶Î§
    amin = [int(x.strip()) for x in lidar_conf['amin'].split(',')]
    amax = [int(x.strip()) for x in lidar_conf['amax'].split(',')]
    dmin = [int(x.strip()) for x in lidar_conf['dmin'].split(',')]
    dmax = [int(x.strip()) for x in lidar_conf['dmax'].split(',')]
    height = [int(x.strip()) for x in lidar_conf['height'].split(',')]
    area = np.int16([amin, amax, dmin, dmax, height]).transpose()
    mean_height = np.int16(np.mean(area[:, 4]))

    port = int(lidar_conf['port'])
    addr = (host_IP, port)
    RawData = RawArray('B', 1024 * 12 * 100)  # ÉèÖÃÒ»¸öÄÚ´æ¿Õ¼ä
    # memoryview(RawData).cast('B')[0] = 0
    last_frame = 0
    p1 = Process(target=read_lidar, args=(addr, RawData, Lock(), flag))  # ¶ÁÈ¡À×´ïÊý¾ÝµÄ½ø³Ì
    # p2 = Process(target=lidar_push_screen, args=(push_clouds, flag, '192.168.48.111'))
    p1.start()
    # p2.start()
    while flag.value:
        Raw_data = np.copy(np.ctypeslib.as_array(RawData))  # ÓÃctypeslib´ÓÄÚ´æÖÐ¶ÁÈ¡Êý¾Ý
        if Raw_data[0] == last_frame:  # Èç¹ûµ±Ç°Ö¡ÊýÎªÉÏÒ»Ö¡...±íÊ¾¸ÃÖ¡ÒÑ¾­¶Á¹ý
            continue
        last_frame = Raw_data[0]  # ÖØÐÂ¸³Öµlast_frame
        data_bytes = Raw_data[1] * 65536 + Raw_data[3] * 256 + Raw_data[2]
        data_reshape = Raw_data[4:data_bytes + 4].reshape((-1, 12, 100))
        Azimuth = ((data_reshape[:, :, 3] * 256 + data_reshape[:, :, 2]).astype(np.float32)) / 100.;  # (?, 12)

        points, Azimuth_scale = Lidar_points(data_reshape, Azimuth, base_scale, Lidar_para['A_modify'], block_V_theta,
                                             Lidar_para['highbit'], Lidar_para['lowbit'], Lidar_para['attenbit'], qc)
        points[:, 2] += mean_height
        points_update = points_range(points, Azimuth_scale, area)  # Éè¶¨À×´ïµãÔÆµÄ·¶Î§
        data_packets = points_update.ravel()
        # data_packets = np.int16(np.arange(1024))
        data_bytes = data_packets.size
        data_length = [data_bytes % 32768, data_bytes // 32768]
        lock.acquire()  # ½ø³ÌËø
        memoryview(Raw_cloud_points).cast('B').cast('h')[:data_bytes + 2] = np.insert(data_packets, 0, data_length)
        lock.release()  # ÊÍ·ÅËø
        Lidar_status_dict[lidar_conf['name']] = Value('i', 1)
    p1.terminate()
    # p2.terminate()


def Lidar_detect(lock, lidar_Ip_list, host_IP, location, Lidar_raw_data, flag, Lidar_status_dict, frame_num=0):
    # np.set_printoptions(precision=6, suppress=True)
    # np.set_printoptions(linewidth=400)
    import sys
    global vss, viewer
    system = sys.platform
    # flag = flag  0 stop, 1 start
    points_cloud, vss, viewer = ini_pcl(system)  # pcl initial

    pickle_name = location.split('_')[0]
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data = pickle.load(pickle_file)
    Lidar_read_list = list()
    P0_list = list()  # ´æ´¢Ã¿¸ö¼¤¹âÀ×´ïµÄÔ­µãÐÅÏ¢
    R_c2w_list = list()  # ´æ´¢Ã¿¸ö¼¤¹âÀ×´ïÐý×ª¾ØÕóÐÅÏ¢
    Cloud_points_list = list()  # ´æ·Å¸÷½ø³ÌµãÔÆÊý¾ÝµÄµØ·½
    last_target_state = np.empty((0, 13),
                                 np.int32)  # class, xc, yc, vx, vy, xmin, xmax, ymin, ymax, zmin, zmax, I, life
    for i in range(len(lidar_Ip_list)):
        Lidar_IP = lidar_Ip_list[i]  # »ñÈ¡¼¤¹âÀ×´ïIP        
        P0_list.append(np.int32(data[Lidar_IP]['P0'][0] * 100))  # the UTM of the lidar_orig
        R_c2w_list.append(data[Lidar_IP]['calibration'][-1]['R_c2w'])  # the rotation matrix of lidar
        Raw_cloud_points = RawArray('h', 1024 * 12 * 100)
        Cloud_points_list.append(RawArray('h', 1024 * 12 * 100))  # the coloud points of lidar... int16
        lidar_conf = load_config(location, Lidar_IP)
        Lidar_read_list.append(Process(target=lidar_receive, args=(Lock(), host_IP, lidar_conf, Raw_cloud_points,
                                                                   Lidar_status_dict, flag)))
        Lidar_read_list[i].start()

    try:
        ini_life, full_life = int(lidar_conf['ini_life']), int(lidar_conf['full_life'])
    except:
        ini_life, full_life = -1, 2
    P0_UTM = P0_list[0]

    tick = time.time()
    while flag.value:
        if time.time() - tick < 0.01:
            continue
        else:
            tick = time.time()
        points_update = np.empty((0, 4), np.int16)
        for i in range(len(lidar_Ip_list)):
            P0_distance = P0_list[i] - P0_list[0]  # every lidar to the org-lidar(cm)
            R_c2w = R_c2w_list[i]
            raw_points = np.copy(np.ctypeslib.as_array(Raw_cloud_points))  # ÓÃctypeslib´ÓÄÚ´æÖÐ¶ÁÈ¡Êý¾Ý
            data_bytes = raw_points[1] * 32768 + raw_points[0]
            # print (data_bytes)
            points_org = raw_points[2:data_bytes + 2].reshape((-1, 4))
            # t1 = time.time()
            points_rot = lidar2UTM(points_org, P0_distance, R_c2w)
            # print (time.time()-t1)
            points_update = np.vstack((points_update, points_rot))
        # t1 = time.time()
        points_update2 = points_downsample(points_update,
                                           cellsize=10)  # ÒÔcellsizeµÄ³ß¶È¶ÔÀ×´ï½øÐÐ½µ²ÉÑù, x,y,z,I
        points_update3 = points_grid(points_update2, cellsize=50)  # ÒÔcellsizeµÄ³ß¶È¶ÔÀ×´ï½øÐÐÕ¤¸ñ»¯
        targets = get_cluster(points_update3)  # xc, yc, xmin, xmax, ymin, ymax, zmin, zmax

        lidar_movement_state = target_movement(targets, last_target_state, ini_life,
                                               full_life)  # class, xc, yc, vx, vy, xmin, xmax, ymin, ymax, zmin, zmax, I
        lidar_state_upload = lidar_movement_state[np.where(lidar_movement_state[:, 12] > 0)]
        last_target_state = lidar_movement_state  # ¸üÐÂÉÏÒ»Ö¡
        lidar_UTM_state = np.copy(lidar_state_upload[:, 0:5])  # class, x, y, vx, vy
        # print (lidar_UTM_state)
        lidar_UTM_state[:, 1:3] = lidar_UTM_state[:, 1:3] + P0_UTM
        # print (lidar_UTM_state)
        packet_head = np.int32([frame_num + 1, lidar_UTM_state.size])  # frame_num+1 ~ range(1,1024)
        packet_data = lidar_UTM_state.reshape(-1)
        packet = np.insert(packet_data, 0, packet_head)

        # ½«½á¹û·ÅÈë Radar_raw_data
        lock.acquire()  # ½ø³ÌËø
        memoryview(Lidar_raw_data).cast('B').cast('i')[:lidar_UTM_state.size + 2] = packet
        lock.release()
        frame_num = (frame_num + 1) % 1024

        points_show = points_update
        # print(points_update[:10])
        # print(lidar_state_upload)
        draw_points_cloud(system, points_cloud, points_show, lidar_state_upload)
    try:
        vss.close()
    except:
        vss.Close()


if __name__ == '__main__':
    from lidar_process import *
    from config_operate import load_config, load_sys_config
    import sys


    def start_keyboard_listener():
        def on_press(key):
            global flag
            try:
                if key.char == 'q': flag.value = 0
            except:
                if key == Key.esc: pass

        def on_release(key):
            try:
                if key.char == 'q': return False
            except:
                if key == Key.esc: pass

        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


    listen_keyboard_thread = Thread(target=start_keyboard_listener, args=())
    listen_keyboard_thread.start()

    edg_conf, gpu_conf, mqtt_conf = load_sys_config()
    mqtt_IP, RSU_IP_list, mqtt_Port = mqtt_conf['mqtt_IP'], mqtt_conf['RSU_IP'], int(mqtt_conf['mqtt_Port'])
    line = re.split(',', RSU_IP_list)
    RSU_IP_list = [x.strip() for x in line]
    host_IP = edg_conf['host_IP']
    computer_type, computer_name = edg_conf['type'], edg_conf['ID']
    Project_ID, version, CNODE = edg_conf['Project_ID'], edg_conf['version'], edg_conf['CNODE']
    topic = '/' + '/'.join([Project_ID, version, computer_name, 'DAFU', CNODE])

    try:
        location = sys.argv[1]
    except:
        location = 'gq02_0'
    plt.figure(figsize=(5, 5))
    draw_lane(location)
    dot = plt.plot(list(), list(), 'go')[0]
    plt.pause(0.001)

    Lidar_status_dict = dict()
    Radar_status_dict = dict()
    Camera_status_dict = dict()
    flag = Value('i', 1)
    # 开启给 RSU 和给后台发消息的两个 mqtt_client
    send_client_RSU_list = list()
    for RSU_IP in RSU_IP_list:
        line = os.popen('ping -c 1 ' + RSU_IP).readlines()[4]
        if '1 received, 0% packet loss' in line:
            print(RSU_IP + ' is online...')
            send_client_RSU_list.append(init_client_RSU(location, RSU_IP, mqtt_Port))
        else:
            print(RSU_IP + ' is offline...')
    send_client_App = init_client_App(location, mqtt_IP, mqtt_Port)
    # 开启发送状态进程 
    # send_status_thread = Thread(target=send_status, args=(send_client_App, Radar_status_dict, Camera_status_dict,
    #                                                       Lidar_status_dict, computer_name, computer_type, flag))
    # send_status_thread.start()

    # À×´ïÊÇÏÈµãÔÆÈÚºÏ£¬È»ºóÔÙ¾ÛÀà·ÖÎö
    Lidar_output = RawArray('i', 640)  # class, Xw(cm), Yw(cm), Vx, Vy
    Lidar_fnum = 0  # store lidar frame number
    lidar_Ip_list = load_config(location)['lidar']
    for i in range(len(lidar_Ip_list)):
        Lidar_conf = load_config(location, lidar_Ip_list[i])
        Lidar_status_dict[Lidar_conf['name']] = Value('i', 1)
    # Lidar_IP = lidar_Ip_list[i]
    # Lidar_conf = load_config(location, Lidar_IP)
    Lidar_detect_process = Process(target=Lidar_detect, args=(Lock(), lidar_Ip_list, host_IP, location,
                                                              Lidar_output, flag, Lidar_status_dict))
    Lidar_detect_process.start()

    pickle_name = location.split('_')[0]
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    try:
        data = pickle.load(pickle_file)[lidar_Ip_list[0]]
    except:
        pass
    L0_UTM = data['L0']

    # ¶ÁÈ¡À×´ï½á¹û²¢ÏÔÊ¾
    last_target_state = np.empty((0, 7), np.int32)  # ID, class, Xw, Yw, Vx, Vy, camera_identify
    tick = time.time()
    while flag.value == 1:
        if time.time() - tick <= 0.099:
            continue
        else:
            tick = time.time()

        raw = np.copy(np.ctypeslib.as_array(Lidar_output))
        if raw[0] == 0 or raw[0] == Lidar_fnum:
            continue
        Lidar_fnum = raw[0]
        lidar_output = raw[2:raw[1] + 2].reshape((-1, 5))

        lidar_output[:, 1:3] = lidar_output[:, 1:3] - np.int32(100 * L0_UTM)

        camera_output = np.empty((0, 5), np.int32)
        target_state = data_fusion(last_target_state, camera_output, radar_output=lidar_output)
        last_target_state = target_state

        target_send = np.float64(target_state[:, 0:6])  # .astype(np.float32)  # 该部分将被发送: ID,class,Xw,Yw,Vx,Vy
        target_send[:, 2:4] = target_send[:, 2:4] / 100 + L0_UTM
        # target_send[:,0] = target_send[:, 0] + (int(direction) - 1) * 100  # 不同方向来车ID有所不同
        try:
            send_fusion_data(send_client_App, target_send, computer_name, topic)
        except:
            pass
        for send_client_RSU in send_client_RSU_list:
            send_fusion_data(send_client_RSU, target_send, computer_name, topic)

        target_UTM = lidar_output[:, 1:3] / 100 + L0_UTM
        dot.set_data(target_UTM[:, 0], target_UTM[:, 1])
        plt.pause(0.001)

    try:
        send_client_App.disconnect()
    except:
        pass
    for send_client_RSU in send_client_RSU_list:
        send_client_RSU.disconnect()
    flag.value = 0
    plt.close()
    os.popen('xrandr --output Virtual1 --mode 1680x1050')
