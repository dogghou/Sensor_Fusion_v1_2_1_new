# -*- coding: utf-8 -*-
"""
Upgrade on 2020.11.23
V1.1.0
editor: chenlei
"""
import os, sys, pickle, time, math
import numpy as np
from multiprocessing import Process, Value, Lock, RawArray
from threading import Thread
from pynput.keyboard import Key, Listener

from config_operate import load_config, load_sys_config
from radar_process import Radar_detect
from camera_process import camera_detect
from core.core_process import DataFusion, same_type_fusion
# try: from lidar_process import Lidar_detcet
# try: from test_xgxx import Lidar_detect
# except: pass
from lidar_process import Lidar_detect
# from fusion_process import combine_camera_output, combine_radar_output
from funs import draw_process, send_message_process
# from ubuntu_GPS import gps_time_check
import pickle

np.set_printoptions(precision=6, suppress=True)
np.set_printoptions(linewidth=400)
i32 = 'l' if 'win' in sys.platform else 'i'


def start_keyboard_listener(flag):
    def on_press(key):
        if key == Key.esc:
            flag.value = 0
        elif key not in Key and key.char == 'M':
            mapshow.value *= -1
        elif key not in Key and key.char == 'C':
            imgshow.value *= -1
        elif key not in Key and key.char == 'R':
            radarshow.value *= -1
        elif key not in Key and key.char == 'L':
            lidarshow.value *= -1
        elif key not in Key and key.char == 'T':  # map中显示不同类传感器目标
            testshow.value *= -1 if mapshow.value > 0 else 1
    def on_release(key):
        if key == Key.esc:
            return False
    Listener(on_press=on_press, on_release=on_release).start()


def multi_fusion(location, flag, show, edg_conf, gpu_conf, IP_dict, target_raw, lock):
    # edg_conf, gpu_conf, mqtt_conf = load_sys_config()
    global mapshow, radarshow, imgshow, lidarshow, testshow
    radarshow, imgshow, lidarshow = show['radar'], show['img'], show['lidar']
    mapshow, testshow = show['map'], show['test']
        
    # 监听键盘
    start_keyboard_listener(flag)
    # 获取L0_UTM
    for pickle_name in os.listdir("location_H"):
        if location.split('_')[0] in pickle_name: 
            break
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data_pickle = pickle.load(pickle_file)
    L0_UTM = data_pickle['L0'] if 'L0' in data_pickle else list(data_pickle.values())[0]['L0'] # 一个方向上的原点,
    print("L0_UTM:", L0_UTM)
    # 建立融合感知类    
    data_fusion = DataFusion() 
    
    # 开启雷达进程
    Radar_output_list = list()
    Radar_fnum_list = list()  # store radar frame number
    radar_IP_list = IP_dict['radar']
    for i in range(len(radar_IP_list)):
        Radar_fnum_list.append(0)
        Radar_IP = radar_IP_list[i]
        Radar_conf = load_config(location, Radar_IP)
        # Radar_status_dict[Radar_conf['name']] = Value('i', 1)
        Radar_output_list.append(RawArray(i32, 1024))
        Radar_detect_process = Process(target=Radar_detect, args=(Lock(), Radar_IP, location, Radar_conf, 
                                                                  Radar_output_list[i], flag, radarshow))
        Radar_detect_process.start()

    # 开启摄像头进程
    Camera_output_list = list()
    Camera_fnum_list = list()
    camera_IP_list = IP_dict['camera']
    for i in range(len(camera_IP_list)):
        Camera_fnum_list.append(0)
        Camera_IP = camera_IP_list[i]
        Camera_conf = load_config(location, camera_IP_list[i])
        # Camera_status_dict[Camera_conf['name']] = Value('i', 1)
        Camera_output_list.append(RawArray(i32, 1024))
        Camera_detect_process = Process(target=camera_detect, args=(Lock(), Camera_IP, location, Camera_conf,
                                                                    Camera_output_list[i], flag))
        Camera_detect_process.start()

    # 开启激光雷达进程
    Lidar_status_dict = dict()
    host_IP = edg_conf['host_IP']
    print("host_IP:", host_IP)
    lidar_IP_list = IP_dict['lidar']
    Lidar_output = RawArray(i32, 1024)# ID, class, Xw(cm), Yw(cm), Vx, Vy, init_heading 
    for i in range(len(lidar_IP_list)):
        Lidar_conf = load_config(location, lidar_IP_list[i])
        Lidar_status_dict[Lidar_conf['name']] = Value('i', 1)
    if len(lidar_IP_list) > 0:
        Lidar_detect_process = Process(target=Lidar_detect, args=(Lock(), lidar_IP_list, host_IP, location, Lidar_output, 
                                                              flag, lidarshow, Lidar_status_dict)) 
        Lidar_detect_process.start()
    
    # 开辟各级传感器目标级数据存储空间和数据列表
    radar_raw = RawArray('d', 1024) if len(radar_IP_list) else None
    camera_raw = RawArray('d', 1024) if len(camera_IP_list) else None
    lidar_raw = RawArray('d', 1024) if len(lidar_IP_list) else None
    raw_list = [x for x in [target_raw, radar_raw, camera_raw, lidar_raw] if x]  # 存储空间
    send_list = list()  # 各类传感器数据列表
        
    # 以下正式开始融合进程
    frame, Time_list = 1, list()  # 总序列的帧号和时间
    f = open('lidar_test_result', 'wb')
    # print("before:", flag.value)
    while flag.value:
        tick = time.time()
        if mapshow.value > 0 and 'p_mapshow' not in vars():  # 开启画图进程
            p_mapshow = Process(target=draw_process, args=(location, raw_list, mapshow, testshow))
            p_mapshow.start()            
        if mapshow.value < 0 and 'p_mapshow' in vars():
            p_mapshow.terminate()
            del p_mapshow
        
        # 取出所有雷达的当前帧的检测结果存放入 radar_targets_state 中
        radar_output = list()  # radar_output: [[id, class, Xw, Yw, Vx, Vy, heading_init], ...]
        for i in range(len(Radar_output_list)):
            raw = np.ctypeslib.as_array(Radar_output_list[i])
            Radar_fnum_list[i] = raw[0]
            Time_list.append(int(raw[2])*1000 + raw[3])  # ms
            radar_target_state = raw[4:raw[1] + 4]
            radar_output = radar_output.append(radar_target_state)        
        radar_output = np.concatenate(radar_output).reshape(-1,7) if len(radar_output) else np.empty((0,7), np.int32)
        radar_output[:, 2:4] = radar_output[:, 2:4] - np.int32(L0_UTM * 100)  # 减去路口方向原点（cm）...便于计算
        radar_output = same_type_fusion(radar_output)  # 融合被多个雷达都测出来的目标
        if radar_raw and testshow.value > 0:
            send_list.append(radar_output[:, 2:4]/100 + L0_UTM)
        
        # 取出所有摄像头的当前帧的检测结果
        camera_output = list()  # camera_output: [[id, class, X, Y, Vx, Vy, heading_init], ...]
        for i in range(len(Camera_output_list)):
            raw = np.ctypeslib.as_array(Camera_output_list[i])  # raw: fnum, output_size, t0_front, t0_back, output...
            Camera_fnum_list[i] = raw[0]
            Time_list.append(int(raw[2])*1000 + raw[3])  # ms
            camera_target_state = raw[4:raw[1] + 4]
            camera_output.append(camera_target_state)
        camera_output = np.concatenate(camera_output).reshape(-1,7) if len(camera_output) else np.empty((0,7), np.int32)
        # print(camera_output)

        camera_output[:, 2:4] = camera_output[:, 2:4] - np.int32(L0_UTM * 100)  # 减去路口方向原点（cm）...便于计算
        camera_output = same_type_fusion(camera_output)
        if camera_raw and testshow.value > 0:
            send_list.append(camera_output[:, 2:4]/100 + L0_UTM)
        
        # 取出激光雷达数据
        #lidar_output = np.zeros((0, 7), np.int32)
        if len(lidar_IP_list):
            raw = np.copy(np.ctypeslib.as_array(Lidar_output))
            Lidar_fnum = raw[0]
            Time_list.append(int(raw[2])*1000 + raw[3])  # ms
            Time_list.append(int(raw[2])*1000 + raw[3])  # ms
            lidar_output = raw[4:raw[1] + 4].reshape(-1,9)
        # print("lidar_output:",lidar_output)
        lidar_output[:, 2:4] = lidar_output[:, 2:4] - np.int32(L0_UTM*100)
        if lidar_raw and testshow.value > 0:
            send_list.append(lidar_output[:, 2:4]/100 + L0_UTM)
        # print("lidar_output.shape[0]:", lidar_output.shape[0])

        #for lidar test
        # print("lidar_output2:\n", lidar_output)
        # lidar_output = lidar_output.astype(np.float64)
        # t0 = min(Time_list) if len(Time_list) else 0
        # t1 = int(time.time() * 1000)
        # packet_head = [frame, lidar_output.size, t0, t1]
        # packet_body = lidar_output.ravel()
        # packet = np.insert(packet_body, 0, packet_head)
        # pickle.dump (lidar_output, f)

        target_state = data_fusion(camera_output, lidar_output)
        # print('target_state:\n', target_state)
        target_send = target_state.astype(np.float64)  # 该部分将被发送: ID,class,Xw,Yw,Vx,Vy,heading:0-3600
        target_send[:, 2:4] = target_send[:, 2:4]/100 + L0_UTM
        # print("target_send:", target_send)
        # print("target_send.shape[0]:",target_send.shape[0])
        # print(target_send[np.abs(target_send[:, 6] - 2968) > 300])
        # print('fusion_send: \n{}'.format(target_send))
        # print('\n\n')

        # 发数据到其他进程
        # print("target_send:\n", target_send)        
        t0 = min(Time_list) if len(Time_list) else 0
        t1 = int(time.time() * 1000)
        packet_head = [frame, target_send.size, t0, t1]
        packet_body = target_send.ravel()
        packet = np.insert(packet_body, 0, packet_head)
        # print(packet)
        
        
        lock.acquire()
        memoryview(target_raw).cast('B').cast('d')[0:target_send.size + 4] = packet
        for raw, send in zip(raw_list[1:1+len(send_list)], send_list):
            memoryview(raw).cast('B').cast('d')[:send.size+1] = np.insert(send.ravel(), 0, send.size)
        lock.release()
        
        del Time_list[:]
        del send_list[:]
        frame += 1 if frame < 65535 else -65534
        
        if time.time() - tick < 0.099:
            time.sleep(0.1-(time.time()-tick))
    # print("after:", flag.value)
    f.close()
    if 'p_mapshow' in vars(): 
        p_mapshow.terminate()
    print("close process of multi_sensor_fusion")


if __name__ == '__main__':
    location = sys.argv[1] if len(sys.argv) > 1 else 'xgxx'
    print("location:", location)
    IP_dict = load_config(location)
    print("IP_dict:", IP_dict)
    # 初始化参数
    flag = Value('i', 1)  # 0 指停止运行, 1 指正常运行
    target_Raw = RawArray('d', 1024)  # np.float64
    event_sent_raw = RawArray('d', 1024)  ## 交通事件数据
    Volume_sent_raw = RawArray('d', 1024)  ## 其他交通事件数据
    lock = Lock()
    show = dict()
    show['map'] = Value('i', 1)  # -1 指不显示底图，1指正常显示
    show['radar'] = Value('i', -1)  # -1 指不显示雷达图，1指正常显示
    show['img'] = Value('i', -1)  # -1 指不显示视频，1指正常显示
    show['lidar'] = Value('i', 1)  # -1 指不显示激光点云，1指正常显示
    show['test'] = Value('i', 1) # -1 指不在底图上显示各自类别的传感器坐标
    edg_conf, gpu_conf, mqtt_conf = load_sys_config()

    # 开启发消息
    #send_message_process = Process(target=send_message_process, args=(location, target_Raw, edg_conf, mqtt_conf, flag))
    #send_message_process.start()

    multi_fusion(location, flag, show, edg_conf, gpu_conf, IP_dict, target_Raw, lock)
