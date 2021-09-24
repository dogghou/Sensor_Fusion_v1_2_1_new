# -*- coding: utf-8 -*-
"""
Upgrade on 2020.11.23
V1.1.0
editor: chenlei
"""
import os
import sys
import pickle
import time
import numpy as np
from multiprocessing import Process, Value, Lock, RawArray
from threading import Thread
from pynput.keyboard import Key, Listener

from config_operate import load_config, load_sys_config
from radar_process import Radar_detect
from camera_process import camera_detect
from core.core_process import DataFusion, same_type_fusion
try:
    from core.lidar_process import Lidar_detcet
except:
    pass
# from fusion_process import combine_camera_output, combine_radar_output
from funs import draw_process, send_message_process
from ubuntu_GPS import gps_time_check


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

    def on_release(key):
        if key == Key.esc:
            return False

    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def multi_fusion(location, flag, show, edg_conf, gpu_conf, IP_dict, target_raw, lock):
    # edg_conf, gpu_conf, mqtt_conf = load_sys_config()
    global mapshow, radarshow, imgshow, lidarshow
    mapshow, radarshow, imgshow, lidarshow = show['map'], show['radar'], show['img'], show['lidar']
    camera_model = gpu_conf['model_name']
    host_IP = edg_conf['host_IP']
    # direction = location.split('_')[-1]
    # direction = direction if str.isdigit(direction) else "1"
    # Radar_status_dict = dict()
    # Camera_status_dict = dict()
    # Lidar_status_dict = dict()
    np.set_printoptions(precision=6, suppress=True)
    np.set_printoptions(linewidth=400)
    # print(IP_dict)
    # 监听键盘
    listen_keyboard_thread = Thread(target=start_keyboard_listener, args=(flag,))
    listen_keyboard_thread.start()
    # pickle_file = open('data.pkl', 'rb')
    # data_pickle = pickle.load(pickle_file)
    # 获取L0_UTM
    for pickle_name in os.listdir("location_H"):
        if location.split('_')[0] in pickle_name: 
            break
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data_pickle = pickle.load(pickle_file)
    L0_UTM = data_pickle['L0'] if 'L0' in data_pickle else list(data_pickle.values())[0]['L0'] # 一个方向上的原点,
    i32 = 'l' if 'win' in sys.platform else 'i'

    data_fusion = DataFusion() # 建立融合感知类
    
    # 开启雷达进程
    Radar_output_list = list()
    Radar_fnum_list = list()  # store radar frame number
    radar_Ip_list = IP_dict['radar']
    for i in range(len(radar_Ip_list)):
        Radar_fnum_list.append(0)
        Radar_IP = radar_Ip_list[i]
        Radar_conf = load_config(location, radar_Ip_list[i])
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
    # Lidar_output = RawArray('i', 800)  # ID, class, Xw(cm), Yw(cm), Vx, Vy
    # Lidar_fnum = 0  # store lidar frame number
    # # Lidar_output_list = list()
    # # Lidar_fnum_list = list()  # store lidar frame number
    # lidar_IP_list = IP_dict['lidar']
    # for i in range(len(lidar_IP_list)):
    #     Lidar_conf = load_config(location, lidar_IP_list[i])
    #     Lidar_status_dict[Lidar_conf['name']] = Value('i', 1)
    # if len(lidar_IP_list) > 0:
    #     Lidar_detect_process = Process(target=Lidar_detcet, args=(Lock(), lidar_IP_list, host_IP, location,
    #                                                               Lidar_output, flag, Lidar_status_dict))
    #     Lidar_detect_process.start()
    data_list = []
    time_check = Value('d', 0)
    Thread(target=gps_time_check, args=(time_check,)).start()
    t0 = int(time.time() * 1000)
    while flag.value:
        tick = time.time()
        # 激光雷达数据
        # raw = np.copy(np.ctypeslib.as_array(lidar_output))
        # if raw[0] == 0 or raw[0] == Lidar_fnum:
        #     continue
        # Lidar_fnum = raw[0]
        # lidar_output = raw[2:raw[1] + 2].reshape((-1, 5))
        # lidar_output[:, 1:3] = lidar_output[:, 1:3] - np.int32(100 * L0_UTM)

        if mapshow.value > 0 and 'p_mapshow' not in vars(): # 开启画图进程
            p_mapshow = Process(target=draw_process, args=(location, target_raw, mapshow))
            p_mapshow.start()
        if mapshow.value < 0 and 'p_mapshow' in vars():
            p_mapshow.terminate()
            del p_mapshow
        
        # 取出所有雷达的当前帧的检测结果存放入 radar_targets_state 中
        radar_output = np.empty((0, 7), np.int32)  # radar_output: [[id, class, Xw, Yw, Vx, Vy, heading_init], ...]
        for i in range(len(Radar_output_list)):
            raw = np.ctypeslib.as_array(Radar_output_list[i])
            # if raw[0] == 0 or raw[0] == Radar_fnum_list[i]:
            #     continue
            Radar_fnum_list[i] = raw[0]
            radar_target_state = raw[2:raw[1] + 2].reshape((-1, 7))
            radar_output = np.vstack((radar_output, radar_target_state))
        radar_output[:, 2:4] = radar_output[:, 2:4] - np.int32(L0_UTM * 100)  # 减去路口方向原点（cm）...便于计算
        radar_output = same_type_fusion(radar_output)  # 融合被多个雷达都测出来的目标
        # 取出所有摄像头的当前帧的检测结果
        camera_output = np.empty((0, 7), np.int32)  # camera_output: [[id, class, X, Y, Vx, Vy, heading_init], ...]
        for i in range(len(Camera_output_list)):
            raw = np.ctypeslib.as_array(Camera_output_list[i])  # raw: fnum, output_size, t0_front, t0_back, output...
            # if raw[0] == 0 or raw[0] == Camera_fnum_list[i]:
            #     continue
            Camera_fnum_list[i] = raw[0]
            # print(raw[0])
            camera_target_state = raw[4:raw[1] + 4].reshape((-1, 7))
            t0 = int(str(raw[2]) + str(raw[3]))
            camera_output = np.vstack((camera_output, camera_target_state))
        
        camera_output[:, 2:4] = camera_output[:, 2:4] - np.int32(L0_UTM * 100)  # 减去路口方向原点（cm）...便于计算
        camera_output = same_type_fusion(camera_output)
        # print(camera_output)
        # 以下进行camera 和 radar融合
        # radar_output = np.array([[-537.191, 2325.123, -143.386,  304.134,  151]])

        target_state = data_fusion(camera_output, radar_output)
        # print('target_state:\n', target_state)
        target_send = target_state.astype(np.float64)  # 该部分将被发送: ID,class,Xw,Yw,Vx,Vy,heading:0-3600
        target_send[:, 2:4] = target_send[:, 2:4] / 100 + L0_UTM
        # print(target_send[np.abs(target_send[:, 6] - 2968) > 300])
        t1 = int(time.time() * 1000)
        # print('fusion_send: \n{}'.format(target_send))
        # print('\n\n')
        t0 += 3600*8*1000 + int(time_check.value*1000)
        t1 += 3600*8*1000 + int(time_check.value*1000)
        # print(t0, t1)
        # 发数据到其他进程
        # print('send \n', target_send)
        #data_list.append(target_send)
        #with open ('data.pkl', 'wb') as f:
        #    pickle.dump(data_list, f)
        packet_head = [t0, t1, target_send.size, 0]
        packet_body = target_send.ravel()
        packet = np.insert(packet_body, 0, packet_head)
        # print(packet)
        lock.acquire()
        memoryview(target_raw).cast('B').cast('d')[0:target_send.size + 4] = packet
        lock.release()
        if time.time() - tick < 0.099:
            time.sleep(0.1-(time.time()-tick))
    if 'p_mapshow' in vars(): 
        p_mapshow.terminate()
    print("close process of multi_sensor_fusion")


if __name__ == '__main__':
    location = sys.argv[1] if len(sys.argv) > 1 else 'cdcs'
    IP_dict = load_config(location)
    print(IP_dict)
    # 初始化参数
    flag = Value('i', 1)  # 0 指停止运行, 1 指正常运行
    target_Raw = RawArray('d', 1024)  # np.float64
    event_sent_raw = RawArray('d', 1024)  ## 交通事件数据
    Volume_sent_raw = RawArray('d', 1024)  ## 其他交通事件数据
    lock = Lock()
    show = dict()
    show['map'] = Value('i', -1)  # -1 指不显示底图，1指正常显示
    show['radar'] = Value('i', -1)  # -1 指不显示雷达图，1指正常显示
    show['img'] = Value('i', -1)  # -1 指不显示视频，1指正常显示
    show['lidar'] = Value('i', -1)  # -1 指不显示激光点云，1指正常显示
    edg_conf, gpu_conf, mqtt_conf = load_sys_config()

    # 开启发消息
    send_message_process = Process(target=send_message_process, args=(location, target_Raw, edg_conf, mqtt_conf, flag))
    send_message_process.start()

    multi_fusion(location, flag, show, edg_conf, gpu_conf, IP_dict, target_Raw, lock)
