import json
import os, pickle, socket, sys, time
from multiprocessing import Process, Value, Lock, RawArray
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt
from pynput.keyboard import Key, Listener

from funs import draw_process, creat_areas
from fusion_process import combine_radar_output
from core.core_process import RadarMovement, DataFusion, same_type_fusion

i32 = "l" if 'win' in sys.platform else "i"


"""
def point_range(target, X1=-1000, X2=1000, Y1=0, Y2=8000, V1=-1, V2=5000, PV1=0, PV2=5000):  # V=-1时不对速度过滤
    target = np.array(
        [x for x in target if (X1 <= x[0] < X2 and Y1 <= x[1] < Y2 and V1 < abs(x[2]) <= V2 and PV1 <= x[3] < PV2)])
    target = np.reshape(target, (-1, 4)).astype(np.int32)
    return target

def point_range(target, radar_arr, V1=-1, V2=5000, PV1=0, PV2=5000):  # V=-1时不对速度过滤
    target = np.array([x for x in target if (cv2.pointPolygonTest(radar_arr, (x[0], x[1]), False) < 0
                                             and V1 < abs(x[2]) <= V2 and PV1 <= x[3] < PV2)])
    target = np.reshape(target, (-1,4)).astype(np.int32)
    return target
"""


def point_fusion(target_update, usage='lane'):  # 如果target_update中有几个目标点的位置和速度十分接近，则将之进行合并
    # target_update: ID, class, Xw, Yw, Vx, Vy, PV
    point_fusion_result, lr = np.zeros_like(target_update), 0  # 创建一个零矩阵
    target_update = target_update[np.argsort(target_update[:, 3])]  # 按target的（Y值）排序

    if 'lan' in usage:
        X_scale, Y_scale, V_scale = 250, 500, 100  # 对于车的间距
    else:
        X_scale, Y_scale, V_scale = 100, 100, 50  # 对于行人的间距
    # 对target_update的几个点进行融合
    while target_update.shape[0] > 0:
        if target_update[0, 0] >= 0:  # 雷达已经做过跟踪，分配ID，则不进行融合，直接返回结果
            return target_update
        vec = np.sum(np.abs(target_update[:, [2, 3, 5]] - target_update[0, [2, 3, 5]]) // [X_scale, Y_scale, V_scale],
                     axis=1)
        tmp = target_update[vec == 0]  # 和target_update[0]在阈值范围内的，认为是同一个目标
        target_update = target_update[vec > 0]  # target_update[0]阈值范围的
        point_fusion_result[lr] = tmp[0, 0], tmp[0, 1], np.mean(tmp[:, 2]), np.min(tmp[:, 3]), np.mean(tmp[:, 4]), \
                                  np.max(tmp[:, 5]), np.sum(tmp[:, 6])
        lr += 1
    return point_fusion_result[:lr]


def radarshow_process(Radar_Data_Raw, Radar_IP, radarshow):
    # radar_range_arr, V1, V2 = radar_range
    fig = plt.figure(num=str(Radar_IP), figsize=(5, 4))
    (plt.xlim(-1000, 1000), plt.ylim(0, 12000), plt.grid(ls='--'))
    dot_g = plt.plot(list(), list(), 'go')[0]
    dot_b = plt.plot(list(), list(), 'bo')[0]

    while radarshow.value:
        tick = time.time()
        data = np.copy(np.ctypeslib.as_array(Radar_Data_Raw))
        radar_data = data[2:data[1] + 2].reshape(-1, 7)  # ID, class, X, Y, Vx, Vy, PV
        radar_data = radar_data[np.logical_and(np.abs(radar_data[:, 2]) > -1, radar_data[:, 3] > 0)]
        # print('radar_ori_data: \n', radar_ori_data)
        targetM = radar_data[radar_data[:, 5] != 0]
        targetS = radar_data[radar_data[:, 5] == 0]
        dot_g.set_data(targetM[:, 2], targetM[:, 3])
        dot_b.set_data(targetS[:, 2], targetS[:, 3])
        plt.pause(0.001)
        # radar_range_data = point_range(radar_ori_data, radar_range_arr, V1)
        # print('radar_ori_data:\n', radar_ori_data)
        if time.time() - tick <= 0.049:
            time.sleep(0.05 - (time.time() - tick))


def data_analysis(x, tick, supplier):
    # 6月21日改:增加慧尔视雷达解析程序
    radar_target = np.empty((0, 7), np.int32)  # ID, class, X, Y, Vx, Vy, PV
    while len(x) > 8 and x[0:4] == [165, 165, 130, 0]:  # 判定是否符合研煌雷达标准
        length = x[4] + x[5] * 256 + 7  # 数据&周期 + 报头(4)+长度(2)+校验(1)
        packet = x[:length]
        if len(packet[8:-1]) % 8 == 0 and sum(packet[:-1]) % 256 == packet[-1]:  # 判断是否为正常帧
            # 将采集数据的目标信息部分转换为无符号十六位矩阵，一行8个数，对应原8个字节
            data = np.array(packet[8:-1], np.uint16).reshape(-1, 8)
            radar_target = -np.ones((data.shape[0], 7), np.int32)  # ID, class, X, Y, Vx, Vy, PV
            radar_target[:, 2] = (data[:, 0] + data[:, 1] * 256).astype('int16')
            radar_target[:, 3] = (data[:, 2] + data[:, 3] * 256)  # Y方向是无符号整形
            radar_target[:, 5] = (data[:, 4] + data[:, 5] * 256).astype('int16')
            radar_target[:, 6] = (data[:, 6] + data[:, 7] * 256)
            tick = time.time()  # 正常解析出一帧, ，更新tick
        del x[:length]
    
    while len(x) > 40 and x[0:4] == [202, 203, 204, 205]:  # 判定是否符合慧尔视雷达报头标准
        length = x[4] * 256 + x[5] + 8
        packet = x[:length]
        class_list = [0, 3, 5, 1, 3, 3, 5, 0]
        if packet[-4:] == [234, 235, 236, 237] and (len(packet) - 41) % 37 == 0:
            data = np.array(packet[35:-6]).reshape(-1, 37)
            radar_target = -np.ones((data.shape[0], 7), np.int32)  # ID, class, X, Y, Vx, Vy, PV
            radar_target[:, 0] = (data[:, 0] * 256 + data[:, 1])  # ID
            radar_target[:, 1] = [class_list[x] for x in data[:, 3]]  # class
            radar_target[:, 2] = (data[:, 10] * 256 + data[:, 11]).astype("int16") * (-1)  # X
            radar_target[:, 3] = (data[:, 8] * 256 + data[:, 9])  # Y
            radar_target[:, 4] = (data[:, 14] * 256 + data[:, 15]).astype("int16") * (-1)  # Vx
            radar_target[:, 5] = (data[:, 12] * 256 + data[:, 13]).astype("int16")
            radar_target[:, 6] = (data[:, 34] * 256 + data[:, 35])  # PV
            tick = time.time()
        del x[:length]
    return radar_target, tick


def read_radar_process(Radar_IP, Radar_Port, Radar_Data_Raw, supplier, flag, lock, online=True, radar_frame=0):
    # 6月17日改：增加可读取离线数据，将数据解析部分分离
    # 6月21日改：增加慧尔视雷达数据解析部分分离
    from config_operate import load_sys_config
    
    sock = socket.socket()
    # supplier = 'hes'  # 定义厂家：hes为慧尔视
    if not online: 
        db = open('./data/radar_data.pkl', 'rb')
        data = pickle.load(db)        
        db.close()    
    connect_state = False if online else True
    
    tick = time.time()
    while flag.value and time.time() - tick < 3:
        if not connect_state:
            try:
                if supplier in 'hes': # 作为服务器端
                    Host_IP = load_sys_config('edg')['host_IP']
                    sock.bind((Host_IP, Radar_Port))  # 该Radar_IP和Radar_Port为雷达数据发送目的地址和发送端口
                    sock.listen(3)
                    conn, addr = sock.accept()
                    print(addr)
                elif supplier in 'yh': # 作为客户端
                    # print((Radar_IP, Radar_Port))
                    sock.connect((Radar_IP, Radar_Port))
                connect_state = True
                sock.setblocking(0)  # 非阻塞模式
                tick = time.time()  # 重新计时
                print('Radar {} connected.'.format(Radar_IP))
            except:
                time.sleep(0.2)
                continue  # 若连接不上，则重新开始循环，再次连接
        else:  # 连接上雷达时        
            try:
                if online and supplier in 'hes':
                    x = list(conn.recv(1024))
                elif online and supplier in 'yh':
                    x = list(sock.recv(1024))  # 接收一个帧
                    sock.send(bytes(0))
                elif not online:
                    x = data.pop(0)
                    time.sleep(0.05)    # 设置休息50ms   
                if not len(x): continue  # 收到一个空列表，尝试下一次读取
                tick = time.time()
            except:
                time.sleep(0.001)
                continue
            
            radar_target, tick = data_analysis(x, tick, supplier)
            # 组包，发送至内存
            packet_head = [radar_frame + 1, radar_target.size]
            packet_body = radar_target.ravel()
            packet = np.insert(packet_body, 0, packet_head)
            lock.acquire()
            memoryview(Radar_Data_Raw).cast('B').cast(i32)[:radar_target.size + 2] = packet
            lock.release()
            radar_frame = (radar_frame + 1) % 256
    # conn.close()
    sock.close()
    del sock
    return  # 终止进程


def Radar_detect(lock, Radar_IP, location, Radar_conf, Radar_Target_Raw, flag, radarshow, frame_num=0):
    # np.set_printoptions(precision=6, suppress=True)
    # np.set_printoptions(linewidth=400)
    # 读取地理信息
    Online = True  # 是否在线读取数据
    for pickle_name in os.listdir("location_H"):
        if location.split('_')[0] in pickle_name:
            break
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data = pickle.load(pickle_file)[Radar_IP]
    init_heading = int(data['Heading'] * 10)  # 记录为10倍角
    # print('init_heading: ', init_heading // 10)
    P0_UTM, P0_radar = np.int32(data['P0'][0] * 100), np.int32(data['P0'][1])  # 雷达测得的第一个点当做P0，单位cm
    Homography = data['Calibration'][-1]['H']

    
    supplier = Radar_conf['supplier']
    # 读取 point range 信息
    areas = creat_areas(100 * np.array(eval(Radar_conf['area_radar']), np.int32) if \
                            ('area_radar' in Radar_conf and Radar_conf['area_radar']) else np.empty(0))  # x1,y1,x2,y2,k,b... x=k*y+b
    
    V1 = int(Radar_conf['vmin']) if 'vmin' in Radar_conf else -1
    ini_life = int(Radar_conf['ini_life']) if 'ini_life' in Radar_conf else -1
    full_life = int(Radar_conf['full_life']) if 'full_life' in Radar_conf else 2
    usage = Radar_conf['usage']
    Radar_Port = int(Radar_conf['port']) if 'port' in Radar_conf and Radar_conf['port'] else 8080

    radar_direction = Radar_conf['direction'] if 'lane' in usage and 'direction' in Radar_conf and \
                                                 Radar_conf['port'] else 'head'
    radar_movement = RadarMovement(P0_UTM, P0_radar, Homography, init_heading, areas, ini_life, full_life, usage)

    IP_num = eval(Radar_IP.split('.')[-1]) * 1000
    # radar_range = (radar_range_arr, V1, V2)

    Radar_Data_Raw = RawArray(i32, 4096)  # 定义np.int32雷达解析数据内存空间

    last_radar_frame = 0
    # last_radar_state = np.empty((0, 8), dtype=np.int32)  # ID, class, Xr, Yr, Vx, Vy, PV, life
    while flag.value:
        # 自动开启读取雷达数据进程
        if 'p_read' not in vars():
            p_read = Process(target=read_radar_process, args=(Radar_IP, Radar_Port, Radar_Data_Raw, supplier, flag,
                                                              Lock(), Online))
            p_read.start()  # 开启接收毫米波雷达数据进程
        elif 'p_read' in vars() and (not p_read.is_alive()):
            del p_read

        if radarshow.value > 0 and 'p_radarshow' not in vars():  # 开启画图进程
            p_radarshow = Process(target=radarshow_process, args=(Radar_Data_Raw, Radar_IP, radarshow))
            p_radarshow.start()  # 开启显示雷达数据进程
        elif radarshow.value < 0 and 'p_radarshow' in vars():
            p_radarshow.terminate()
            del p_radarshow

        data = np.copy(np.ctypeslib.as_array(Radar_Data_Raw))  # 从内存中读取数据
        if data[0] == last_radar_frame:
            time.sleep(0.001)
            continue
        last_radar_frame = data[0]
        radar_ori_data = data[2:data[1] + 2].reshape(-1, 7)  # ID, class, Xr, Yr, Vx, Vy, PV
        # print(radar_ori_data)
        radar_data = radar_ori_data[np.logical_and(np.abs(radar_ori_data[:, 1]) > 0, radar_ori_data[:, -1] > 0)]
        radar_data = point_fusion(radar_data, usage) if 'yh' in supplier else radar_data
        radar_data[:, 3] = radar_data[:, 3] + 500 if 'tail' in radar_direction else radar_data[:, 3]

        # print('radar_data: \n', radar_data)
        radar_UTM_state = radar_movement(radar_data, supplier)  # 在里面已经把坐标转化啥的已经做了
        # print('radar_UTM_state:\n', radar_UTM_state)
        # ID(0), class(1), Xr(2), Yr(3), Vx(4), Vy(5), RCS(6), head(7), life(8)
        radar_UTM_state = radar_UTM_state[np.logical_and(radar_UTM_state[:, 1] > 1, radar_UTM_state[:, 8] > 0)]
        radar_UTM_state = radar_UTM_state[:, [0, 1, 2, 3, 4, 5, 7]]  # ID, class, Xr, Yr, Vx, Vy, head
        # print('radar_state:\n', radar_movement.last_radar_state)
        # print('radar_UTM_state:\n', radar_UTM_state)
        radar_UTM_state[:, 0] = radar_UTM_state[:, 0] + IP_num
        # print('\n\n')

        # radar_UTM_state = radar2UTM(radar_movement_state, radar_range_arr, IP_num, Homography, radar_direction, P0_radar, P0_UTM)  # class, Xw, Yw, Vx, Vy
        packet_head = np.int32([frame_num + 1, radar_UTM_state.size])  # frame_num+1 ~ range(1,1024)
        packet_body = radar_UTM_state.ravel()
        packet = np.insert(packet_body, 0, packet_head)

        # 将结果放入 Radar_raw_data
        lock.acquire()  # 进程锁
        memoryview(Radar_Target_Raw).cast('B').cast(i32)[:radar_UTM_state.size + 2] = packet
        lock.release()
        frame_num = (frame_num + 1) % 256

    if 'p_read' in vars():
        p_read.terminate()
    if 'p_radarshow' in vars():
        p_radarshow.terminate()


if __name__ == '__main__':
    from radar_process import *
    from config_operate import load_config


    def start_keyboard_listener():
        def on_press(key):
            global flag
            if key == Key.esc:
                flag.value = 0
            elif key not in Key and key.char == 'm':
                mapshow.value *= -1
            elif key not in Key and key.char == 'r':
                radarshow.value *= -1

        def on_release(key):
            if key == Key.esc:
                return False

        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


    flag = Value('i', 1)
    mapshow = Value('i', 1)  # -1 指不显示底图，1指正常显示
    radarshow = Value('i', -1)  # -1 指不显示雷达图，1指正常显示
    listen_keyboard_thread = Thread(target=start_keyboard_listener, args=())
    listen_keyboard_thread.start()

    location = sys.argv[1] if len(sys.argv) > 1 else 'cdcs'

    Target_Send_Raw = RawArray('d', 1024)  # np.float64
    lock = Lock()

    pickle_name = location.split('_')[0]
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data_pickle = pickle.load(pickle_file)
    L0_UTM = data_pickle['L0'] if 'L0' in data_pickle else list(data_pickle.values())[0]['L0']  # 一个方向上的原点

    Radar_output_list = list()
    Radar_fnum_list = list()  # store radar frame number
    radar_IP_list = load_config(location)['radar']
    print(radar_IP_list)
    # radar_IP_list = ['172.16.11.143']
    camera_output = np.empty((0, 7), np.int32)  # camera_output: [[ID, class, X, Y, Vx, Vy, heading], ...]

    for i in range(len(radar_IP_list)):
        Radar_output_list.append(RawArray('i', 1024))  # class, Xw(cm), Yw(cm), Vx, Vy
        Radar_fnum_list.append(0)
        Radar_IP = radar_IP_list[i]
        Radar_conf = load_config(location, radar_IP_list[i])
        Radar_detect_process = Process(target=Radar_detect, args=(Lock(), Radar_IP, location, Radar_conf,
                                                                  Radar_output_list[i], flag, radarshow))
        Radar_detect_process.start()

    data_fusion = DataFusion()

    # 读取雷达结果并显示
    while flag.value == 1:
        tick = time.time()

        if mapshow.value > 0 and 'p_mapshow' not in vars():  # 开启画图进程
            p_mapshow = Process(target=draw_process, args=(location, Target_Send_Raw, mapshow))
            p_mapshow.start()
        elif mapshow.value < 0 and 'p_mapshow' in vars():
            p_mapshow.terminate()
            del p_mapshow

        radar_output = np.empty((0, 7), np.int32)
        for i in range(len(Radar_output_list)):
            raw = np.copy(np.ctypeslib.as_array(Radar_output_list[i]))
            # if raw[0] == Radar_fnum_list[i]:
            #     continue
            Radar_fnum_list[i] = raw[0]
            # print(raw[0])
            radar_target_state = raw[2:raw[1] + 2].reshape((-1, 7)).astype(np.int32)
            radar_output = np.vstack((radar_output, radar_target_state))

        # print (time.time()-t1)
        radar_output[:, 2:4] = radar_output[:, 2:4] - np.int32(L0_UTM * 100)
        radar_output = same_type_fusion(radar_output)

        target_state = data_fusion(camera_output, radar_output)

        target_send = target_state.astype(np.float64)  # 该部分将被发送: ID,class,Xw,Yw,Vx,Vy,heading
        target_send[:, 2:4] = target_send[:, 2:4] / 100 + L0_UTM

        packet_head = [0, 0, target_send.size, 0]
        packet_body = target_send.ravel()
        packet = np.insert(packet_body, 0, packet_head)
        lock.acquire()
        memoryview(Target_Send_Raw).cast('B').cast('d')[0:target_send.size + 4] = packet
        lock.release()

        if time.time() - tick < 0.099:
            time.sleep(0.1-(time.time()-tick))

    if 'p_mapshow' in vars():
        p_mapshow.terminate()
