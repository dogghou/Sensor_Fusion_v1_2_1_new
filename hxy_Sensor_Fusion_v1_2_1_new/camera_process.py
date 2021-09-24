import time, socket, pickle, cv2, sys, os
import numpy as np
from funs import draw_process, creat_areas
from multiprocessing import Process, Manager, Value, Lock, RawArray
from threading import Thread
from pynput.keyboard import Key, Listener
import json
from config_operate import load_sys_config

from core.core_process import CameraMovement, DataFusion, same_type_fusion

# from push_screen import camera_push_screen, push_rtmp

i32 = "l" if 'win' in sys.platform else "i"
# gpu_IP = load_sys_config('gpu')['gpu_IP']
labels = [None, 'person', 'bike', 'car', 'bus', 'truck']
Padding = False  # 缩放的时候是否通过填充使得图像的边为32的倍数, False时采用拉伸


def init_socket(gpu_IP, gpu_server_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # print('try connect:', (gpu_IP, gpu_server_port))
        sock.connect((gpu_IP, gpu_server_port))
        sock.settimeout(3)
        print(sock.recv(1024).decode())
        return sock
    except:
        # print('GPU server connection Refused!')
        return None


def camera_detect(lock, Camera_IP, location, Camera_conf, Camera_raw_data, flag, frame_num=0):
    # 6-22：融合离线和在线
    Online = True  # 是否在线读取数据
    gpu_server_port = 6161
    for pickle_name in os.listdir("location_H"):
        if location.split('_')[0] in pickle_name:
            break
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')

    Camera_conf['location'] = location
    data = pickle.load(pickle_file)[Camera_IP]
    init_heading = int(data['Heading'] * 10) if 'Heading' in data else 3600  # 记录为10倍角
    P0_UTM = np.int32(data['P0'] * 100)[0]  # cm
    Homography = data['Calibration'][-1]['H']
    (Kx, Ky) = (3, 3) if 'ind' in Camera_conf['usage'] else (1, 1)
    # (Kx, Ky) = (1, 1) if 'ind' in Camera_conf['usage'] else (3, 3)
    area_human = creat_areas(np.array(eval(Camera_conf['area_human']), np.int32) if \
                                 ('area_human' in Camera_conf and Camera_conf['area_human']) else np.empty(0))
    area_vehicle = creat_areas(np.array(eval(Camera_conf['area_vehicle']), np.int32) if \
                                   ('area_vehicle' in Camera_conf and Camera_conf['area_vehicle']) else np.empty(0))

    camera_delay = 0.05  # ？？？
    dt_base = 1 / 30 if 'ind' in Camera_conf['usage'] else 1 / 25
    ini_life = int(Camera_conf['ini_life']) if 'ini_life' in Camera_conf else 1
    full_life = int(Camera_conf['full_life']) if 'full_life' in Camera_conf else 5

    camera_movement = CameraMovement(P0_UTM, Homography, init_heading, area_human, area_vehicle, ini_life, full_life)

    IP_num = eval(Camera_IP.split('.')[-1]) * 1000
    data_conf = json.dumps(Camera_conf)  # 发送给gpu_server

    if not Online:
        db = open('./data/data3.pkl', 'rb')
        camera_detections = pickle.load(db)[2:]
        db.close()
    sock = None if Online else True # 离线状态下sock赋值为True

    while flag.value:
        if not sock:  # 连接不上继续返回空值
            sock = init_socket('127.0.0.1', gpu_server_port)
            if not sock:
                gpu_server_port += 1 if gpu_server_port < 6164 else -4
                continue
            sock.send(data_conf.encode())
        try:  # 断开连接... 接到
            if Online:
                data = sock.recv(1024)  # 接收GPU服务器识别的结果
                data = np.frombuffer(data)  # t0, len_frame, data_size, data...
                t0 = int(data[0])
                detections = data[3:int(data[2] + 3)].reshape(-1, 4).astype(np.int32)  # ID, class, Xp, Yp
                len_frame = int(data[1])
            elif len(camera_detections):  # 非在线模式
                detections, len_frame = camera_detections.pop(0)
                t0 = time.time()
                time.sleep(dt_base)
            else: break

        except Exception as e:  # 好像都不会出现了
            print(e, 'timed out/wrong data, reconnnect')
            continue

        detections[:, 2:4] = detections[:, 2:4] * [Kx, Ky]
        camera_data = np.empty((0, 4), np.int32) if (
                    detections == np.array([1000, 0, 0, 0])).all() else detections.astype(np.int32)
        # print(frame_num + 1)
        # print('camera_data:\n', (camera_data))          
        camera_UTM_state = camera_movement(camera_data, dt_base, len_frame)
        # print('camera_state:\n', (camera_movement.last_camera_state))
        # print('camera_position:\n', camera_movement.camera_position[:, 1:10, :])
        camera_UTM_state = camera_UTM_state[camera_UTM_state[:, 6] > 0, 0:7]  # ID, class, x, y, vx, vy, head
        # print(camera_UTM_state)
        camera_UTM_state[:, 0] += IP_num
        delay_select = np.logical_and((camera_UTM_state[:, 1] > 2),
                                      (np.linalg.norm(camera_UTM_state[:, 4:6], axis=1) > 800))
        camera_UTM_state[delay_select, 2:4] += (camera_UTM_state[delay_select, 4:6] * camera_delay).astype(np.int32)
        camera_UTM_state[:, 2:4] += P0_UTM
        # print ('camera_UTM_state: \n', camera_UTM_state)
        # print('\n\n')
        t0_front = int(str(t0)[0:8])
        try: t0_back = int(str(t0)[8:])
        except: t0_back = 0
        packet_head = np.int32([frame_num + 1, camera_UTM_state.size, t0_front, t0_back])  # frame_num+1 ~ range(1,1024)
        packet_data = camera_UTM_state.ravel()
        packet = np.insert(packet_data, 0, packet_head)
        # print(packet)
        # 将结果放入Camera_raw_data
        lock.acquire()  # 进程锁
        memoryview(Camera_raw_data).cast('B').cast(i32)[:camera_UTM_state.size + 4] = packet
        lock.release()
        frame_num = (frame_num + 1) % 1024

    if 'sock' in vars():
        try:
            sock.shutdown(2)
            sock.close()
        except:
            pass

    print('socket {} to gpu_server shutdown'.format(location))


if __name__ == '__main__':
    from config_operate import load_config

    def start_keyboard_listener():
        def on_press(key):
            if key == Key.esc:
                flag.value = 0
            elif key not in Key and key.char == 'm':
                mapshow.value *= -1
            elif key not in Key and key.char == 'c':
                imgshow.value *= -1

        def on_release(key):
            if key == Key.esc:
                return False

        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


    global flag
    flag = Value('i', 1)
    mapshow = Value('i', -1)  # -1 指不显示底图，1指正常显示
    imgshow = Value('i', -1)  # -1 指不显示image，1指正常显示
    listen_keyboard_thread = Thread(target=start_keyboard_listener, args=())
    listen_keyboard_thread.start()

    location = sys.argv[1] if len(sys.argv) > 1 else 'cdcs'
    Target_Send_Raw = RawArray('d', 1024)  # np.float64
    lock = Lock()
    # p_mapshow = Process(target=draw_process, args=(location, Target_Send_Raw, mapshow))
    # p_mapshow.start()

    gpu_conf = load_sys_config('gpu_server')
    camera_model = gpu_conf['model_name']

    # 获取L0_UTM
    for pickle_name in os.listdir("location_H"):
        if location.split('_')[0] in pickle_name:
            break
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data_pickle = pickle.load(pickle_file)
    L0_UTM = data_pickle['L0'] if 'L0' in data_pickle else list(data_pickle.values())[0]['L0']  # 一个方向上的原点,
    i32 = 'l' if 'win' in sys.platform else 'i'

    Camera_output_list = list()
    Camera_fnum_list = list()
    # camera_IP_list = load_config(location)['camera']
    camera_IP_list = ['172.16.11.146']
    camera_output = np.empty((0, 8), np.int32)  # camera_output: [[id, class, X, Y, Vx, Vy, init_head], ...]
    print(camera_IP_list)

    for i in range(len(camera_IP_list)):
        Camera_fnum_list.append(0)
        Camera_IP = camera_IP_list[i]
        Camera_conf = load_config(location, camera_IP_list[i])
        Camera_output_list.append(RawArray(i32, 1024))
        Camera_detect_process = Process(target=camera_detect, args=(Lock(), Camera_IP, location, Camera_conf,
                                                                    Camera_output_list[i], flag))
        Camera_detect_process.start()

    data_fusion = DataFusion()

    while flag.value == 1:
        tick = time.time()

        print(mapshow.value)
        if mapshow.value > 0 and 'p_mapshow' not in vars():  # 开启画图进程
            p_mapshow = Process(target=draw_process, args=(location, Target_Send_Raw, mapshow))
            p_mapshow.start()
        if mapshow.value < 0 and 'p_mapshow' in vars():
            p_mapshow.terminate()
            del p_mapshow

        camera_output = np.empty((0, 7), np.int32)  # camera_output: [[id, class, X, Y, Vx, Vy, heading], ...
        for i in range(len(Camera_output_list)):
            raw = np.copy(np.ctypeslib.as_array(Camera_output_list[i]))
            # Camera_fnum_list[i] = raw[0]
            camera_target_state = raw[2:raw[1] + 2].reshape((-1, 7)).astype(np.int32)
            camera_output = np.vstack((camera_output, camera_target_state))

        camera_output[:, 2:4] = camera_output[:, 2:4] - np.int32(L0_UTM * 100)  # 减去路口方向原点（cm）...便于计算
        # print('camera_output:\n', camera_output)
        camera_output = same_type_fusion(camera_output)
        radar_output = camera_output
        target_state = data_fusion(camera_output, radar_output)
        # print('target_state:\n', target_state)
        target_send = target_state.astype(np.float64)  # 该部分将被发送: ID,class,Xw,Yw,Vx,Vy
        target_send[:, 2:4] = target_send[:, 2:4] / 100 + L0_UTM
        # print('target_state:\n', target_state)
        # print('\n\n')

        packet_head = [target_send.size, 0]
        packet_body = target_send.ravel()
        packet = np.insert(packet_body, 0, packet_head)
        lock.acquire()
        memoryview(Target_Send_Raw).cast('B').cast('d')[0:target_send.size + 2] = packet
        lock.release()
        while time.time() - tick <= 0.0999:
            continue
    if 'p_mapshow' in vars():
        p_mapshow.terminate()
