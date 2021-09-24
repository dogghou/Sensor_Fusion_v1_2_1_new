import time, socket, pickle, cv2, sys, os
import numpy as np
from sdk import cksdk
from funs import draw_process
from multiprocessing import Process, Manager, Value, Lock, RawArray
from threading import Thread
from pynput.keyboard import Key, Listener
import json
from config_operate import load_sys_config

# from push_screen import camera_push_screen, push_rtmp

i32 = "l" if 'win' in sys.platform else "i"
gpu_IP = load_sys_config('gpu')['gpu_IP']
gpu_server_port = 6080
labels = [None, 'person', 'bike', 'car', 'bus', 'truck']
Padding = False  # 缩放的时候是否通过填充使得图像的边为32的倍数, False时采用拉伸
imgsz = 416  # 图像的长边... 一般选为416, 512, 640


def init_socket(gpu_IP, gpu_server_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print('try connect:', (gpu_IP, gpu_server_port))
        sock.connect((gpu_IP, gpu_server_port))
        print(sock.recv(1024).decode())
        return sock
    except:
        print('GPU server connection Refused!')


def camera2UTM(Homography, Xp, Yp):
    # 利用单应性变换将某平面坐标转为世界坐标
    Pp = np.array([[Xp], [Yp], [1]])  # 图像齐次坐标系
    tmp = np.dot(Homography, Pp)
    Pw = tmp[0:2] / tmp[-1]  # 当前帧目标的位置坐标，该坐标Y轴指向正北方，X轴指向东方
    return Pw.T[0]  # 输出单位是（cm）


def camera_detect(location, Camera_conf):
    save = False
    for file_name in os.listdir("location_H"):
        if location.split('_')[0] in file_name:
            pickle_name = file_name
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    # camera_IP = "10.130.29.128"
    data = pickle.load(pickle_file)
    data = data[camera_IP]
    Homography = data['calibration'][-1]['H']
    print(Homography)
    cali_shape = data['calishape'] if 'calishape' in data else [1920, 1080]
    Camera_conf['calishape'] = cali_shape

    data_conf = json.dumps(Camera_conf)
    tick_save = time.time()
    sock = None
    while flag.value:
        if not sock:  # 连接不上继续返回空值
            sock = init_socket(gpu_IP, gpu_server_port)
            sock.send(data_conf)

        if save and 8 <= time.localtime().tm_hour <= 18:
            if time.time() - tick_save > 300:
                i = len(os.listdir("/home/user/dataset")) + 1
                cv2.imwrite(f"/home/user/dataset/{str(i).zfill(6)}.jpg", ori_image)
                tick_save = time.time()

        try:  # 断开连接... 接到
            data = sock.recv(960)  # 接收GPU服务器识别的结果
            print(np.frombuffer(data))
            detections = np.frombuffer(data).reshape(-1, 6)
            print(detections)
        except:
            print('timed out/wrong data, reconnnect')
            sock.shutdown(2)
            sock.close()
            sock = None
            continue
    print('socket {} to gpu_server shutdown'.format(location))
    if 'sock' in vars():
        sock.shutdown(2)
        sock.close()


if __name__ == '__main__':
    from config_operate import load_config, load_sys_config


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
    camera_status = Value('i', 1)
    mapshow = Value('i', -1)  # -1 指不显示底图，1指正常显示
    imgshow = Value('i', -1)  # -1 指不显示image，1指正常显示
    listen_keyboard_thread = Thread(target=start_keyboard_listener, args=())
    listen_keyboard_thread.start()
    location = sys.argv[1] if len(sys.argv) > 1 else 'tuanjielu'
    Target_Send_Raw = RawArray('d', 1024)  # np.float64
    lock = Lock()
    p_mapshow = Process(target=draw_process, args=(location, Target_Send_Raw, mapshow))
    p_mapshow.start()

    gpu_conf = load_sys_config('gpu_server')
    camera_model = gpu_conf['model_name']

    pickle_name = location.split('_')[0]
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data_pickle = pickle.load(pickle_file)
    try:
        L0_UTM = data_pickle['L0']  # 一个方向上的原点
    except:
        L0_UTM = list(data_pickle.values())[0]['L0']
    direction = 1
    Camera_output_list = list()
    Camera_fnum_list = list()
    camera_IP_list = load_config(location)['camera']
    last_target_state = np.empty((0, 8), dtype=np.int32)  # ID, class, Xr, Yr, Vx, Vy, PV, life
    radar_output = np.empty((0, 5), np.int32)  # camera_output: [[class, X, Y, Vx, Vy], ...]
    for i in range(len(camera_IP_list)):
        Camera_output_list.append(RawArray('i', 642))
        Camera_fnum_list.append(0)
        camera_IP = camera_IP_list[i]
        Camera_conf = load_config(location, camera_IP_list[i])
        # Camera_detect_process = Process(target=camera_detect, args=(Lock(), camera_IP, location,
        #                                                             Camera_conf, Camera_output_list[i], camera_model,
        #                                                             flag, imgshow, camera_status))
        # Camera_detect_process.start()
        camera_detect(location, Camera_conf)

