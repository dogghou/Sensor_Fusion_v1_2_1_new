# coding=utf-8
import cksdk
import cv2
import numpy as np
from ctypes import *
import time
import os
from multiprocessing import process, RawArray, Value
import pynput
from pynput.keyboard import Key, Listener
from threading import Thread


if __name__ == '__main__':
    def start_keyboard_listener():
        def on_press(key):
            global flag, save
            if key == Key.esc:
                flag = False
            if key == Key.f3:
                save = True

        def on_release(key):
            global save
            if key == Key.esc:
                return False
            if key == Key.f3:
                save = False
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


    listen_keyboard_thread = Thread(target=start_keyboard_listener, args=())
    listen_keyboard_thread.start()
    print('The Camera is for industrial monitoring')
    result = cksdk.CameraEnumerateDevice()
    if result[0] != 0:
        print("Don't find camera")

    print("Find cameras number: %d" % result[1])
    # 初始化相机
    camera_IP = "192.168.6.212"
    result = cksdk.CameraInitEx2(camera_IP)
    if result[0] != 0:
        print("open camera failed")

    hCamera = result[1]
    cksdk.CameraReadParameterFromFile(hCamera, 'UGSMT200C_Cfg_A.bin')  # 载入相机配置参数360p
    cksdk.CameraSetIspOutFormat(hCamera, cksdk.CAMERA_MEDIA_TYPE_BGR8)  # 设置相机输出格式
    cksdk.CameraSetTriggerMode(hCamera, 0)  # 设置为连续拍照模式
    cksdk.CameraSetAeState(hCamera, True)  # 设置为自动曝光模式
    # 开启相机
    cksdk.CameraPlay(hCamera)
    frame_list = []

    i = len(os.listdir("/home/user/dataset"))
    flag = True
    save = False
    while flag:
        result = cksdk.CameraGetImageBufferEx(hCamera, 1000)
        img_data = result[0]
        if img_data is not None:
            img_info = result[1]
            bytes_count = img_info.iWidth * img_info.iHeight * 3
            img_array = cast(img_data, POINTER(c_char * bytes_count))
            frame = np.frombuffer(img_array.contents, dtype=np.uint8, count=bytes_count)
            frame.shape = (img_info.iHeight, img_info.iWidth, 3)
            # frame = cv2.resize(frame, (1920, 1080))
            if save:
                cv2.imwrite(f"/home/user/dataset/{str(i).zfill(6)}.jpg", frame)
                i = i + 1
                print(i)
                # cv2.imwrite("/home/data/" + "")
            # frame_list.append(frame)
            cv2.imshow("q", frame)
            cv2.waitKey(1)
    # main()
    print("exit!!!")
