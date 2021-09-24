#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:32:28 2020

@author: gac
"""

from multiprocessing import Process, RawArray, Lock, Value
import subprocess, pyautogui, time, os
# os.popen('sh /home/gac/nginx.sh')
import cv2
import numpy as np


def push_rtmp(filename, IP='127.0.0.1', shape=(960, 540)):
    rtmp = 'rtmp://' + IP + ':11935/live/' + filename
    print(rtmp)
    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', "{}x{}".format(shape[0], shape[1]),
               # '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'flv',
               rtmp]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def lidar_push_screen(flag=Value('i', 1), IP='192.168.48.111', shape=(960, 540)):
    def push_process(push, Rawdata, flag, shape):
        tick = time.time()
        while flag.value:
            img_data = np.ctypeslib.as_array(Rawdata)
            img = np.copy(img_data).reshape(shape[1], shape[0], 3)
            try:
                push.stdin.write(img.tobytes())
            except:
                print('pip broken')
            while time.time() - tick < 0.04:
                continue

    push = push_rtmp(filename='192_168_48_173', IP=IP, shape=shape)
    Rawdata = RawArray('B', shape[0] * shape[1] * 3)
    lock = Lock()
    # os.popen('xrandr --newmode "800x600_60.00" 38.25  800 832 912 1024  600 603 607 624 -hsync +vsync')
    # os.popen('xrandr --addmode Virtual1 800x600_60.00')
    os.popen('xrandr --output Virtual1 --mode 800x600')
    X, Y = 70, 70
    push_process = Process(target=push_process, args=(push, Rawdata, flag, shape))  # 读取雷达数据的进程
    push_process.start()
    while flag.value:
        # for i in range(10):
        t1 = time.time()
        img = pyautogui.screenshot(region=(X, Y, X + 640, Y + 420))
        # print(time.time() - t1)
        # img=pyautogui.screenshot()
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, shape)
        lock.acquire()
        memoryview(Rawdata).cast('B')[:] = img.ravel()
        lock.release()

    push_process.terminate()


def camera_push_screen(push_img, img_arr, shape=(960, 540)):
    while True:
        # t1 = time.time()
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = np.array(img_arr)
        img = img.reshape((540, 960, 3))
        # cv2.imshow("a", img)
        # cv2.waitKey(1)
        img = cv2.resize(img, shape)
        try:
            push_img.stdin.write(img.tobytes())
        except:
            pass
        # print(time.time() - t1)


if __name__ == '__main__':
    flag = Value('i', 1)
    lidar_push_screen(flag)
    os.popen('xrandr --output Virtual1 --mode 1680x1050')
    flag.Value = 0
