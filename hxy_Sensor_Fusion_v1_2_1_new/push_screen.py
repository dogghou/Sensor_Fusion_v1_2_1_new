#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:32:28 2020

@author: gac
"""

from multiprocessing import Process, Value, RawArray
from threading import Thread
import subprocess, time, os
# os.popen('sh /home/gac/nginx.sh')
import cv2
import numpy as np
import json
import paho.mqtt.client as mqtt


class playVideoOnDemand:
    def __init__(self, image_raw, cam_IP, flag):
        self.pushVideo_dict = dict()

        def on_connect(client, userdata, flags, rc):
            print("Connected with result code " + str(rc))

        def on_message(client, userdata, msg):
            self.pushVideo_dict = dict(json.loads(msg.payload.decode()))
            # print(msg.topic + " " + str(msg.payload))

        self.flag_dict = {}
        self.openVideo = []
        self.cam_IP = cam_IP
        self.client = mqtt.Client()
        self.client.on_connect = on_connect
        self.client.on_message = on_message
        self.client.connect("127.0.0.1", 1883, 60)
        self.client.subscribe('/Caeri/Push_Stream', qos=0)
        self.client.loop_start()
        while flag.value == 1:
            # print(self.pushVideo_dict)
            try:
                for cameraIp, para in list(self.pushVideo_dict.items()):
                    print(para.get('set'))
                    if para.get('set') == 1:
                        if cameraIp in self.openVideo or cameraIp != self.cam_IP:
                            time.sleep(0.5)
                            break
                        self.flag_dict[cameraIp] = 1
                        self.openVideo.append(cameraIp)
                        pushType = para.get('pushType')
                        pushUri = para.get('pushUri')
                        try:
                            pushTimeMin = int(para.get('pushTimeMin'))
                        except:
                            pushTimeMin = 10
                        if cameraIp and pushUri:
                            Thread(target=lambda: self.push_stream(cameraIp, image_raw, pushType, pushUri,
                                                                   pushTimeMin)).start()
                    else:
                        self.flag_dict[cameraIp] = 0
                        del self.pushVideo_dict[cameraIp]
            except:
                pass
            time.sleep(0.5)

    def push_stream(self, cameraIp, push_img_raw, pushType, pushUri, pushTimeMin):
        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', "{}x{}".format(960, 540),
                   '-r', '25',
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-f', 'flv',
                   pushUri]
        # 管道配置
        p = subprocess.Popen(command, stdin=subprocess.PIPE)
        del self.pushVideo_dict[cameraIp]
        t0 = time.time()
        while self.flag_dict[cameraIp]:
            t = time.time()
            if t - t0 < int(pushTimeMin) * 60:
                data = np.copy(np.ctypeslib.as_array(push_img_raw))
                height = data[0] + data[1] * 256
                if height == 0:
                    continue
                width = data[2] + data[3] * 256
                image = data[4:height * width * 3 + 4].reshape(height, width, 3)
                try:
                    image = cv2.resize(image, (960, 540))
                    p.stdin.write(image.tobytes())
                except:
                    continue
            else:
                self.openVideo.remove(cameraIp)
                break
            if time.time() - t < 1/25:
                time.sleep(1/25 - (time.time() - t))
        self.openVideo.remove(cameraIp)


def push_rtmp(filename, IP='127.0.0.1', shape=(640, 360)):
    rtmp = 'rtmp://116.63.160.171:31935/live/' + filename
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


def camera_push_screen(file_name, img_arr, flag, img_size=(640, 360)):
    tick = time.time()
    while flag:
        t1 = time.time()
        if 'push_img' not in vars():
            push_img = push_rtmp(filename=file_name)
        if subprocess.Popen.poll(push_img) != None:
            push_img.kill()
            del push_img
            continue
        if time.time() - tick > 10:
            print('push status:', subprocess.Popen.poll(push_img))
            tick = time.time()
        data = np.copy(np.ctypeslib.as_array(img_arr))
        height = data[0] + data[1] * 256
        width = data[2] + data[3] * 256
        img = data[4:height * width * 3 + 4].reshape(height, width, 3)
        if height == width == 0:
            # print('image error')
            continue
        else:
            show_image = cv2.resize(img, img_size)
        try:
            # cv2.imshow('a', show_image)
            # cv2.waitKey(1)
            push_img.stdin.write(show_image.tobytes())
            # print('push stream success')
        except:
            print('Push stream failed!')
        # print(time.time() - t1)
        while time.time() - t1 <= 1 / 25:
            continue


if __name__ == '__main__':
    img = cv2.imread("D:/Desktop/000001.jpg")
    flag = Value('i', 1)
    Image_Raw = RawArray("B", 1920 * 1080 * 3 + 4)  # np.uint8
    i_height, i_width = img.shape[0], img.shape[1]
    i_packet_head = [i_height % 256, i_height // 256, i_width % 256, i_width // 256]
    i_packet_body = img.ravel()
    i_packet = np.insert(i_packet_body, 0, i_packet_head)
    memoryview(Image_Raw).cast('B')[:img.size + 4] = i_packet
    # a = Process(target=camera_push_screen, args=('172_16_11_130', Image_Raw, flag))
    # a.start()
    cam_IP = '10.130.210.30'
    b = Process(target=playVideoOnDemand, args=(Image_Raw, cam_IP, flag))
    b.start()
    while True:
        t1 = time.time()
        # print(aa.value)
        while time.time() - t1 <= 1 / 2:
            continue
