# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:36:39 2019

@author: Administrator
"""
import socket, os, time
import numpy as np
import pickle
try: import serial
except: pass

from pynput.keyboard import Key, Listener
from threading import Thread
from multiprocessing import Process, Lock, RawArray, Value

flag = Value('i',1)
radarshow = Value('i',1) # -1 指不显示底图，1指正常显示
radar_status = Value('i',1)
import matplotlib.pyplot as plt

# 雷达参数配置
Radar_IP = '172.16.11.143' # 雷达IP地址
Radar_Port = 8080 # 雷达端口号

# 创建文件存放地址
resultpath = 'data/'
if not os.path.exists(resultpath): # 判定该文件是否存在，若否则创建
    os.makedirs(resultpath)

def draw_process(Radar_Data_Raw, radarshow):
    while True:
        if radarshow.value < 0:
            plt.close()
            try: del fig
            except: pass 
            continue
        try: fig
        except: 
            fig = plt.figure(figsize=(9,6))
            plt.xlim(-1200,1200)
            plt.ylim(0,12000)
            plt.grid(ls='--')
            dot_g = plt.plot(list(), list(), 'go')[0]
            dot_b = plt.plot(list(), list(), 'bo')[0]
        data = np.copy(np.ctypeslib.as_array(Radar_Data_Raw))
        radar_target = data[2:2+data[1]].reshape(-1,4)
        targetM = radar_target[np.where(radar_target[:,2]!=0)]
        targetS = radar_target[np.where(radar_target[:,2]==0)]
        dot_g.set_data(targetM[:,0], targetM[:,1])
        dot_b.set_data(targetS[:,0], targetS[:,1])
        plt.pause(0.001)

def point_range(target, X1=-8000, X2=8000, Y1=0, Y2=8000, V1=0, V2=5000, PV1=-1, PV2=65535): # V=-1时不对速度过滤
    target = np.array([x for x in target if (X1<=x[0]<X2 and Y1<=x[1]<Y2 and V1<abs(x[2])<=V2 and PV1<x[3]<=PV2)])
    target = np.reshape(target, (len(target), 4)).astype(np.int32)
    return target

def read_radar_process(Radar_IP, Radar_Port, Radar_Data_Raw, radar_status, flag, lock, radar_frame=0):
    connect_state = False
    radar_data = list()
    while flag.value:
        try: sock  # 查看sock是否已经初始化
        except: sock = socket.socket()
        while flag.value and (connect_state is not True): # 若没有连接上
            try: 
                # print ((Radar_IP, Radar_Port))
                sock.connect((Radar_IP, Radar_Port))
            except: 
                time.sleep(0.2)
                continue  # 若连接不上，则重新开始循环，再次连接
            connect_state = True
            print ('connect_state: ', connect_state)
        radar_target = np.empty((0,4),np.int32) # 得到一个标准的空矩阵
        sock.setblocking(0)  # 非阻塞模式
        tick = time.time()
        while connect_state:   # 连接上雷达时
            try: 
                x = list(sock.recv(1024))  # 接收一个帧
                radar_data.append(x)
                # print ('x: ', x)
                if x[0] == 170:
                    x = [165, 165, 130, 0, 10, 0, 58, 153, 0, 0, 16, 39, 232, 3, 200, 0, 147]                
                radar_status.value = 1
                sock.send(bytes(0))
                tick = time.time()  # 更新tick
                break
            except: 
                if time.time()-tick < 2: 
                    continue # 2s内继续尝试读取
                sock.close()  # 超过2s断开连接，跳出循环
                del sock
                connect_state, x = False, list() # 默认x为一个空列表
                time.sleep(1)
                print ('connect_state: ', connect_state)
        try:
            if x[0:4] == [165, 165, 130, 0] and (sum(x[:-1])%256 == x[-1]):  # 判断是否为正常帧
                # 将采集数据的目标信息部分转换为无符号十六位矩阵，一行8个数，对应原8个字节
                data = np.array(x[8:-1], np.uint16).reshape(-1, 8)
                Xp = (data[:,0]+data[:,1]*256).astype('int16')
                Yp = (data[:,2]+data[:,3]*256)  # Y方向是无符号整形
                Vy = (data[:,4]+data[:,5]*256).astype('int16')
                PV = (data[:,6]+data[:,7]*256)
                radar_target = np.column_stack((Xp,Yp,Vy,PV)) # np.int32
        except: print("bad data")
        packet_head = [radar_frame+1, radar_target.size]
        packet_body = radar_target.ravel()

        packet = np.insert(packet_body, 0, packet_head)
        print(packet)

        print (packet.dtype)
        lock.acquire()
        packet = packet.astype('long')
        memoryview(Radar_Data_Raw).cast('B').cast('l')[:radar_target.size+2] = packet
        lock.release()
        radar_frame = (radar_frame+1)%256
        output = open('radar.pkl', 'wb')
        pickle.dump(radar_data, output)
        output.close()


if __name__ == '__main__':
    def start_keyboard_listener():
        def on_press(key):
            global flag
            if key == Key.esc:
                flag.value = 0
            elif key not in Key and key.char == 'r':
                radarshow.value *= -1

        def on_release(key):
            if key == Key.esc:
                return False
            
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listen_keyboard_thread = Thread(target=start_keyboard_listener, args=())
    listen_keyboard_thread.start()
    
    # 创建端口连接
    try:
        ser=serial.Serial()
        ser=serial.Serial("com5",115200,timeout=0.5) # 打开端口5
    except: pass
        
    Radar_Data_Raw = RawArray('l', 1024)
    p1 = Process(target=read_radar_process, args=(Radar_IP, Radar_Port, Radar_Data_Raw, radar_status, flag, Lock()))
    p2 = Process(target=draw_process, args=(Radar_Data_Raw, radarshow))    
    p1.start()  # 开启接收毫米波雷达数据进程
    p2.start()
    
    last_radar_frame = 0
    while flag.value:
        x = [165, 165, 130, 0, 18, 0, 58, 153]
        data = np.copy(np.ctypeslib.as_array(Radar_Data_Raw))  # 从内存中读取数据
        if data[0] == last_radar_frame:
            continue
        last_radar_frame = data[0]
        radar_target = data[2:data[1]+2].reshape(-1,4)       
        # radar_target = np.array([[-324,1500,0,150],[421,4356,1065,860],[1034,2679,788,315]], np.int16)
        print ('radar_target: \n', radar_target)
        target = point_range(radar_target, X1=-8000, X2=8000, Y1=0, Y2=80000, V1=-1) # 仅保留欲检测范围内的目标点
        
        radar_show = radar_target
        data_head = [radar_show.size%32768, 0]
        data_body = radar_show.ravel()
        """
        lock.acquire()
        memoryview(Radar_Target_Raw).cast('B').cast('h')[0:radar_show.size+2] = np.insert(data_body, 0, data_head)
        lock.release()
        """
        
        bytes_num = target.shape[0]*8+2  # 经处理后，雷达数据包含的字节数
        x[4]=bytes_num%256  # 对x的字节段数据进行更新
        x[5]=bytes_num//256

    # 以下将数据还原成原格式，以虚拟串口写入特定软件演示
        # print ('bytes_num: ', bytes_num)
        if bytes_num > 2: # 即测的目标数大于0
            target_display =  target # 只显示X Y Vy PV
            target_uint16 = target_display.astype('uint16') # 将原数值先转换为无符号十六位整形
            a = target_uint16%256
            b = target_uint16//256
            data_new = np.vstack((a[:,0],b[:,0],a[:,1],b[:,1],a[:,2],b[:,2],a[:,3],b[:,3])).T.ravel()
        else: # 否则添加一个（X,Y）=(0,0)的数据点
            x[4] = 10 # 一个目标，字节数为 1*8+2 = 10
            x[5] = 0
            data_new = [1,0,0,0,0,0,0,0]
        x_new = x[0:8]+list(data_new)
        x_new = x_new + [sum(x_new)%256]

        try: ser.write(bytes(x_new)) # 将最终数据通过特定串口发送并显示出来
        except: pass
        
    try: ser.close()
    except: pass
    try:
        sock.close()
        del sock
    except: pass
    p1.terminate()
    p2.terminate()
