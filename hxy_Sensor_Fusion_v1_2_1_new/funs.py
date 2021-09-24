# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:37:43 2019

@author: Administrator
"""
import numpy as np
import paho.mqtt.client as mqtt
import json
import time, os, re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.text import Text
from math import radians, sin, cos
from threading import Thread
from pyproj import Proj

converter = Proj(proj='utm', zone=48, ellps='WGS84', south=False, north=True, errcheck=True)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

"""
def release_port(host):
    ret = os.popen("netstat -nao|findstr " + host[0] + ':' + str(host[1]))
    str_list = re.split('\n', ret.read())
    str_list.pop(-1)
    print(str_list)
    for i in range(len(str_list)):
        line = str_list[i].split()
        if line[-1] != 0:
            os.popen('taskkill.exe /pid ' + line[-1] + ' /F')
"""

def W842UTM(lat, lon):
    x, y = converter(lon, lat, inverse=False)  # x for East, y for North
    return x, y

def UTM2W84(P0, D=[0, 0]):
    x, y = P0 + D
    lon, lat = converter(x, y, inverse=True)
    return lat, lon

def creat_areas(area): # x1,y1,x2,y2,k,b... x=k*y+b
    # area 中的点必须是连续围城的
    if area.shape[0] <= 2:
        return np.empty(0, np.float64)
    if (area[-1]-area[0]).any():
        area = np.vstack((area, area[0]))
    areas = np.zeros((area.shape[0]-1,6), np.float64)
    areas[:,:2], areas[:,2:4] = area[:-1], area[1:]
    areas[:,4] = (areas[:,2]-areas[:,0])/(areas[:,3]-areas[:,1])
    areas[:,5] = areas[:,0] - areas[:,4]*areas[:,1]
    return areas

def draw_lane(location, ax=None): # 根据location画车道线，
    # 2021-6-21修改：舍弃了用try来尝试读取的方法，增加了输入参数ax
    # 2021-6-22：当经纬度数量太多时，采用间隔画点
    file_list, Lines, Lanes = list(), list(), list()
    for file in os.listdir('Map'):
        if location not in file: 
            continue
        file_list.append(file)
        filepath = 'Map/' + file
        with open(filepath, 'r') as fr:
            Lines.extend(fr.readlines())
        fr.close()
    jump = len(Lines)//80000 + 1
    for i in range(len(Lines)):
        if not (i%jump==0): continue
        line = re.split(', ', Lines[i].strip())
        if ':' in line[0] or len(line[0]) == 0: continue
        line[0] = float(line[0]) if line[0][0].isdigit() else float(line[0][1:])
        line[1] = float(line[1]) if line[1][-1].isdigit() else float(line[1][:-1])
        Lanes.append(line)
    Lanes = np.asarray(Lanes)
    plt.scatter(Lanes[:,0], Lanes[:,1], s=3) if not ax else ax.scatter(Lanes[:,0], Lanes[:,1], s=3)
    plt.pause(0.001)


def draw_process(location, args, mapshow, testshow): # 利用matplot投影场景地的图和目标
    # 2021-6-21修改：作为进程函数，mapshow为负时结束进程
    # 2021-7-1日改：基于testshow分别显示不同类传感器给出的位置
    args = [args] if not isinstance(args, list) else args
    Target_Send_Raw = args.pop(0)
    
    fig = plt.figure(num=str(location), figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1) 
    draw_lane(location, ax)
    Arrow = np.array([[0, 2.5], [-0.75, -0.5], [0, 0], [0.75, -0.5]])  # 箭头基本型
    Arrow[:,1] = Arrow[:,1]-2.5  # 把头设为原点
    id_max = 256
           
    color, dots = ['ys', 'cs', 'ms'], list()  # 三类传感器的三种颜色代码
    for i in range(len(args)):
        dots.append(ax.plot(list(), list(), color[i], markersize=3)[0])
    
    t, p, v = list(), list(), list()
    for i in range(id_max):
        t.append(Text(0, 0, text=str(i+1), visible=False))  # 初始不可见
        p.append(Circle([0, 0], radius=0.8, color='b', visible=False))
        v.append(Polygon([[0, 0]], color='g', visible=False))
        ax.add_table(t[i])
        ax.add_patch(p[i])
        ax.add_patch(v[i])
    
    while mapshow.value > 0:
        tick = time.time()
        data = np.copy(np.ctypeslib.as_array(Target_Send_Raw))
        target_draw = data[4:int(4 + data[1])].reshape(-1, 7)  # np.float64: ID,class,Xw,Yw,Vx,Vy,heading
        # print("target_draw:\n", target_draw)
        ID_list = (target_draw[:, 0].astype(int)).tolist()
        class_list = target_draw[:, 1].tolist()
        position_list = target_draw[:, 2:4].tolist()
        heading_list = (target_draw[:, 6] / 10).tolist()
        for i in range(id_max):
            person, vehicle, text = p[i], v[i], t[i]
            if i in range(len(ID_list)):
                ID = ID_list[i]
                xy, theta, cls = np.array(position_list[i]), radians(heading_list[i]), class_list[i]
                if cls <= 2:
                    vehicle.set_visible(False)
                    draw = xy
                    person.center = draw  # 更新位置
                    person.set_visible(True)  # 设为可见
                else:
                    person.set_visible(False)
                    cos_theta, sin_theta = (cos(theta), sin(theta))
                    R = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])  # 计算旋转弧度
                    draw = np.dot(Arrow, R.T) + xy
                    vehicle.set_visible(True)  # 设为可见
                    vehicle.set_xy(draw)  # 更新位置
                text.set_position(xy + [0, 1])
                text.set_visible(True)
                text.set_text(ID)
            else:
                text.set_visible(False)
                vehicle.set_visible(False)
                person.set_visible(False)
        # 画每类传感器的目标位置
        for dot,arg in zip(dots, args):
            _ = dot.set_visible(True) if testshow.value > 0 else dot.set_visible(False)
            data = np.copy(np.ctypeslib.as_array(arg))
            draw = data[1:int(1 + data[0])].reshape(-1, 2)                
            dot.set_data(draw[:,0], draw[:,1]) 
                
        plt.pause(0.001)
        if time.time()-tick <= 0.099:
            time.sleep(0.1-(time.time()-tick))


def get_original_target(Homography, R_w2c, Pw, Vw=np.zeros((2, 1))):
    Pp = np.vstack((Pw[0], Pw[1], 1))
    tmp = np.dot(np.linalg.inv(Homography), Pp)
    Pc = tmp[0:2] / tmp[-1]
    Vy = np.dot(R_w2c, Vw.reshape(2, 1))[1]
    return (Pc.T[0], Vy)


def init_client_mqtt(client_ID, mqtt_IP='127.0.0.1', mqtt_Port=1883):
    def on_connect(client, userdata, flags, rc):  # 当代理响应连接请求时调用
        print('mqtt_client connected')

    client = mqtt.Client(client_ID)  # client_ID 唯一识别
    # client.username_pw_set(client_ID, "public")
    client.on_connect = on_connect
    try:
        client.connect(mqtt_IP, mqtt_Port, 2)  # 感觉不需要那么长时间
        Thread(target=client.loop_forever).start()
        return client
    except:
        print('mqtt_client connect failed')


def send_fusion_data(client, target_send, computer_name, topic='/caeri_test/server/DAFU/cxdd/1'):
    def dataToJSON(target_send):
        width = [100, 100, 250, 300, 300]
        length = [100, 200, 400, 800, 800]
        height = [200, 200, 200, 400, 400]

        target_list = list()
        for target_data in target_send:
            lat, lon = UTM2W84(target_data[2:4])
            target = {
                "type": 3 if target_data[1] == 1 else 1,
                "id": int(target_data[0]),
                "source": 4,
                "lat": int(lat * 10000000),  # str(29.74656455)
                "lon": int(lon * 10000000),  # str(106.55214063)
                "speed": int(round(np.linalg.norm([target_data[4:6]]) / 100.0, 4) / 0.02),
                "heading": int(target_data[6] * 8),
                "width": width[int(target_data[1] - 1)],
                "length": length[int(target_data[1] - 1)],
                "height": int(height[int(target_data[1] - 1)] / 5)
            }
            target_list.append(target)
        data = {
            "participants": target_list,
            'device_ID': computer_name
        }

        param = json.dumps(data, sort_keys=False)
        return param
    if client.is_connected():
        payload = dataToJSON(target_send)
        # print(topic, '\n')
        # print(payload)
        client.publish(topic, payload=payload, qos=0)
    else:
        print('mqtt_client disconnected.')


def send_message_process(location, Target_Raw, edg_conf, mqtt_conf, flag):
    Project_ID, version, CNODE = edg_conf['Project_ID'], 'v1', edg_conf['CNODE']
    host_IP = edg_conf['host_IP']
    computer_type, computer_name = edg_conf['type'], edg_conf['name']
    send_mqtt_client_list = list()
    mqtt_IP_list = ['127.0.0.1']
    mqtt_Port_list = [1883]

    try:
        line = mqtt_conf['RSU_IP'].split(",")
        mqtt_IP_list += [x.strip() for x in line]
        mqtt_Port_list += [int(mqtt_conf['RSU_Port'])] * (len(mqtt_IP_list)-1)  # 防止本机端口和RSU端口不一致
    except:
        print("no RSU published")

    direction = location.split('_')[-1]
    direction = direction if str.isdigit(direction) else "1"
    topic = '/' + '/'.join([Project_ID, version, computer_name, 'DAFU', CNODE, direction])
    print(topic)

    # 开启给 RSU 和给后台发消息的 mqtt_client
    def connect_client():
        while mqtt_IP_list:
            for mqtt_IP, mqtt_Port in zip(mqtt_IP_list, mqtt_Port_list):
                client = init_client_mqtt(location, mqtt_IP, mqtt_Port)  # 连接不成功时返回None
                if client:
                    send_mqtt_client_list.append(client)
                    mqtt_IP_list.remove(mqtt_IP)
                    mqtt_Port_list.remove(mqtt_Port)
        print("all mqtt have connected")

    Thread(target=connect_client).start()  # 开启连接线程

    while flag.value:
        tick = time.time()
        data = np.copy(np.ctypeslib.as_array(Target_Raw))
        target_send = data[2:int(2 + data[0])].reshape(-1, 7)  # np.float64: ID,class,Xw,Yw,Vx,Vy,heading

        # 发送数据到mqtt
        for send_mqtt_client in send_mqtt_client_list:
            send_fusion_data(send_mqtt_client, target_send, computer_name, topic)

        if time.time() - tick < 0.099:
            time.sleep(0.1-(time.time()-tick))

    for send_mqtt_client in send_mqtt_client_list:
        send_mqtt_client.disconnect()

    print("close process of send_message")
