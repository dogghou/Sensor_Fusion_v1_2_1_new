#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:50:09 2021
@author: user
"""
import time, math, cv2
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Value, RawArray, Lock
import pickle

def traffic_event_detect_cdcs(flag, target_vehicle_raw, detect_area, lock, road_heading, VehicleStop, Highspeed, Lowspeed,
                         VehicleConverse, VehicleEmergency):
    vehicle_v_list = [500, 2000 / 3.6, 4000/ 3.6]
    # [np.zeros((5, 2), dtype=np.float64), 0, loucount=0,0, tick,0]
    traffic_event_dict = dict() #目标数据字典
    data_list=[]
    data_list1=[]
    while flag.value:
        t1 = time.time()
        data = np.copy(np.ctypeslib.as_array(target_vehicle_raw))
        target_vehicle = data[4:int(data[2] + 4)].reshape(-1, 7)
        target_vehicle = target_vehicle[target_vehicle[:, 1] >= 3]  ##提取车辆数据
        tt = data[0]/1000
        person_data = target_vehicle[target_vehicle[:, 1] < 3]
        for vehicle in target_vehicle:
            target_v = np.sqrt(np.square(vehicle[4]) + np.square(vehicle[5]))  ##计算车辆的速度
#            print('target_V:',target_v)
            for j in range(len(detect_area)):
                if Whether_in_area(vehicle[2:4], detect_area[j]):  ##判断车辆在哪个检测区域（车道）
                    if vehicle[0] not in traffic_event_dict.keys():  ##若目标级字典中无此目标的ID 则新建目标数据集
                        values = np.zeros((5, 2), dtype=np.float64)
                        values[0, :] = vehicle[2:4]
                        start_time = tt
                        end_time = tt
                        count = 0
                        line = j+1
                        v=target_v
                        traffic_event_dict[vehicle[0]] = [values, 0, 0, 0, start_time, end_time, count, line, 0, v]
                        ##[x,y],低速/违停, 高速，逆行，stime, etime,count,line,应急车道
                    else:
                        traffic_event_dict[vehicle[0]][6] += 1
                        # count += 1
                        if 0 < traffic_event_dict[vehicle[0]][6] < 5:  ##判断目标级中位置数组是否存储完毕
                            num = traffic_event_dict[vehicle[0]][6]
                            traffic_event_dict[vehicle[0]][0][num] = vehicle[2:4]
                            # values[num] = vehicle[2:4]
                        else:  ##若已存储完毕 则向上替代
                            traffic_event_dict[vehicle[0]][0][1:] = traffic_event_dict[vehicle[0]][0][:-1]
                            traffic_event_dict[vehicle[0]][0][-1] = vehicle[2:3]
                    if target_v > 0:  ##逆行  应急车道
                        traffic_event_dict[vehicle[0]][8] = min(traffic_event_dict[vehicle[0]][8] + 1, 5)
                        if not whether_vehicle_drive_direction_error(vehicle[6], road_heading):
                            traffic_event_dict[vehicle[0]][3] = min(traffic_event_dict[vehicle[0]][3] + 1, 5)
#                            traffic_event_dict[vehicle[0]][8] = min(traffic_event_dict[vehicle[0]][8] + 1, 5)

                        else:
                            traffic_event_dict[vehicle[0]][3] = max(traffic_event_dict[vehicle[0]][3] - 1, 0)
#                            traffic_event_dict[vehicle[0]][8] = min(traffic_event_dict[vehicle[0]][8] + 1, 5)

                    if target_v < vehicle_v_list[1]:  ##低速+违停
#                        print('lowspeed',target_v)
                        traffic_event_dict[vehicle[0]][1] = min(traffic_event_dict[vehicle[0]][1] + 1, 5)
                        traffic_event_dict[vehicle[0]][2] = max(traffic_event_dict[vehicle[0]][2] - 1, 0)
                        traffic_event_dict[vehicle[0]][5] = tt
                        traffic_event_dict[vehicle[0]][9] = target_v
                    elif target_v > vehicle_v_list[2]:  ##高速
#                        print('hispeed',target_v)
                        traffic_event_dict[vehicle[0]][1] = max(traffic_event_dict[vehicle[0]][1] - 1, 0)
                        traffic_event_dict[vehicle[0]][2] = min(traffic_event_dict[vehicle[0]][2] + 1, 5)
                        traffic_event_dict[vehicle[0]][5] = tt
                        traffic_event_dict[vehicle[0]][9] = target_v
                    if tt - traffic_event_dict[vehicle[0]][5] > 8:
                        del traffic_event_dict[vehicle[0]]
                        continue
                    # print(traffic_event_dict)

        VehicleStop_event = []
        highspeed_event = []
        lowspeed_event = []
        VehicleConverse_event = []
        VehicleEmergency_event = []
        id_list = traffic_event_dict.keys()
        for id in id_list:
            if traffic_event_dict[id][6] >= 3:  # 判断目标是否存储够一定数量的数据
                start_point = traffic_event_dict[id][0][0]
                end_point = traffic_event_dict[id][0][-1]
                if traffic_event_dict[id][1] >= 5:
#                    print('low:V:',traffic_event_dict[id][9])
                    if get_distance_from_point_to_point(start_point, end_point) < 5 and traffic_event_dict[id][7] == 4:
                        temp = [id, traffic_event_dict[id][0][-1][0], traffic_event_dict[id][0][-1][1],
                                traffic_event_dict[id][4], traffic_event_dict[id][5]]
                        VehicleStop_event.append(temp)
                    else:
                        temp = [id, traffic_event_dict[id][0][-1][0], traffic_event_dict[id][0][-1][1],
                                traffic_event_dict[id][4], traffic_event_dict[id][5], traffic_event_dict[id][7]]
                        lowspeed_event.append(temp)
                if traffic_event_dict[id][2] >= 1:
#                    print('high_V:',traffic_event_dict[id][9])
                    temp = [id, traffic_event_dict[id][0][-1][0], traffic_event_dict[id][0][-1][1],
                            traffic_event_dict[id][4], traffic_event_dict[id][5], traffic_event_dict[id][7]]
                    highspeed_event.append(temp)
                if traffic_event_dict[id][3] >= 5 and traffic_event_dict[id][7] == 4:
                    temp = [id, traffic_event_dict[id][0][-1][0], traffic_event_dict[id][0][-1][1],
                            traffic_event_dict[id][4], traffic_event_dict[id][5]]
                    VehicleConverse_event.append(temp)
                if traffic_event_dict[id][8] >= 1 and traffic_event_dict[id][7] == 4:
#                    print("***")
                    temp = [id, traffic_event_dict[id][0][-1][0], traffic_event_dict[id][0][-1][1],
                            traffic_event_dict[id][4], traffic_event_dict[id][5]]
                    VehicleEmergency_event.append(temp)
        print('VehicleStop_event', VehicleStop_event)
        data_list.append(VehicleStop_event)
        with open ('VehicleStop_eventdata.pkl', 'wb') as f:
            pickle.dump(data_list, f)
        dangerous_vehicle = np.array(VehicleStop_event)
        dangerous_vehicle = dangerous_vehicle.reshape(1, -1)
        dangerous_vehicle = dangerous_vehicle.astype(np.float64)
        VehicleStop_packet_head = [dangerous_vehicle.size, 0]
        VehicleStop_packet_body = dangerous_vehicle.ravel()
        VehicleStop_packet = np.insert(VehicleStop_packet_body, 0, VehicleStop_packet_head)
        VehicleStop_packet = VehicleStop_packet.astype(np.float64)
#        print('lowspeed_event', lowspeed_event)
        lowspeed = np.array(lowspeed_event)
        lowspeed = lowspeed.reshape(1, -1)
        lowspeed = lowspeed.astype(np.float64)
        lowspeed_head = [lowspeed.size, 0]
        lowspeed_body = lowspeed.ravel()
        lowspeed_packet = np.insert(lowspeed_body, 0, lowspeed_head)
        lowspeed_packet = lowspeed_packet.astype(np.float64)
#        print('highspeed_event', highspeed_event)
        highspeed = np.array(highspeed_event)
        highspeed = highspeed.reshape(1, -1)
        highspeed = highspeed.astype(np.float64)
        highspeed_head = [highspeed.size, 0]
        highspeed_body = highspeed.ravel()
        highspeed_packet = np.insert(highspeed_body, 0, highspeed_head)
        highspeed_packet = highspeed_packet.astype(np.float64)
        print('Vehicleconverse', VehicleConverse_event)
        data_list1.append(VehicleConverse_event)
        with open ('VehicleConverse_event.pkl', 'wb') as f:
            pickle.dump(data_list1, f)
        Vehicleconverse = np.array(VehicleConverse_event)
        Vehicleconverse = Vehicleconverse.reshape(1, -1)
        Vehicleconverse = Vehicleconverse.astype(np.float64)
        Vehicleconverse_head = [Vehicleconverse.size, 0]
        Vehicleconverse_body = Vehicleconverse.ravel()
        Vehicleconverse_packet = np.insert(Vehicleconverse_body, 0, Vehicleconverse_head)
        Vehicleconverse_packet = Vehicleconverse_packet.astype(np.float64)
#        print('Vehicle_emergency', VehicleEmergency_event)
        Vehicle_emergency = np.array(VehicleEmergency_event)
        Vehicle_emergency = Vehicle_emergency.reshape(1, -1)
        Vehicle_emergency = Vehicle_emergency.astype(np.float64)
        Vehicle_emergency_head = [Vehicle_emergency.size, 0]
        Vehicle_emergency_body = Vehicle_emergency.ravel()
        Vehicle_emergency_packet = np.insert(Vehicle_emergency_body, 0, Vehicle_emergency_head)
        Vehicle_emergency_packet = Vehicle_emergency_packet.astype(np.float64)

        lock.acquire()
        memoryview(VehicleStop).cast('B').cast('d')[0:dangerous_vehicle.size + 2] = VehicleStop_packet
#        memoryview(Lowspeed).cast('B').cast('d')[0:lowspeed.size + 2] = lowspeed_packet
#        memoryview(Highspeed).cast('B').cast('d')[0:highspeed.size + 2] = highspeed_packet
        memoryview(VehicleConverse).cast('B').cast('d')[0:Vehicleconverse.size + 2] = Vehicleconverse_packet
#        memoryview(VehicleEmergency).cast('B').cast('d')[0:Vehicle_emergency.size + 2] = Vehicle_emergency_packet
        lock.release()
#        while tt - t1 < 0.1:
#            print(tt-t1)
#            time.sleep(0.1 - (tt- t1))


def data_producation(temp):  # volumn detect data
    target_vehicle_data = []
    for i in range(temp):
        x = np.random.randint(3, 16, 1)
        y = np.random.randint(15, 20, 1)
        vx = np.random.randint(200, 800, 1)
        vy = np.random.randint(200, 800, 1)
        heading = np.random.randint(90, 270, 1)
        temp1 = np.vstack((np.array(i + 1), np.array(3), x, y, vx, vy, heading)).reshape(1, -1)
        target_vehicle_data.append(temp1[0, :])
    return target_vehicle_data


def multi_data_producation(target_vehicle_raw, lock, class_traffic):
    count = 0
    if class_traffic == 1:
        k = 1
        vehicle_num = np.random.randint(6, 15, 1)[0]
        kk = 20
    elif class_traffic == 2:
        k = 0
        vehicle_num = 2
        kk = 20
    elif class_traffic == 3:
        k = 0.2
        vehicle_num = np.random.randint(6, 15, 1)[0]
        kk = 2000
    elif class_traffic == 4:
        k = 1
        vehicle_num = np.random.randint(6, 15, 1)[0]
        kk = 20
    target_vehicle_data = np.array(data_producation(vehicle_num)).astype(np.float64)
    while True:
        t1 = time.time()
        if count == 0:
            if class_traffic != 2:
                target_vehicle_data = np.array(data_producation(vehicle_num)).astype(np.float64)
                if class_traffic == 4:
                    target_vehicle_data[-1, 3] = -1
                    target_vehicle_data[-1, 6] = 10
            count += 1
        else:

            if class_traffic == 4:
                target_vehicle_data[0:-1, 3] -= k
                target_vehicle_data[-1, 3] += k
            elif class_traffic == 1:
                target_vehicle_data[0:vehicle_num - 2, 3] -= k
                target_vehicle_data[vehicle_num - 2:vehicle_num, 3] -= 4
                target_vehicle_data[vehicle_num - 2:vehicle_num, 4] = 1100
                target_vehicle_data[vehicle_num - 2:vehicle_num, 5] = 1100
            else:
                target_vehicle_data[:, 3] -= k
            count += 1
        if count > kk:
            count = 0
        # elif count == 3 and class_traffic == 2:
        #     temp = target_vehicle_data[0, :]
        #     target_vehicle_data = np.delete(target_vehicle_data, 0, axis=0)
        # elif count == 15 and class_traffic == 2:
        #     target_vehicle_data = np.insert(target_vehicle_data, 0, values=temp, axis=0)
        target_send = target_vehicle_data.reshape(1, -1)
        packet_head = [target_send.size, 0]
        packet_body = target_send.ravel()
        packet = np.insert(packet_body, 0, packet_head)
        packet = packet.astype(np.float64)
        lock.acquire()
        memoryview(target_vehicle_raw).cast('B').cast('d')[0:target_send.size + 2] = packet
        lock.release()
        while time.time() - t1 < 1:
            continue


def draw_process(Target_Send_Raw, volumn_area, volumn_area2, traffic_area, detect_area, line_UTM1, line_UTM2):
    fig = plt.Figure()
    plot_data = plt.plot(list(), list(), 'r+')[0]
    draw_plt(traffic_area, 'g')
    draw_plt(detect_area, 'b')
    draw_plt(volumn_area, 'k')
    draw_plt(volumn_area2, 'c')
    plt.plot(line_UTM1, line_UTM2, color='r')
    while True:
        data = np.copy(np.ctypeslib.as_array(Target_Send_Raw))
        target_draw = data[2:int(data[0] + 2)].reshape(-1, 7)
        x = target_draw[:, 2]
        y = target_draw[:, 3]
        plot_data.set_data(list(x), list(y))
        plt.pause(0.001)


def draw_plt(area, color):
    plt.plot([area[0][0], area[1][0]], [area[0][1], area[1][1]], color=color)
    plt.plot([area[1][0], area[2][0]], [area[1][1], area[2][1]], color=color)
    plt.plot([area[2][0], area[3][0]], [area[2][1], area[3][1]], color=color)
    plt.plot([area[3][0], area[0][0]], [area[3][1], area[0][1]], color=color)


def dangerous_car_count(lock, VehicleStop, flag, Target_vehicle_raw, detect_area):  ##异常停车检测发送
    vehicle_v_temp = 200  # cm/s
    longtime_limit = 5
    vehicle_num_temp = 3
    target_vehicle_dangerous = np.zeros((0, 5), dtype=np.float64)
    t1 = time.time()
    while flag.value:
        data = np.copy(np.ctypeslib.as_array(Target_vehicle_raw))
        target_vehicle = data[4:int(data[2] + 4)].reshape(-1, 7)
        for i in range(target_vehicle.shape[0]):
            if target_vehicle[i, 1] < 3:
                np.delete(target_vehicle, i, 0)
        if target_vehicle.shape[0] > 0:
            # continue
            target_v = np.sqrt(np.square(target_vehicle[:, 4]) + np.square(target_vehicle[:, 5]))
            target_vehicle_error = target_vehicle[target_v < vehicle_v_temp]
            if 0 < target_vehicle_error.shape[0] < vehicle_num_temp:
                target_vehicle_dangerous = dangerous_car_detect(flag, detect_area, target_vehicle_dangerous,
                                                                target_vehicle_error)
            # else:
            #     continue
            target_vehicle_dangerous2 = target_vehicle_dangerous[
                target_vehicle_dangerous[:, 4] - target_vehicle_dangerous[:, 3] > longtime_limit]
            if target_vehicle_dangerous2.shape[0] >= 1:
                dangerous_vehicle_status = True
                dangerous_vehicle = target_vehicle_dangerous2
                dangerous_vehicle = dangerous_vehicle.reshape(1, -1)
                dangerous_vehicle = dangerous_vehicle.astype(np.float64)
                packet_head = [dangerous_vehicle.size, 0]
                packet_body = dangerous_vehicle.ravel()
                packet = np.insert(packet_body, 0, packet_head)
                packet = packet.astype(np.float64)
                lock.acquire()
                memoryview(VehicleStop).cast('B').cast('d')[0:dangerous_vehicle.size + 2] = packet
                lock.release()
        while time.time() - t1 < 1:
            time.sleep(1 - (time.time() - t1))
        # time.sleep(0.009)


def dangerous_car_detect(flag, detect_area, target_vehicle_dangerous, target_vehicle_error):  ##异常停车检测
    distance_limit = 1  # m
    time_limit = 600
    while flag.value:
        t1 = time.time()
        target_vehicle_dangerous_shape = target_vehicle_dangerous.shape[0]
        for i in range(target_vehicle_error.shape[0]):
            distance_group = []
            if Whether_in_area(target_vehicle_error[i, 2:4], detect_area):
                if target_vehicle_dangerous_shape < 1:
                    temp = np.array(
                        [target_vehicle_error[i, 0], target_vehicle_error[i, 2], target_vehicle_error[i, 3], t1, t1])
                    target_vehicle_dangerous = np.insert(target_vehicle_dangerous, target_vehicle_dangerous.shape[0],
                                                         temp, axis=0)
                else:
                    for j in range(target_vehicle_dangerous.shape[0]):
                        distance = get_distance_from_point_to_point(target_vehicle_dangerous[j, 1:3],
                                                                    target_vehicle_error[i, 2:4])
                        distance_group.append(distance)

                    if np.min(distance_group) < distance_limit:
                        t = distance_group.index(np.min(distance_group))
                        if time.time() - target_vehicle_dangerous[t, 4] < time_limit:
                            target_vehicle_dangerous[t, 4] = time.time()
                        else:
                            target_vehicle_dangerous[t, 1:3] = target_vehicle_error[i, 2:4]
                            target_vehicle_dangerous[t, 3:5] = time.time()
                    else:
                        temp = np.array(
                            [target_vehicle_error[i, 0], target_vehicle_error[i, 2], target_vehicle_error[i, 3], t1,
                             t1])
                        target_vehicle_dangerous = np.insert(target_vehicle_dangerous,
                                                             target_vehicle_dangerous.shape[0],
                                                             values=temp, axis=0)
        # while time.time() - t1 < 0.099:
        #     continue
        return target_vehicle_dangerous


def per_bike_detect(flag, detect_area, PersonBike_group, target_vehicle,tt):
    while flag.value:
        t1 = time.time()
        target_vehicle_shape = target_vehicle.shape[0]
        for i in range(target_vehicle_shape):
            if Whether_in_area(target_vehicle[i, 2:4], detect_area):
                if PersonBike_group.shape[0] < 1:
                    temp = np.array(
                        [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt])
                    PersonBike_group = np.insert(PersonBike_group, PersonBike_group.shape[0], temp, axis=0)
                else:
                    PersonBike_group_ID = (PersonBike_group[:, 0].astype(int)).tolist()
                    if target_vehicle[i, 0] in PersonBike_group_ID:
                        PersonBike_group[np.where(PersonBike_group_ID == target_vehicle[i, 0]), 4] = tt
                    else:
                        temp = np.array([target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt])
                        PersonBike_group = np.insert(PersonBike_group, PersonBike_group.shape[0], temp, axis=0)
        return PersonBike_group


def per_bike_conut(flag, lock, detect_area, Target_vehicle_raw, PersonBike):
    PersonBike_group = np.zeros((0, 5), dtype=np.float64)
    t1 = time.time()
    while flag.value:
        data = np.copy(np.ctypeslib.as_array(Target_vehicle_raw))
        target_vehicle = data[4:int(data[2] + 4)].reshape(-1, 7)
        tt  =data[0]/1000
        target_vehicle = target_vehicle[target_vehicle[:, 1] ==1]
        if target_vehicle.shape[0] > 0:
            PersonBike_group = per_bike_detect(flag, detect_area, PersonBike_group, target_vehicle,tt)

        if PersonBike_group.shape[0] >= 1:
            personBike_group = PersonBike_group
            print('personBike_group', personBike_group)
            personBike_group = personBike_group.reshape(1, -1)
            personBike_group = personBike_group.astype(np.float64)
            packet_head = [personBike_group.size, 0]
            packet_body = personBike_group.ravel()
            packet = np.insert(packet_body, 0, packet_head)
            packet = packet.astype(np.float64)
            lock.acquire()
            memoryview(PersonBike).cast('B').cast('d')[0:personBike_group.size + 2] = packet
            lock.release()
        while time.time() - t1 < 0.099:
            time.sleep(0.1 - (time.time() - t1))


def Speed_conut(flag, lock, highspeed_limit, lowspeed_limit, detect_area, Target_vehicle_raw, highSpeed, lowspeed):
    highSpeed_group = np.zeros((0, 6), dtype=np.float64)
    lowspeed_group = np.zeros((0, 6), dtype=np.float64)
    t1 = time.time()
    while flag.value:
        data = np.copy(np.ctypeslib.as_array(Target_vehicle_raw))
        target_vehicle = data[4:int(data[2] + 4)].reshape(-1, 7)
        target_vehicle = target_vehicle[target_vehicle[:, 1] > 2]
        target_v = np.sqrt(np.square(target_vehicle[:, 4]) + np.square(target_vehicle[:, 5]))
        if target_vehicle.shape[0] <= 0:
            continue
        else:
            highSpeed_group, lowsped_group = Speed_detect(flag,
                                                          detect_area, highSpeed_group, lowspeed_limit,
                                                          highspeed_limit, lowspeed_group, target_vehicle, target_v)
        HighSpeed_group = highSpeed_group
        HighSpeed_group = HighSpeed_group.reshape(1, -1)
        HighSpeed_group = HighSpeed_group.astype(np.float64)
        HighSpeed_packet_head = [HighSpeed_group.size, 0]
        HighSpeed_packet_body = HighSpeed_group.ravel()
        HighSpeed_packet = np.insert(HighSpeed_packet_body, 0, HighSpeed_packet_head)
        HighSpeed_packet = HighSpeed_packet.astype(np.float64)

        LowSpeed_group = lowspeed_group
        LowSpeed_group = LowSpeed_group.reshape(1, -1)
        LowSpeed_group = LowSpeed_group.astype(np.float64)
        LowSpeed_packet_head = [LowSpeed_group.size, 0]
        LowSpeed_packet_body = LowSpeed_group.ravel()
        LowSpeed_packet = np.insert(LowSpeed_packet_body, 0, LowSpeed_packet_head)
        LowSpeed_packet = LowSpeed_packet.astype(np.float64)

        lock.acquire()
        memoryview(highSpeed).cast('B').cast('d')[0:HighSpeed_group.size + 2] = HighSpeed_packet
        memoryview(lowspeed).cast('B').cast('d')[0:LowSpeed_group.size + 2] = LowSpeed_packet
        lock.release()
        while time.time() - t1 < 0.099:
            time.sleep(0.1 - (time.time() - t1))


def Speed_detect(flag, detect_area, highSpeed_group, lowspeed_limit, highspeed_limit,
                 lowspeed_group, target_vehicle, target_v):
    while flag.value:
        t1 = time.time()
        target_vehicle_shape = target_vehicle.shape[0]
        for i in range(target_vehicle_shape):
            for j in range(len(detect_area)):
                if Whether_in_area(target_vehicle[i, 2:4], detect_area[j]):
                    if target_v[i] > highspeed_limit:
                        if highSpeed_group.shape[0] < 1:
                            temp = np.array(
                                [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], t1, t1, j])
                            highSpeed_group = np.insert(highSpeed_group, highSpeed_group.shape[0], temp, axis=0)
                        else:
                            highSpeed_group_ID = (highSpeed_group[:, 0].astype(int)).tolist()
                            if target_vehicle[i, 0] in highSpeed_group_ID:
                                highSpeed_group[np.where(highSpeed_group_ID == target_vehicle[i, 0]), 5] = time.time()
                            else:
                                temp = np.array(
                                    [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], t1, t1, j])
                                highSpeed_group = np.insert(highSpeed_group, highSpeed_group.shape[0], temp, axis=0)
                    elif target_v[i] < lowspeed_limit:
                        if lowspeed_group.shape[0] < 1:
                            temp = np.array(
                                [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], t1, t1, j])
                            lowspeed_group = np.insert(lowspeed_group, lowspeed_group.shape[0], temp, axis=0)
                        else:
                            lowspeed_group_ID = (lowspeed_group[:, 0].astype(int)).tolist()
                            if target_vehicle[i, 0] in lowspeed_group_ID:
                                lowspeed_group[np.where(lowspeed_group_ID == target_vehicle[i, 0]), 5] = time.time()
                            else:
                                temp = np.array(
                                    [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], t1, t1, j])
                                lowspeed_group = np.insert(lowspeed_group, lowspeed_group.shape[0], temp, axis=0)
        return highSpeed_group, lowspeed_group


def Line_change_conut(lock, flag, area1, area2, area3, area4, Target_vehicle_raw, LineChange):
    lineChange_dict=dict()
    t1 = time.time()
    detect_area= [area1, area2, area3, area4]
    while flag.value:
        data = np.copy(np.ctypeslib.as_array(Target_vehicle_raw))
        target_vehicle = data[4:int(data[2] + 4)].reshape(-1, 7)
        tt = data[0]/1000
        target_vehicle = target_vehicle[target_vehicle[:, 1] > 2]
        for vehicle in target_vehicle:
            for j in range(len(detect_area)):
                if Whether_in_area(vehicle[2:4], detect_area[j]):  ##判断车辆在哪个检测区域（车道）
                    if vehicle[0] not in lineChange_dict.keys():  ##若目标级字典中无此目标的ID 则新建目标数据集
                        values = np.zeros((1, 2), dtype=np.float64)
                        values[0] = vehicle[2:4]
                        sline = j+1
                        eline = j+1
                        start_time = tt
                        end_time = tt
                        lineChange_dict[vehicle[0]] = [values, sline, eline, start_time, end_time]
                        ##[x,y],sline,eline,stime,endtime
                    else:
                        if (j+1) != lineChange_dict[vehicle[0]][2]:
                            lineChange_dict[vehicle[0]][0] = vehicle[2:4]
#                            lineChange_dict[vehicle[0]][1] = lineChange_dict[vehicle[0]][2]
                            lineChange_dict[vehicle[0]][2] = j+1
                            lineChange_dict[vehicle[0]][3] = lineChange_dict[vehicle[0]][4]
                            lineChange_dict[vehicle[0]][4] = tt
                        else:
                            lineChange_dict[vehicle[0]][3] = tt

        Line_change_group = []
        id_list = lineChange_dict.keys()
        for id in id_list:
            if lineChange_dict[id][1]!=lineChange_dict[id][2]:
                temp = [id, lineChange_dict[id][0][0], lineChange_dict[id][0][1],
                        lineChange_dict[id][3],lineChange_dict[id][4], lineChange_dict[id][1],lineChange_dict[id][2]]
                Line_change_group.append(temp)
        Linechange_group = np.array(Line_change_group)
        print('Linechange_group', Linechange_group)
        Linechange_group = Linechange_group.reshape(1, -1)
        Linechange_group = Linechange_group.astype(np.float64)
        Linechange_packet_head = [Linechange_group.size, 0]
        Linechange_packet_body = Linechange_group.ravel()
        Linechange_packet = np.insert(Linechange_packet_body, 0, Linechange_packet_head)
        Linechange_packet = Linechange_packet.astype(np.float64)
        lock.acquire()
        memoryview(LineChange).cast('B').cast('d')[0:Linechange_group.size + 2] = Linechange_packet
        lock.release()
        while time.time() - t1 < 1:
            time.sleep(1 - (time.time() - t1))
            # continue


def line_change_detect(flag, area1, area2, area3, area4, Line_change_group, target_vehicle, tt):
    while flag.value:
        t1 = time.time()
        target_vehicle_shape = target_vehicle.shape[0]
        for i in range(target_vehicle_shape):
            if Whether_in_area(target_vehicle[i, 2:4], area1):
                if get_distance_from_point_to_line(target_vehicle[i, 2:4], area1[2, :], area1[3, :]) <= 0.5:
                    if Line_change_group.shape[0] < 1:
                        temp = np.array(
                            [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 1, 2])
                        Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
                    else:
                        Line_change_group_ID = (Line_change_group[:, 0].astype(int)).tolist()
                        if target_vehicle[i, 0] in Line_change_group_ID:
                            Line_change_group[np.where(Line_change_group_ID == target_vehicle[i, 0]), 5] = tt
                        else:
                            temp = np.array(
                                [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 1, 2])
                            Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
            elif Whether_in_area(target_vehicle[i, 2:4], area2):
                if get_distance_from_point_to_line(target_vehicle[i, 2:4], area2[2, :], area2[3, :]) <= 0.5:
                    if Line_change_group.shape[0] < 1:
                        temp = np.array(
                            [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 2, 3])
                        Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
                    else:
                        Line_change_group_ID = (Line_change_group[:, 0].astype(int)).tolist()
                        if target_vehicle[i, 0] in Line_change_group_ID:
                            Line_change_group[np.where(Line_change_group_ID == target_vehicle[i, 0]), 5] = tt
                        else:
                            temp = np.array(
                                [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 2, 3])
                            Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
                elif get_distance_from_point_to_line(target_vehicle[i, 2:4], area1[2, :], area1[3, :]) <= 0.5:
                    if Line_change_group.shape[0] < 1:
                        temp = np.array(
                            [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 2, 1])
                        Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
                    else:
                        Line_change_group_ID = (Line_change_group[:, 0].astype(int)).tolist()
                        if target_vehicle[i, 0] in Line_change_group_ID:
                            Line_change_group[np.where(Line_change_group_ID == target_vehicle[i, 0]), 5] = tt
                        else:
                            temp = np.array(
                                [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 2, 1])
                            Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
            elif Whether_in_area(target_vehicle[i, 2:4], area3):
                if get_distance_from_point_to_line(target_vehicle[i, 2:4], area3[2, :], area3[3, :]) <= 0.5:
                    if Line_change_group.shape[0] < 1:
                        temp = np.array(
                            [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 3, 4])
                        Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
                    else:
                        Line_change_group_ID = (Line_change_group[:, 0].astype(int)).tolist()
                        if target_vehicle[i, 0] in Line_change_group_ID:
                            Line_change_group[np.where(Line_change_group_ID == target_vehicle[i, 0]), 5] = tt
                        else:
                            temp = np.array(
                                [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 3, 4])
                            Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
                elif get_distance_from_point_to_line(target_vehicle[i, 2:4], area2[2, :], area2[3, :]) <= 0.5:
                    if Line_change_group.shape[0] < 1:
                        temp = np.array(
                            [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 3, 2])
                        Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
                    else:
                        Line_change_group_ID = (Line_change_group[:, 0].astype(int)).tolist()
                        if target_vehicle[i, 0] in Line_change_group_ID:
                            Line_change_group[np.where(Line_change_group_ID == target_vehicle[i, 0]), 5] = tt
                        else:
                            temp = np.array(
                                [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 3, 2])
                            Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
            elif Whether_in_area(target_vehicle[i, 2:4], area4):
                if get_distance_from_point_to_line(target_vehicle[i, 2:4], area3[2, :], area3[3, :]) <= 0.5:
                    if Line_change_group.shape[0] < 1:
                        temp = np.array(
                            [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 4, 3])
                        Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)
                    else:
                        Line_change_group_ID = (Line_change_group[:, 0].astype(int)).tolist()
                        if target_vehicle[i, 0] in Line_change_group_ID:
                            Line_change_group[np.where(Line_change_group_ID == target_vehicle[i, 0]), 5] = tt
                        else:
                            temp = np.array(
                                [target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], tt, tt, 4, 3])
                            Line_change_group = np.insert(Line_change_group, Line_change_group.shape[0], temp, axis=0)

        return Line_change_group


def traffic_jam_detect(lock, TrafficJam, flag, Target_vehicle_raw, line_UTM1, line_UTM2,
                       traffic_area):  ## traffic jag and vehicle length
    vehicle_num_temp = 3
    max_vehicle_length = 12
    vehicle_v_temp = 500  # cm/s
    while flag.value:
        target_vehicle_length = []
        data = np.copy(np.ctypeslib.as_array(Target_vehicle_raw))
        target_vehicle = data[4:int(data[2] + 4)].reshape(-1, 7)
        for i in range(target_vehicle.shape[0]):
            if target_vehicle[i, 1] != 3:
                np.delete(target_vehicle, i, 0)
        if target_vehicle.shape[0] <= 0:
            continue

        target_v = np.sqrt(np.square(target_vehicle[:, 4]) + np.square(target_vehicle[:, 5]))
        target_vehicle_error = target_vehicle[target_v < vehicle_v_temp]
        t1 = time.time()
        if target_vehicle_error.shape[0] < vehicle_num_temp:
            continue
        if target_vehicle_error.shape[0] >= vehicle_num_temp:
            for i in range(target_vehicle_error.shape[0]):
                if Whether_in_area(target_vehicle_error[i, 2:4], traffic_area):
                    temp = get_distance_from_point_to_line(target_vehicle[i, 2:4], line_UTM1, line_UTM2)
                    target_vehicle_length.append(temp)
        # print(target_vehicle_length)
        if len(target_vehicle_length) < 1:
            # print("There has no traffic jag")
            vehicle_length = 0
        else:
            max_target_vehicle_length = np.max(target_vehicle_length)
            t1 = time.time()
            vehicle_length = max_vehicle_length if np.ceil(
                max_target_vehicle_length) >= max_vehicle_length else np.ceil(max_target_vehicle_length)
            # if vehicle_length < 1:
            #     print("There has no traffic jag")
            # elif vehicle_length < max_vehicle_length:
            #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)))
            #     print('There has small traffic jag, the vehicle length is {}'.format(vehicle_length))
            # else:
            #     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
            #     print('There has serious traffic jag, the vehicle length is above {}'.format(vehicle_length))
        if vehicle_length > 5:
            traffic_jam = np.vstack((np.array(t1), np.array(vehicle_length)))
            traffic_jam = traffic_jam.reshape(1, -1)
            traffic_jam = traffic_jam.astype(np.float64)
            packet_head = [traffic_jam.size, 0]
            packet_body = traffic_jam.ravel()
            packet = np.insert(packet_body, 0, packet_head)
            packet = packet.astype(np.float64)
            lock.acquire()
            memoryview(TrafficJam).cast('B').cast('d')[0:traffic_jam.size + 2] = packet
            lock.release()
            # print(packet)
            while time.time() - t1 < 0.099:
                time.sleep(0.1 - (time.time() - t1))


def vehicle_direction_detect(lock, VehicleConverse, flag, Target_vehicle_raw, road_heading, detect_area):  ##逆行检测发送
    vehicle_num_temp = 10
    vehicle_v_temp = 500
    vehicle_direction_error = np.zeros((0, 6), dtype=np.float64)
    t2 = time.time()
    while flag.value:
        t1 = time.time()
        data = np.copy(np.ctypeslib.as_array(Target_vehicle_raw))
        target_vehicle = data[4:int(data[2] + 4)].reshape(-1, 7)
        for i in range(target_vehicle.shape[0]):
            if target_vehicle[i, 1] != 3:
                np.delete(target_vehicle, i, 0)
        if target_vehicle.shape[0] <= 0:
            continue
        target_v = np.sqrt(np.square(target_vehicle[:, 4]) + np.square(target_vehicle[:, 5]))
        target_vehicle = target_vehicle[target_v > vehicle_v_temp]
        if 1 <= target_vehicle.shape[0] <= vehicle_num_temp:
            vehicle_direction_error = vehicle_direction_count(flag, vehicle_direction_error, target_vehicle,
                                                              road_heading, detect_area)
        else:
            vehicle_direction_error = np.zeros((0, 6), dtype=np.float64)
        # if time.time() - t1 > 5:
        vehicle_direction_error2 = vehicle_direction_error[vehicle_direction_error[:, 0] > 3]
        if vehicle_direction_error2.shape[0] >= 1:
            # print('Now time is:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
            # print('There is {} vehicle drive direction error:'.format(vehicle_direction_error.shape[0]),
            #       vehicle_direction_error)
            vehicleConverse = vehicle_direction_error2[:, 1:6]  # id x,y,start_time,end_time
            vehicleConverse = vehicleConverse.reshape(1, -1)
            vehicleConverse = vehicleConverse.astype(np.float64)
            packet_head = [vehicleConverse.size, 0]
            packet_body = vehicleConverse.ravel()
            packet = np.insert(packet_body, 0, packet_head)
            packet = packet.astype(np.float64)
            lock.acquire()
            memoryview(VehicleConverse).cast('B').cast('d')[0:vehicleConverse.size + 2] = packet
            lock.release()
            # print(packet)
        if time.time() - t2 > 30:
            vehicle_direction_error = np.zeros((0, 6), dtype=np.float64)
            t2 = time.time()
        while time.time() - t1 < 0.099:
            time.sleep(0.1 - (time.time() - t1))


def vehicle_direction_count(flag, vehicle_direction_error, target_vehicle, road_heading, detect_area):  ##逆行检测
    t1 = time.time()
    vehicle_direction_error_shape = vehicle_direction_error.shape[0]
    for i in range(target_vehicle.shape[0]):
        if Whether_in_area(target_vehicle[i, 2:4], detect_area) and whether_vehicle_drive_direction_error(
                target_vehicle[i, 6], road_heading):
            if vehicle_direction_error_shape < 1:
                temp = np.array([0, target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], t1, t1])
                vehicle_direction_error = np.insert(vehicle_direction_error, vehicle_direction_error.shape[0], temp,
                                                    axis=0)
            else:
                vehicle_direction_error_ID = (vehicle_direction_error[:, 1].astype(int)).tolist()
                if target_vehicle[i, 0] in vehicle_direction_error_ID:
                    vehicle_direction_error[np.where(vehicle_direction_error_ID == target_vehicle[i, 0]), 0] += 1
                    vehicle_direction_error[
                        np.where(vehicle_direction_error_ID == target_vehicle[i, 0]), 5] = time.time()
                else:
                    temp = np.array(
                        [0, target_vehicle[i, 0], target_vehicle[i, 2], target_vehicle[i, 3], t1, t1])
                    vehicle_direction_error = np.insert(vehicle_direction_error,
                                                        vehicle_direction_error.shape[0], temp,
                                                        axis=0)
        # while time.time() - t1 < 0.1:
        #     continue
        return vehicle_direction_error


def isinpolygon(point, areas):
    def creat_areas(area):
        la = int(area.shape[0])
        if (area[la - 1] - area[0]).any():
            area = np.vstack((area, area[0]))
        areas = np.zeros((la - 1, 6), np.float64)
        areas[:, :2], areas[:, 2:4] = area[:la - 1], area[1:la]
        areas[:, 4] = (areas[:, 2] - areas[:, 0]) / (areas[:, 3] - areas[:, 1])
        areas[:, 5] = areas[:, 0] - areas[:, 4] * areas[:, 1]
        return areas

    x = point[0]
    y = point[1]
    cross = 0

    if areas.shape[0] < 3:
        return 1  # 区域不构成多边形
    elif areas.shape[1] == 2:  # 只包含多边形顶点
        areas = creat_areas(areas)
        # areas=areas
    for i in range(areas.shape[0]):  # 判断是否在顶点上
        if x == areas[i, 0] and y == areas[i, 1]: return 1
    for i in range(areas.shape[0]):  # 对于不在顶点上的情形
        if (areas[i, 0] < x) and (areas[i, 2] < x):
            continue  # 两个点皆该点左面
        elif (areas[i, 1] > y) and (areas[i, 3] > y):
            continue  # 两个点皆在该点上面
        elif (areas[i, 1] < y) and (areas[i, 3] < y):
            continue  # 两个点皆在下面
        elif abs(areas[i, 4]) == np.inf:  # 斜率无限大，线段两端y相等，线段与x轴平行且与该点射线重合于一条直线
            if areas[i, 0] <= x <= areas[i, 2]:
                return 1  # 该点在多边型的边上
            # 其它情况下穿过的端点数必然为偶数
        else:
            x0 = int(round(areas[i, 4] * y + areas[i, 5]))
            if x0 == x:
                return 1
            elif x0 > x:
                if y == areas[i, 1]:  # 线段起点在射线上
                    j = i - 1 if i > 0 else int(areas.shape[0]) - 1  # 寻找上一截的y
                    if (areas[i, 3] - y) * (areas[j, 1] - y) < 0:
                        continue
                cross += 1
    return cross % 2


def Whether_in_area(data, area):  ## target whether in area
    # A = cv2.pointPolygonTest(area, (data[0], data[1]), False)
    A = isinpolygon(data, area)
    return True if A > 0 else False


def cal_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def emergence_detect(flag, lock, Target_vehicle_raw, area, VehicleEmergency):  ##占用应急车道
    vehicle_v_temp = 200  # cm/s
    while flag.value:
        t1 = time.time()
        data = np.copy(np.ctypeslib.as_array(Target_vehicle_raw))
        target_vehicle_1 = data[4:int(data[2] + 4)].reshape(-1, 7)
        tt = data[0]/1000
        target_vehicle = target_vehicle_1[target_vehicle_1[:, 1] >= 3]
        if target_vehicle.shape[0] < 1:
            continue
        target_v = np.sqrt(np.square(target_vehicle[:, 4]) + np.square(target_vehicle[:, 5]))
        target_vehicle_error = target_vehicle[target_v >= vehicle_v_temp]
        em_list = []
        for i in range(target_vehicle_error.shape[0]):
            if Whether_in_area(target_vehicle[i, 2:4], area):
                temp = np.array([target_vehicle_error[i, 0], target_vehicle_error[i, 2], target_vehicle_error[i, 3],
                                 tt, tt])  ## id x y start_time  end_time
                em_list.append(temp)
        em_array = np.array(em_list)
        em_array = em_array.reshape(1, -1)
        em_array = em_array.astype(np.float64)
        packet_head = [em_array.size, 0]
        packet_body = em_array.ravel()
        packet = np.insert(packet_body, 0, packet_head)
        packet = packet.astype(np.float64)
        lock.acquire()
        memoryview(VehicleEmergency).cast('B').cast('d')[0:em_array.size + 2] = packet
        lock.release()
        while time.time() - t1 < 0.099:
            time.sleep(0.1 - (time.time() - t1))


def Volumn_count(count1, count2, count3, count4, target_vehicle, area1, area2, area3, area4):  ## volumn count
#    print("area1",area1)
    for i in range(target_vehicle.shape[0]):
        if Whether_in_area(target_vehicle[i, 2:4], area1) and target_vehicle[i, 0] not in count1:
            count1.append(target_vehicle[i, 0])
        elif Whether_in_area(target_vehicle[i, 2:4], area2) and target_vehicle[i, 0] not in count2:
            count2.append(target_vehicle[i, 0])
        elif Whether_in_area(target_vehicle[i, 2:4], area3) and target_vehicle[i, 0] not in count3:
            count3.append(target_vehicle[i, 0])
        elif Whether_in_area(target_vehicle[i, 2:4], area4) and target_vehicle[i, 0] not in count4:
            count4.append(target_vehicle[i, 0])
#        else:
#            print('none volmue')
    return count1, count2, count3, count4


def Volumn_send(lock, VehicleVolume, flag, Target_vehicle_raw, area1, area2, area3, area4):  ## send traffic volumn
    conut = np.zeros((1, 7), dtype=np.float64)
    send_cycle = 30
    while flag.value:
        count1, count2, count3, count4 = [], [], [], []
        tick = time.time()
        
        # conut[0, 0] = tick
        flag2 = 1
        while time.time() - tick <= send_cycle:
            tick2 =time.time()
            data = np.copy(np.ctypeslib.as_array(Target_vehicle_raw))
            tt = data[0]/1000
            if flag2 ==1:
                conut[0, 0] = tt
                flag2 = 0
            target_vehicle_1 = data[4:int(data[2] + 4)].reshape(-1, 7)
            target_vehicle = []
            for i in range(target_vehicle_1.shape[0]):
                if target_vehicle_1[i, 1] == 3:
                    target_vehicle.append(target_vehicle_1[i, :])
            target_vehicle = np.array(target_vehicle)
#            print("_____________________________")
#            print(target_vehicle)
            # target_v = np.sqrt(np.square(target_vehicle[:, 4]) + np.square(target_vehicle[:, 5]))
            count1, count2, count3, count4 = Volumn_count(count1, count2, count3, count4, target_vehicle, area1, area2,
                                                          area3, area4)
            
#            print(count1, count2,count3,count4)
            while time.time() - tick2 < 1:
                time.sleep(1 - (time.time() - tick2))
        conut[0, 3] = len(count1)
        conut[0, 4] = len(count2)
        conut[0, 5] = len(count3)
        conut[0, 6] = len(count4)
        conut[0, 2] = sum(conut[0, 3:7])
        conut[0, 1] = tt
        if conut[0, 2] >= 0:
            volumn = np.array(conut)
            volumn = volumn.reshape(1, -1)
            volumn = volumn.astype(np.float64)
            packet_head = [volumn.size, 0]
            packet_body = volumn.ravel()
            packet = np.insert(packet_body, 0, packet_head)
            packet = packet.astype(np.float64)
            lock.acquire()
            memoryview(VehicleVolume).cast('B').cast('d')[0:volumn.size + 2] = packet
            lock.release()
#        while time.time() - tick < 0.099:
#            time.sleep(0.1 - (time.time() - tick))


def get_distance_from_point_to_line(point, line_point1, line_point2):  ## calculate distance from point to line
    line_point1 = [line_point1[0], line_point2[1]]
    line_point2 = [line_point1[1], line_point2[1]]
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + (line_point2[0] - line_point1[0]) * line_point1[1]
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A ** 2 + B ** 2))
    return distance


def get_distance_from_point_to_point(point1, point2):
    distance = np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))
    return distance


def whether_vehicle_drive_direction_error(vehicle_heading, road_heading):
    limit_heading = 90
    drive_direction_error_status = True if np.abs(vehicle_heading - np.mod(road_heading + 180,
                                                                           360)) < limit_heading else False
    return drive_direction_error_status


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    flag = Value('i', 1)  # 0 指停止运行, 1 指正常运行
    Target_vehicle_raw = RawArray('d', 100)  ## 数据
    VehicleConverse = RawArray('d', 1024)  ## 逆行
    VehicleVolume = RawArray('d', 1024)  ## 流量
    TrafficJam = RawArray('d', 1024)  ## 交通拥堵及排队长度
    VehicleStop = RawArray('d', 1024)  ## 危险停车
    Highspeed = RawArray('d', 1024)  ## 交通拥堵及排队长度
    Lowspeed = RawArray('d', 1024)  ## 危险停车
    VehicleEmergency = RawArray('d', 1024)
    lock = Lock()
    class_traffic = 4  ## 1 Volumn 2:dangerous_car_count 3: traffic_jam_detect 4:vehicle_direction_detect
    send_cycle = 5
    send_staus = False
    restrict_v = 1000
    restrict_v2 = 2000
    count = 0
    traffic_area1 = np.array([[1, 5], [1, 18], [18, 18], [18, 5]])
    volumn_area = np.array([[1, 0.8], [1, 1.5], [18, 1.5], [18, 0.8]])
    volumn_area2 = np.array([[1, 1.6], [1, 4.6], [18, 4.6], [18, 1.6]])
    detect_area = np.array([[0.5, 0.5], [0.5, 20], [20, 20], [20, 0.5]])

    traffic_area = [traffic_area1, volumn_area, volumn_area2, detect_area]
    line_UTM1 = [1, 18]
    line_UTM2 = [5, 5]
    road_heading = 180

    Process(target=multi_data_producation, args=(Target_vehicle_raw, lock, class_traffic)).start()
    # Process(target=draw_process,
    #         args=(
    #             Target_vehicle_raw, volumn_area, volumn_area2, traffic_area, detect_area, line_UTM1, line_UTM2)).start()

    # if class_traffic == 1:
    #     Process(target=Volumn_send,
    #             args=(lock, VehicleVolume, flag, Target_vehicle_raw, volumn_area,
    #                   volumn_area2, restrict_v)).start()
    # elif class_traffic == 2:
    #     Process(target=dangerous_car_count, args=(lock, VehicleStop, flag, Target_vehicle_raw, detect_area)).start()
    # elif class_traffic == 3:
    #     Process(target=traffic_jam_detect,
    #             args=(lock, TrafficJam, flag, Target_vehicle_raw, line_UTM1, line_UTM2, traffic_area)).start()
    # elif class_traffic == 4:
    Process(target=traffic_event_detect,
            args=(flag, Target_vehicle_raw, traffic_area, lock, road_heading, VehicleStop, Highspeed, Lowspeed,
                  VehicleConverse, VehicleEmergency)).start()
    # elif class_traffic == 5:
    #     pass
    tt = time.time()
    while flag.value:
        Volumedata = np.copy(np.ctypeslib.as_array(VehicleVolume))
        Volumedata = Volumedata[2:int(Volumedata[0] + 2)].reshape(-1, 5)  ##time, volume
        VehicleStopdata = np.copy(np.ctypeslib.as_array(VehicleStop))
        VehicleStopdata = VehicleStopdata[2:int(VehicleStopdata[0] + 2)].reshape(-1, 5)  ## x, y, start_time, end_time
        TrafficJamdata = np.copy(np.ctypeslib.as_array(TrafficJam))
        TrafficJamdata = TrafficJamdata[2:int(TrafficJamdata[0] + 2)].reshape(-1, 5)  ## time, vehicle_length
        VehicleConversedata = np.copy(np.ctypeslib.as_array(VehicleConverse))
        VehicleConversedata = VehicleConversedata[2:int(VehicleConversedata[0] + 2)].reshape(-1, 5)  ## ID, x, y, time

        if time.time() - tt > send_cycle:
            # print(VehicleStopdata)
            print('111', Volumedata)
            if Volumedata.shape[0] > 0:
                print(Volumedata[0, 1])
            #     if VehicleStopdata.shape[0] > 0:
            #         print(VehicleStopdata[0,:])
            # print(TrafficJamdata)
            # if VehicleConversedata.shape[0] > 0:
            #     print(VehicleConversedata[0, :])
            tt = time.time()
