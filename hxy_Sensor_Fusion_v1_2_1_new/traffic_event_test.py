import time,pickle,os
import numpy as np
from multiprocessing import Value, RawArray, Process, Lock
from process_test import *
from traffic_event import send_traffic_event
from get_region import get_XY
from pyproj import Proj


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def coordinate_transform(mode, intput_data):
    converter = Proj(proj='utm', zone=48, ellps='WGS84', south=False, north=True, errcheck=True)
    if mode == 'gps2utm':
        x, y = converter(intput_data[1], intput_data[0], inverse=False)  # x for East, y for North
        return [x, y]
    elif mode == 'utm2gps':
        lon, lat = converter(intput_data[0], intput_data[1], inverse=True)
        return [lat, lon]


def traffic_event_detect(flag, location,target_Raw, lock, event_sent_raw, Volume_sent_raw):
    filename = "detect_area.xml"
    filename1 = "detect_area1.xml"
    VehicleConverse = RawArray('d', 1024)  ## 逆行数据
    VehicleVolume = RawArray('d', 1024)  ## 流量数据
    VehicleStop = RawArray('d', 1024)  ## 危险停车数据
    VehicleEmergency = RawArray('d', 1024)  ##占用应急车道数据
    PersonBike = RawArray('d', 1024)  ##行人闯入数据
    highspeed = RawArray('d', 1024)  ##超速数据
    lowspeed = RawArray('d', 1024)  ##低速数据
    linechange = RawArray('d', 1024)  ##变道数据
    Radar_IP = '192.168.6.211'
    restrict_v = 72 / 3.6 * 100  ##流量检测 速度界限值
    highspeed_limit = 40 / 3.6 * 100
    lowspeed_limit = 20 / 3.6 * 100
    ##区域限定
    P4 = coordinate_transform('gps2utm', [104.2398811, 30.56871362])
    P5 = coordinate_transform('gps2utm', [104.2399154, 30.56885976])
    P6 = coordinate_transform('gps2utm', [104.2407264, 30.56850347])
    P7 = coordinate_transform('gps2utm', [104.2407946, 30.56864426])
    P8 = coordinate_transform('gps2utm', [104.2407705, 30.56860354])
    P9 = coordinate_transform('gps2utm', [104.2404242, 30.56869522])
    P10 = coordinate_transform('gps2utm', [104.2404355, 30.56872668])

    line1 = [P6, P4]
    line2 = [P10, P9]
    x, y = line_intersection(line1, line2)  #
    ##需采集四个点
#    A = coordinate_transform('gps2utm', [])
#    B = coordinate_transform('gps2utm', [])
#    C = coordinate_transform('gps2utm', [])
#    D = coordinate_transform('gps2utm', [])
    line1_area = get_XY(filename, 3)  ## 车道1区域范围
    line2_area = get_XY(filename, 2)  ##车道2区域范围
    line3_area = get_XY(filename, 1)  ##车道3区域范围,逆行检测范围
    line4_area = get_XY(filename, 0) ##车道4区域范围,异常停车、占用应急车道、道路遗撒、行人闯入 检测范围
    whole_detect = [line1_area, line2_area, line3_area, line4_area]  ##全部检测范围, 超低速检测范围
    line_1_area = get_XY(filename1, 3)  ## 车道1 liu检测区域
    line_2_area = get_XY(filename1, 2)  ## 车道2 检测区域
    line_3_area = get_XY(filename1, 1)  ## 车道3 检测区域
    line_4_area = get_XY(filename1, 0)  ## 车道4 检测区域
#    whole_detect = [line_1_area, line_2_area, line_3_area, line_4_area]  ##全部检测范围, 超低速检测范围

    for pickle_name in os.listdir("location_H"):
        if location.split('_')[0] in pickle_name:
            break
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data = pickle.load(pickle_file)[Radar_IP]
    road_heading = int(data['Heading'] * 10)
    # road_heading = 316  ## 道路航向角
    Process(target=Volumn_send, args=(lock, VehicleVolume, flag, target_Raw,
                                      line_1_area, line_2_area,
                                      line_3_area, line_4_area)).start()  ## 流量检测进程
    # Process(target=dangerous_car_count, args=(lock, VehicleStop, flag, target_Raw, line4_area)).start()  ##违停检测进程
    # Process(target=vehicle_direction_detect,
    #         args=(lock, VehicleConverse, flag, target_Raw, road_heading, line3_area)).start()  ## 逆行检测进程
    # Process(target=emergence_detect, args=(flag, lock, target_Raw, line4_area, VehicleEmergency)).start()  ## 占用应急车道检测进程
#    Process(target=per_bike_conut, args=(flag, lock, line4_area, target_Raw, PersonBike)).start()
#    # Process(target=Speed_conut,
#    #         args=(flag, lock, highspeed_limit, lowspeed_limit, whole_detect, target_Raw, highspeed, lowspeed)).start()
#    Process(target=Line_change_conut,
#            args=(lock, flag, line1_area, line2_area, line3_area, line4_area, target_Raw, linechange)).start()
#    Process(target=traffic_event_detect_cdcs,
#            args=(flag, target_Raw, whole_detect, lock, road_heading, VehicleStop, highspeed, lowspeed,
#                  VehicleConverse, VehicleEmergency)).start()
    EventTypelist = ['0', '6', '3', '4', '2', '1', '11', '8']
    EventRadius = [200, 100, 200, 100, 300, 500, 200, 300]
    # #异常停车、实线变道、超速、低速、占用应用车道、逆行、道路遗撒、行人闯入
    # 影响半径
    ## 
    # # traffic_event = np.zeros((0, 5), dtype=np.float64)  ## type, x, y, start_time, end_time, data
    num = 0
    enventID = 0
    while flag.value:
        tick = time.time()
        Volumedata = np.copy(np.ctypeslib.as_array(VehicleVolume))  ##  流量数据
        Volumedata = Volumedata[2:int(Volumedata[0] + 2)].reshape(-1, 7)
#        print(Volumedata)
        # start_time, end_time, sum_volume, line1_volume, line2_volume,line3_volume,line4_volume,
        #
        VehicleStopdata = np.copy(np.ctypeslib.as_array(VehicleStop))  ## 违法停车 --> line 4
        VehicleStopdata = VehicleStopdata[2:int(VehicleStopdata[0] + 2)].reshape(-1, 5)
        ## ID,x, y, start_time, end_time

        VehicleConversedata = np.copy(np.ctypeslib.as_array(VehicleConverse))  ## 道路逆行 -->3
        VehicleConversedata = VehicleConversedata[2:int(VehicleConversedata[0] + 2)].reshape(-1, 5)
        ## id x, y, stime,etime
        VehicleEmergencydata = np.copy(np.ctypeslib.as_array(VehicleEmergency))  ##占用应急车道 --> line 4
        VehicleEmergencydata = VehicleEmergencydata[2:int(VehicleEmergencydata[0] + 2)].reshape(-1, 5)
        ## id x, y, stime, etime

        PersonBikedata = np.copy(np.ctypeslib.as_array(PersonBike))  ##行人闯入 --> line 4
        PersonBikedata = PersonBikedata[2:int(PersonBikedata[0] + 2)].reshape(-1, 5)
        ## id x y stime etime

        Highspeeddata = np.copy(np.ctypeslib.as_array(highspeed))  ## 超速
        Highspeeddata = Highspeeddata[2:int(Highspeeddata[0] + 2)].reshape(-1, 6)
        Lowspeeddata = np.copy(np.ctypeslib.as_array(lowspeed))  ##低速
        Lowspeeddata = Lowspeeddata[2:int(Lowspeeddata[0] + 2)].reshape(-1, 6)
        ## id x y stime  etime line

        Linechangedata = np.copy(np.ctypeslib.as_array(linechange))  ##实线变道
        Linechangedata = Linechangedata[2:int(Linechangedata[0] + 2)].reshape(-1, 7)
        ## id x y stime etime sline eline

        traffic_event = []
        for i in VehicleStopdata:
            temp = ['5', EventTypelist[0], '5', i[1], i[2], i[3], i[4] - i[3], EventRadius[0], '1',
                    '4', i[0]]
            traffic_event.append(temp)

        for i in VehicleConversedata:
            temp = ['5', EventTypelist[5], '5', i[1], i[2], i[3], i[4] - i[3], EventRadius[5], '1',
                    '4', i[0]]
            traffic_event.append(temp)

        for i in VehicleEmergencydata:
            enventID += 1
            temp = ['5', EventTypelist[4], '5', i[1], i[2], i[3], i[4] - i[3], EventRadius[4], '1',
                    '4', i[0]]
            traffic_event.append(temp)

        for i in PersonBikedata:
            temp = ['5', EventTypelist[7], '5', i[1], i[2], i[3], i[4] - i[3], EventRadius[7], '1',
                    '4', i[0]]
            traffic_event.append(temp)

        for i in Highspeeddata:
            temp = ['5', EventTypelist[2], '5', i[1], i[2], i[3], i[4] - i[3], EventRadius[2], '1',
                    '4', i[0]]
            traffic_event.append(temp)

        for i in Lowspeeddata:
            temp = ['5', EventTypelist[3], '5', i[1], i[2], i[3], i[4] - i[3], EventRadius[3], '1',
                    '4', i[0]]
            traffic_event.append(temp)

        for i in Linechangedata:
            temp = ['5', EventTypelist[1], '5', i[1], i[2], i[3], i[4] - i[3], EventRadius[1], '1',
                    '4', i[0]]
            traffic_event.append(temp)

        #     ## 厂商、事件类型、事件来源、经度、维度、开始时间、持续时间、作用半径、置信度、所属车道号、涉及感知ID列表
        #     traffic_volumn = [] ## 帧序号、时间戳、厂商、开始时间、结束时间、总流量、车道编号、车道流量、车道编号、车道流量······
        traffic_volumn = np.array(Volumedata).reshape(1, -1)
        volumn_packet_head = [traffic_volumn.size, 0]
        volumn_packet_body = traffic_volumn.ravel()
        volumn_packet = np.insert(volumn_packet_body, 0, volumn_packet_head)
        volumn_packet = volumn_packet.astype(np.float64)

        traffic_event = np.array(traffic_event).reshape(1, -1)
        packet_head = [traffic_event.size, 0]
        packet_body = traffic_event.ravel()
        packet = np.insert(packet_body, 0, packet_head)
        packet = packet.astype(np.float64)
        lock.acquire()
        memoryview(Volume_sent_raw).cast('B').cast('d')[0:traffic_volumn.size + 2] = volumn_packet
        memoryview(event_sent_raw).cast('B').cast('d')[0:traffic_event.size + 2] = packet
        lock.release()
        time.sleep(0.009)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    flag = Value('i', 1)  # 0 指停止运行, 1 指正常运行
    Detect_data = RawArray('c', 1024)  ## 检测数据
    event_sent_raw = RawArray('d', 1024)  ## 交通事件数据
    Volume_sent_raw = RawArray('d', 1024)  ## 其他交通事件数据
    lock = Lock()
    Process(target=traffic_event_detect,
            args=(flag, Detect_data, lock, event_sent_raw, Volume_sent_raw)).start()

    # while True:
    #     data = np.copy(np.ctypeslib.as_array(event_sent_raw))
    #     event_sent_data = data[2:int(data[0] + 2)].astype(np.str).reshape(-1, 6)
    #     for i in range(event_sent_data.shape[0]):
    #         event_sent_data[i, 0] = '0' + np.str(event_sent_data[i, 0][0:-2])
    #     print(event_sent_data)
