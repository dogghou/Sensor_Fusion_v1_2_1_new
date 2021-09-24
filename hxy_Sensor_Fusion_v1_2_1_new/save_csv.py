import os
import time,datetime
import random
import csv
import numpy as np
from pyproj import Proj
from pynput.keyboard import Key,Listener
from threading import Thread
from traffic_event_test import traffic_event_detect
class CSV:
    def __init__(self,target_Raw, event_raw,flow_raw):
        self.converter = Proj(proj='utm', zone=48, ellps='WGS84', south=False, north=True, errcheck=True)
        self.sensor_type_dict = {'camera': 3, 'radar': 4, 'lidar': 6}   # 传感器类型对应的数字
        self.width_scale = [100, 100, 250, 300, 300]
        self.length_scale = [100, 200, 400, 800, 800]
        self.height_scale = [200, 200, 200, 400, 400]
        self.running=1

        date=str(datetime.datetime.now()).strip().split(' ')[0]
        month,day=date.split('-')[1],date.split('-')[2]
        self.dirpath=os.path.join(os.environ['HOME'], 'Sensor_Fusion_x64','data',month+day)
        if not os.path.exists(self.dirpath):
            #os.mkdir(self.dirpath)
            os.makedirs(self.dirpath)

        ptcFile=os.path.join(self.dirpath,'ptc.csv')
        if os.path.exists(ptcFile):
            self.f_ptc=open(ptcFile,'a',newline='')
            self.writer_ptc=csv.writer(self.f_ptc,dialect='excel')
        else:
            self.f_ptc=open(ptcFile,'w',newline='')
            self.writer_ptc=csv.writer(self.f_ptc,dialect='excel')
            self.writer_ptc.writerow(['帧序号','t0时间戳','t1时间戳','厂商编号','目标ID','类型','经度','纬度','速度','航向角','长度','宽度','高度'])

        rteFile=os.path.join(self.dirpath,'rte.csv')
        if os.path.exists(rteFile):
            self.f_rte=open(rteFile,'a',newline='')
            self.writer_rte=csv.writer(self.f_rte,dialect='excel')
        else:
            self.f_rte=open(rteFile,'w',newline='')
            self.writer_rte=csv.writer(self.f_rte,dialect='excel')
            self.writer_rte.writerow(['帧序号','时间戳','厂商编号','事件ID','事件类型','事件来源','事件位置经度','事件位置纬度','事件开始时间','事件持续时间','影响范围半径','事件置信度','所属车道','涉及感知ID列表'])

        flowFile=os.path.join(self.dirpath,'flow.csv')
        if os.path.exists(flowFile):
            self.f_flow=open(flowFile,'a',newline='')
            self.writer_flow=csv.writer(self.f_flow,dialect='excel')
        else:
            self.f_flow=open(flowFile,'w',newline='')
            self.writer_flow=csv.writer(self.f_flow,dialect='excel')
            self.writer_flow.writerow(['帧序号','时间戳','厂商编号','统计开始时间','统计结束时间','总车流量','车道编号','车流量','车道编号','车流量','车道编号','车流量','车道编号','车流量'])

        self.keyboard_listener = Listener(on_press=self.on_press)
        Thread(target=self.keyboard_listener.start).start()

        self.frame=1
        while self.running:
            self.timestamp=time.time()
            # np.float64: ID,class,Xw,Yw,Vx,Vy，heading
            self.targetRaw = np.copy(np.ctypeslib.as_array(target_Raw))
            self.target_send = self.targetRaw[4:int(4 + self.targetRaw[2])].reshape(-1,7)
            # 厂商、事件类型、事件来源、经度、维度、开始时间、持续时间、作用半径、置信度、所属车道号、涉及感知ID列表
            self.eventRaw = np.copy(np.ctypeslib.as_array(event_raw))
            self.event_send = self.eventRaw[2:int(self.eventRaw[0] + 2)].astype(np.str).reshape(-1, 11)
            # start_time, end_time, sum_volume, line1_volume, line2_volume,line3_volume,line4_volume
            self.flowRaw = np.copy(np.ctypeslib.as_array(flow_raw))
            self.flow_send = self.flowRaw[2:int(self.flowRaw[0] + 2)].astype(np.str).reshape(-1, 7)

            self.traffic_participant()
            self.traffic_event()
            self.traffic_flow()

            self.frame+=1
            time.sleep(0.1)


    def on_press(self, key):
        if key == Key.esc:
            self.running=0
            try:
                self.f_ptc.close()
            except:pass
            try:
                self.f_rte.close()
            except:pass
            try:
                self.f_flow.close()
            except:pass
            self.keyboard_listener.stop()


    def traffic_participant(self):
        ptcList=list()
        for target in self.target_send:
            ptc=[]
            t0=self.targetRaw[0]/1000
            t1=self.targetRaw[1]/1000
            supplier=5
            ptcId=int(target[0])
            # ptcType = 3 if target[1] == 1 else 1
            if target[1] == 1:
                ptcType = 3
            elif target[1] == 2:
                ptcType = 2
            else:
                ptcType = 1
            lon, lat = self.converter(target[2], target[3], inverse=True)
            speed = int(round(np.linalg.norm([target[4:6]]) / 100.0, 4) / 0.02)
            heading = int(target[6] * 8)
            length=self.length_scale[int(target[1] - 1)]
            width = self.width_scale[int(target[1] - 1)]
            height=int(self.height_scale[int(target[1] - 1)] / 5)
            ptc.extend([self.frame,t0,t1,supplier,ptcId,ptcType,lon,lat,speed,heading,length,width,height])
            ptcList.append(ptc)
        if self.target_send.size == 0:
        # if False:
            ptc=[]
            t0=self.targetRaw[0]/1000
            t1=self.targetRaw[1]/1000
            ptcType = 0
            ptcId = 0
            lon = 0
            lat = 0
            speed = 0
            heading = 0
            length = 0
            width = 0
            height = 0
            ptc.extend([self.frame,t0,t1,5,ptcId,ptcType,lon,lat,speed,heading,length,width,height])
            ptcList.append(ptc)
        try:
            self.writer_ptc.writerows(ptcList)
        except: self.writer_ptc.writerow([self.frame])         


    def traffic_event(self):
        rteList = list()
        for event in self.event_send:
            rte = []
            supplier = event[0]
            rteId=random.randint(0,99)
            eventType=f"0{int(float(event[1]))}"
            eventSource=event[2]
            lon=event[3]
            lat=event[4]
            startTime=event[5]
            runningTime=event[6]
            eventRadius=event[7]
            eventConfidence=event[8]
            laneId=event[9]
            fusionId=event[10]
            rte.extend([self.frame,self.timestamp,supplier,rteId,eventType,eventSource,lon,lat,startTime,runningTime,eventRadius,eventConfidence,laneId,fusionId])
            rteList.append(rte)
        try:
            self.writer_rte.writerows(rteList)
        except: self.writer_rte.writerow([self.frame])
            

    def traffic_flow(self):
        flowList=list()
        for flow in self.flow_send:
            volume=[]
            supplier =5
            start_time = flow[0]
            end_time = flow[1]
            sum_volume = flow[2]
            line1_volume = flow[3]
            line2_volume = flow[4]
            line3_volume = flow[5]
            line4_volume = flow[6]
            volume.extend([self.frame,self.timestamp,supplier,start_time,end_time,sum_volume,1,line1_volume,2,line2_volume,3,line3_volume,4,line4_volume])
            flowList.append(volume)
        try:
            self.writer_flow.writerows(flowList)
        except: self.writer_flow.writerow([self.frame])



if __name__ == '__main__':
    print(datetime.datetime.now())
    from multiprocessing import RawArray,Process, Lock, Value
    flag = Value('i', 1)
    # np.float64
    target_Raw = RawArray('d', 1024)
    event_Raw = RawArray('d', 1024)
    flow_Raw = RawArray('d', 1024)
    p1=Process(target=CSV,args=(target_Raw,event_Raw,flow_Raw))
    p1.start()
    lock = Lock()
    p2=Process(target=traffic_event_detect,
            args=(flag, target_Raw, lock, event_Raw, flow_Raw))
    p2.start()
    p1.join()
    p2.join()




