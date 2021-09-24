import os,sys,subprocess
import time,datetime
import hashlib,random,collections
from multiprocessing import Process
import numpy as np
from threading import Thread
import requests
import json
from jsonpath import jsonpath
from pyproj import Proj
import xml.etree.ElementTree as ET
import paho.mqtt.client as mqtt

class Mqtt:
    def __init__(self,local_list,edg_conf,gpu_conf,mqtt_conf,NetworkRate_dict,SensorOnline_dict,pc_status_dict,target_Raw, event_sent_raw,pushVideo_dict,secureId='test',secureKey='test'):
        self.local_list = local_list
        self.edg_conf = edg_conf
        self.gpu_conf = gpu_conf
        self.mqtt_conf = mqtt_conf
        self.NetworkRate_dict = NetworkRate_dict
        self.SensorOnline_dict = SensorOnline_dict
        self.pc_status_dict = pc_status_dict
        self.target_Raw = target_Raw
        self.event_sent_raw = event_sent_raw
        self.pushVideo_dict = pushVideo_dict
        self.secureId = secureId
        self.secureKey = secureKey

        self.envir = os.getcwd()
        self.converter = Proj(proj='utm', zone=48, ellps='WGS84', south=False, north=True, errcheck=True)
        self.sensor_type_dict = {'camera': 3, 'radar': 4, 'lidar': 6}   # 传感器类型对应的数字
        self.width_scale = [100, 100, 250, 300, 300]
        self.length_scale = [100, 200, 400, 800, 800]
        self.height_scale = [200, 200, 200, 400, 400]
        self.database = {}
        self.child_status = {}
        self.xml_tmp = {}

        # mqtt初始化
        self.edgId = self.edg_conf['id']
        self.client = mqtt.Client(self.edgId)
        # mqtt用户认证
        self.authenticate()
        # mqtt功能部署
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_publish = self.on_publish
        self.client.on_message = self.on_message
        self.client.on_subscribe= self.on_subscribe
        # mqtt连接
        self.app_IP = mqtt_conf['APP_IP']
        self.app_Port = int(mqtt_conf['APP_Port'])
        while not self.client.is_connected():
            try:
                self.client.connect(self.app_IP, self.app_Port,60)
                Thread(target=self.client.loop_start).start()
                # self.client.loop(keepalive=60) # 心跳函数,60s没有发布消息则断开连接
                # self.client.loop_forever() # 事件循环,自动处理掉线重连问题,程序阻塞,重连间隔时间指数级增长
                while not self.client.is_connected():
                    continue
            except:
                print('mqtt APP 连接失败！')
                time.sleep(10)
                
        # 点播推流
        self.pushVideo_client = mqtt.Client('P'+self.edgId)
        self.pushVideo_client.on_connect = self.on_connect
        while not self.pushVideo_client.is_connected():
            if True:
                self.pushVideo_client.connect('127.0.0.1', 11883, 60)
                Thread(target=self.pushVideo_client.loop_start).start()
                while not self.pushVideo_client.is_connected():
                    continue
            else:
                print('本地推流连接失败！')
                time.sleep(10)
        

        t_report=t_pull=time.time()
        while True:
            os.chdir(self.envir)
            t=time.time()
            if t-t_report>10:
                print('-----------------------------------')
                self.send_edg_properties()
                print(f'\n{datetime.datetime.now()}\n-----------------------------------')
                t_report=t
            if t-t_pull>3600:
                print('-----------------------------------')
                topic = f'/EDG/{self.edgId}/firmware/pull'
                payload={
                    'headers':{'force':'false','latest':'true'},
                    'deviceId':self.edgId,
                    'timestamp':int(time.time())*1000,
                    'messageId':random.randint(0, 9999999999),
                    # 'requestVersion':'v1.1.0'
                }
                self.client.publish(topic, payload=json.dumps(payload), qos=0)
                print(f'\n{datetime.datetime.now()}\n-----------------------------------')
                t_pull=t

    def authenticate(self):
        username = f'{self.secureId}|{int(time.time() * 1000)}'
        m = hashlib.md5()
        m.update(f'{username}|{self.secureKey}'.encode())
        password = m.hexdigest()
        self.client.username_pw_set(username, password)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"<{self.edgId}>: Connected at {datetime.datetime.now()}!")

    def on_disconnect(self, client, userdata, rc):
        print(f"<{self.edgId}>: Disconnected at {datetime.datetime.now()}")
        self.authenticate()

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("On Subscribed: qos = %d" % granted_qos)

    def on_message(self, client, userdata, msg):
        print('--------------------------------------')
        print(f"time:\n    {datetime.datetime.now()}")
        print(f"topic:\n    {msg.topic}")
        print(f"info:\n    {msg.payload.decode('utf-8')}")
        print('--------------------------------------')
        topic = msg.topic
        payload = json.loads(msg.payload.decode('utf-8'))
        reply = collections.OrderedDict()
        reply['timestamp'] = int(time.time() * 1000)
        reply['messageId'] = payload['messageId']
        deviceId = payload['deviceId']

        ### 读取设备属性
        if '/properties/read' in topic:
            try:
                properties = payload['properties']
                command = collections.OrderedDict()
                for p in properties:
                    command[p] = self.database[deviceId]['properties'][p]
                reply['properties'] = command
                reply['success'] = 'true'
            except:
                reply['success'] = 'false'
                error = sys.exc_info()
                reply['code'] = str(error[0]).split("'")[1]
                reply['message'] = error[1]
            client.publish(topic+'/reply', payload=json.dumps(reply), qos=0)

        ### 修改设备属性
        elif '/properties/write' in topic:
            for file in os.listdir('config/'):
                filepath = os.path.join('config',file)
                tree = ET.parse(filepath)
                root = tree.getroot()
                if file == 'system.xml':
                    s = root.find('edg')
                else:
                    filename, tree, s = self.xml_tmp[deviceId]
                if s.find('id').text == deviceId:
                    try:
                        properties = payload['properties']
                        for item in list(properties.items()):
                            s.find(item[0]).text = item[1]
                        tree.write(filepath)
                        reply['success'] = 'true'
                    except:
                        reply['success'] = 'false'
                    break
            client.publish(topic+'/reply', payload=json.dumps(reply), qos=0)

        ### 调用功能
        elif '/function/invoke' in topic:
            # 重启MEC
            if payload['function'] == 'reboot':
                reply['output'] = 'success'
                reply['success'] = 'true'
                client.publish(topic+'/reply', payload=json.dumps(reply), qos=0)
                # os.popen("echo '123456' | sudo -S reboot")
            # 推流设置
            elif payload['function'] == 'publishSet':
                inputs=payload['inputs']
                result={}
                for para in inputs:
                    result[para['name']]=para['value']
                result['set']=1
                cameraIp=result['cameraIp']
                del result['cameraIp']
                self.pushVideo_dict[cameraIp]=result
                print(self.pushVideo_dict)
                self.pushVideo_client.publish('/Caeri/Push_Stream', payload=json.dumps(dict(self.pushVideo_dict)), qos=0)
                reply['output'] = payload['inputs']
                reply['success'] = 'true'
                client.publish(topic + '/reply', payload=json.dumps(reply), qos=0)
            # 停止推流
            elif payload['function'] == 'publishStop':
                inputs = payload['inputs']
                result = {}
                for para in inputs:
                    result[para['name']] = para['value']
                result['set'] = 0
                cameraIp = result['cameraIp']
                del result['cameraIp']
                self.pushVideo_dict[cameraIp] = result
                self.pushVideo_client.publish('/Caeri/Push_Stream', payload=json.dumps(dict(self.pushVideo_dict)), qos=0)
                reply['output'] = payload['inputs']
                reply['success'] = 'true'
                client.publish(topic + '/reply', payload=json.dumps(reply), qos=0)

        ### 远程升级
        elif '/firmware/pull/reply' in topic or '/firmware/upgrade' in topic:
            iotVersion=payload.get('version')
            url=payload.get('url')
            signMethod=payload.get('signMethod')
            sign=payload.get('sign')
            self.upgrade_process('false', 100, 'true', iotVersion, '升级失败，升级包信息不全！')
            return
            if iotVersion and url and signMethod and sign:
                if iotVersion[0] in ['v', 'V']:
                    iotVersion = iotVersion[1:]
                # url = url.replace('10.130.210.67','jetlinks-cqxdq.i-vista.org:11891')
                self.automatic_update(iotVersion,url,signMethod,sign)
            else:
                self.upgrade_process('false', 100, 'true', iotVersion, '升级失败，升级包信息不全！')

    def on_publish(self, client, userdata, mid):
        print('published!')

    # EDG属性上报
    def send_edg_properties(self):
        ###---------------------edg属性上报----------------------------start
        edg_payload = collections.OrderedDict()
        edg_payload['timestamp'] = int(time.time() * 1000) # 毫秒时间戳
        edg_payload['messageId'] = str(random.randint(0, 9999999999))
        properties = collections.OrderedDict()
        properties['id'] = self.edg_conf['id']      # EDG设备编号
        properties['name'] = self.edg_conf['name']      # EDG设备名称
        properties['sn'] = self.edg_conf['sn']      # EDG序列号
        properties['supplier'] = self.edg_conf['supplier']      # EDG供应商
        properties['softwareVersion'] = self.edg_conf['softwareVersion']    # EDG软件版本号
        properties['hardwareVersion'] = self.edg_conf['hardwareVersion']    # EDG硬件版本号
        properties['webVersion'] = self.edg_conf['webVersion']     # EDG近端维护硬件版本号
        # properties['temperature'] = self.pc_status_dict['Temperature']      # CPU(最大)工作温度
        # properties['cpuUsage'] = self.pc_status_dict['Memory']      # CPU占用率
        # properties['memUsage'] = self.pc_status_dict['Swap']        # 内存占用率
        # properties['diskUsage'] = self.pc_status_dict['Disk']['percent']    # 磁盘使用率

        sensor_list, online_list = [], []
        camera_list, radar_list, lidar_list = [], [], []
        for local in self.local_list:
            filename = os.path.join('config',local)
            tree = ET.parse(filename)
            root = tree.getroot()
            sensor = root.findall('Sensor')
            for s in sensor:
                type = s.get('type')
                ip = s.get('ip')
                sensorId = s.find('id').text
                self.xml_tmp[sensorId] = (filename, tree, s)
                sensor_list.append(ip)
                eval(f'{type}_list.append("{ip}")')
                # ping ...... >/dev/null 2>&1默认不输出结果
                # code,result=subprocess.getstatusoutput('ping -c 1 %s'%ip) # ping不通時程序会阻塞
                # if self.SensorOnline_dict.get(ip) == '断开':
                #     ### 子设备离线上报
                #     if self.child_status.get(ip) != 0:
                #         payload = {"deviceId": sensorId,
                #                    "messageId": random.randint(0, 9999999999),
                #                    "timestamp": int(time.time() * 1000)}
                #         topic = f'/EDG/{self.edgId}/child/{sensorId}/disconnect'
                #         self.client.publish(topic, payload=json.dumps(payload), qos=0)
                #         self.child_status[ip] = 0
                # else:
                #     online_list.append(ip)
                #     ### 子设备上线上报
                #     if not self.child_status.get(ip):
                #         payload = {"deviceId": sensorId,
                #                    "messageId": random.randint(0, 9999999999),
                #                    "timestamp": int(time.time() * 1000)}
                #         topic = f'/EDG/{self.edgId}/child/{sensorId}/connected'
                #         self.client.publish(topic, payload=json.dumps(payload), qos=0)
                #         self.child_status[ip] = 1
                #
                #     ### 子设备(Radar/Camera/Lidar)状态上报
                #     self.send_sensor_properties(s)

        properties['onlineRate'] = f'{len(online_list)}/{len(sensor_list)}'     # 子设备在线率
        properties['cameraList'] = camera_list
        properties['radarList'] = radar_list
        properties['lidarList'] = lidar_list
        # properties['uptime'] = os.popen('uptime').readline()
        # 应用详情
        properties['applications'] = [{"id": "000001",
                                       "type": 1,
                                       "name": self.edg_conf['name']+'融合感知',
                                       "state": 1,
                                       "startupTimes": '6/M'
                                       },
                                      {"id": "000002",
                                       "type": 2,
                                       "name": self.edg_conf['name']+'流量检测',
                                       "state": 0,
                                       "startupTimes": '1/M'
                                       }]

        # 通信参数
        protocol = 'TCPClient'  # mqtt
        app_ip = self.mqtt_conf['APP_IP']
        app_port = self.mqtt_conf['APP_Port']
        edg_ip = self.edg_conf['host_IP']
        properties['commParam'] = f'{protocol},{app_ip},{app_port},{edg_ip},80'

        # 订阅信息表
        properties['rss'] = []
        edg_payload['properties'] = properties
        self.database[self.edgId] = edg_payload
        edg_topic = f'/EDG/{self.edgId}/properties/report'
        # print(json.dumps(payload, indent=2))
        self.client.publish(edg_topic, payload=json.dumps(edg_payload), qos=0)
        ###---------------------edg属性上报-----------------------------end

        ###---------------------rsm事件上报----------------------------start
        rsm_topic = f'/EDG/{self.edgId}/event/rsm'
        rsm_payload = collections.OrderedDict()
        rsm_payload['timestamp'] = int(time.time() * 1000)
        rsm_payload['messageId']=str(random.randint(0, 9999999999))
        data=collections.OrderedDict()
        data['appId']= '000001'
        participants=[]
        self.target_raw = np.copy(np.ctypeslib.as_array(self.target_Raw))
        self.target_send=self.target_raw[2:int(2 + self.target_raw[0])].reshape(-1, 7)  # np.float64: ID,class,Xw,Yw,Vx,Vy，heading
        for target in self.target_send:
            participant = collections.OrderedDict()
            participant['ptcType']=3 if target[1] == 1 else 1
            participant['ptcId']=int(target[0])
            participant['source']=4
            participant['id']=int(target[0])
            participant['secMark']= 6550 #secMark
            lon, lat = self.converter(target[2],target[3],inverse=True)
            participant['pos']= {"offsetLL":{"lon": int(lon * 10000000),"lat": int(lat * 10000000)},
                                 "offsetV":{"elevation": 6777}} #elevation
            participant['posConfidence']= {"pos": 0,"elevation": 0} #posConfidence置信度
            participant['transmission']= 2 #transmission
            participant['speed']= int(round(np.linalg.norm([target[4:6]]) / 100.0, 4) / 0.02)
            participant['heading']= int(target[6]*8)
            participant['angle']= 127 #angle车轮角度
            participant['motionCfd']= {"speedCfd": 0,"headingCfd": 0,"steerCfd": 0} #motionCfd
            participant['accelSet']= {"long": 2001,"lat": 2001,"vert": -127,"yaw": 0} #accelSet
            participant['size']= {"width": self.width_scale[int(target[1] - 1)],
                                  "length": self.length_scale[int(target[1] - 1)],
                                  "height": int(self.height_scale[int(target[1] - 1)] / 5)}
            participant['vehicleClass']= {"classification": 60} #classification
            participants.append(participant)
        data['participants']=participants
        rsm_payload['data']=data
        self.client.publish(rsm_topic, json.dumps(rsm_payload), qos=0)
        ###---------------------rsm事件上报------------------------------end

       

    # 子设备属性上报
    def send_sensor_properties(self, s):
        sensorId = s.find('id').text
        protocol = 'UDP'
        sensor_ip = s.get('ip')
        sensor_port = s.find('port').text
        app_ip = self.edg_conf['host_IP']
        sensor_payload = collections.OrderedDict()
        sensor_payload['messageId'] = random.randint(0, 9999999999)
        properties = collections.OrderedDict()
        properties['id'] = sensorId
        properties['name'] = s.find('name').text
        properties['type'] = self.sensor_type_dict[s.get('type')]   # 子设备类型
        properties['supplier'] = s.find('supplier').text
        # 检测范围
        detect_area = s.find('detect_area')
        if s.get('type') == 'radar':
            xmin = detect_area.find('xmin').text
            xmax = detect_area.find('xmax').text
            ymin = detect_area.find('ymin').text
            ymax = detect_area.find('ymax').text
            properties['detectionRange'] = {'x': int(xmax) - int(xmin), 'y': int(ymax) - int(ymin)}
        elif s.get('type') == 'camera':
            properties['detectionRange'] = {'x': int(detect_area.find('x_offset').text),
                                            'y': int(detect_area.find('y_offset').text)}
        elif s.get('type') == 'lidar':
            pass
        properties['trafficStatistics'] = f"{self.NetworkRate_dict[sensor_ip]}M/s"  # 流量统计
        if s.get('type') == 'camera':
            properties['resolution'] = '1080x1920'
        properties['commParam'] = f'{protocol},{sensor_ip},{sensor_port},{app_ip},80'
        sensor_payload['properties'] = properties
        self.database[sensorId] = sensor_payload
        sensor_topic = f"/EDG/{self.edgId}/child/{sensorId}/properties/report"
        self.client.publish(sensor_topic, payload=json.dumps(sensor_payload), qos=0)

    def automatic_update(self,iotVersion,url,signMethod,sign):
        tree = ET.parse('config/system.xml')
        self.softwareVersion = tree.getroot().find('edg').find('softwareVersion').text
        if self.softwareVersion > iotVersion:
            self.upload()
        elif self.softwareVersion < iotVersion:
            self.download(iotVersion,url,signMethod,sign)
        else:
            self.upgrade_process('false',100,'true',iotVersion,'当前软件包已是最新版本!')

        # self.delete_tmp()

    # 上传
    def upload(self):
        # os.chdir(os.environ['HOME'])
        # versionName=os.popen("echo $(ls -l Sensor_Fusion_x64) | awk '{print $11}'").readline().strip()
        # versionZip=versionName+'.zip'
        # zip=subprocess.Popen(f"zip -rq {versionZip} {versionName} -x '*config*' '*location_H' '*map*'",shell=True)
        # if zip.wait():
        #    print('设备端软件包压缩失败')
        #    return

        token = self.get_token()
        # 上传软件包
        package='CoLSrGCwudWAZ2MeAAAvxelXJs455.xlsx'
        # size=os.popen("echo $(ls -l %s) | awk '{print $5}'"%package).readline().strip()
        files = {'file': ('Sensor_Fusion_x64', open(package, 'rb'), 'application/zip')}
        headers={'X-Access-Token': token}
        # response = requests.post(url='http://jetlinks-cqxdq.i-vista.org:11891/jetlinks/file/static', headers=headers,files=files).json()
        response = requests.post(url='http://10.130.210.87:8080/jetlinks/file/static', headers=headers,files=files).json()

        # 新建固件
        url = jsonpath(response, '$..result')[0]
        # url = url.replace('10.130.210.67','jetlinks-cqxdq.i-vista.org:11891')
        data = {
            "productId":'EDG', #产品ID
            "productName":'EDG路测单元', #产品名称
            "name":'Sensor_Fusion_x64', #固件名称
            "version":self.softwareVersion, #版本号(要求0.0.0)
            "versionOrder": self.softwareVersion.replace('.',''), #版本序号
            "url":url, #固件文件地址
            "sign":self.file_md5(package), #固件文件签名
            "signMethod":'MD5', #固件文件签名方式，如MD5,SHA256
            "size":4096,
            # "size":int(size), #固件文件大小
        }
        headers = {'Content-Type':'application/json;charset=UTF-8','X-Access-Token': token}
        # response = requests.post(url='http://jetlinks-cqxdq.i-vista.org:11891/jetlinks/firmware',data=json.dumps(data), headers=headers).json()
        requests.post(url='http://10.130.210.87:8080/jetlinks/firmware',data=json.dumps(data), headers=headers).json()
        if jsonpath(response,'$..code')[0] == 'success':
            self.upgrade_process('true', 100, 'true', self.softwareVersion, '上传成功!')
        else:
            self.upgrade_process('false', 100, 'true', self.softwareVersion, '上传失败!')

    def get_token(self):
        data = {
            "username": "admin",
            "password": "admin",
            "expires": -1,
            "tokenType": "default",
            "verifyKey": "",
            "verifyCode": ""
        }
        headers = {'Content-Type':'application/json;charset=UTF-8'}
        # response = requests.post('http://jetlinks-cqxdq.i-vista.org:11891/jetlinks/authorize/login', data=json.dumps(data),headers=headers).json()
        response = requests.post('http://10.130.210.87:8080/jetlinks/authorize/login', data=json.dumps(data),headers=headers).json()
        return jsonpath(response, '$..token')[0]

    # 下载
    def download(self,iotVersion,url,signMethod,sign):
        package=os.path.basename(url)
        # os.chdir(os.environ['HOME'])
        try:
            download = requests.get(url, stream=True)
            f = open(package, 'wb')
            for chunk in download.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
            f.close()
            self.upgrade_process('true',50,'true',iotVersion,'成功下载升级包!')
        except:
            self.upgrade_process('false', 100, 'true', iotVersion, '升级包下载失败!')
            return

        if signMethod == 'MD5':
            sign_new=self.file_md5(package)
            if sign_new == sign:
                self.upgrade_process('true',60,'false',iotVersion,'MD5校验成功!')
            else:
                self.upgrade_process('false',100,'true',iotVersion,'MD5校验失败!')
                return
        else:
            self.upgrade_process('false',100,'true',iotVersion,'暂不支持MD5以外的校验方式!')

        #     # 旧版本文件迁移
        #     for dir in ['config','location_H','map']:
        #         p=subprocess.Popen(f"cp -rf Sensor_Fusion_x64/{dir} {versionName}/",shell=True)
        #         if not p.wait():
        #             continue
        #
        #     p=subprocess.Popen(f"unlink Sensor_Fusion_x64 && ln -s {versionName} Sensor_Fusion_x64",shell=True)
        #     if not p.wait():
        #         os.chdir(f'../{versionName}')
        #         # 同步system.xml版本号
        #         tree = ET.parse('config/system.xml')
        #         root = tree.getroot()
        #         root.find('edg').find('softwareVersion').text = iotVersion
        #         tree.write('config/system.xml')
        #         # 编译
        #         try:
        #             os.popen('sh core/make.sh')
        #         except:pass
        #         self.upgrade_process('true',100,'true',iotVersion,'升级成功')
        #         print('true',100,'true',iotVersion,'升级成功')
        #     else:
        #         self.upgrade_process('false',100,'true',iotVersion,'升级失败，本地升级未通过')
        #         print('false',100,'true',iotVersion,'升级失败，本地升级未通过')
        # else:
        #     self.upgrade_process('false',100,'true',iotVersion,'升级失败，本地已存在最新软件包')
        #     print('false',100,'true',iotVersion,'升级失败，本地已存在最新软件包')
        # self.upgrade_process('false',100,'true',iotVersion,'升级失败')

    def file_md5(self,file):
        try:
            f = open(file, 'rb')
            m = hashlib.md5()
            while True:
                data = f.read(1024)
                if not data:
                    break
                m.update(data)
            f.close()
            return m.hexdigest()
        except:pass

    # 升级进度上报
    def upgrade_process(self,result,progress,complete,iotVersion,detail):
        # topic = f'/EDG/{self.edgId}/firmware/upgrade/progress'
        topic = f'/EDG/{self.edgId}/firmware/upgrade/progress'
        payload = {'success':result,
                   'message': detail, # 测试无效
                   'progress':progress,
                   'complete':complete,
                   'deviceId':self.edgId,
                   'version':iotVersion,
                   'timestamp':int(time.time())*1000,
                   }
        self.client.publish(topic, json.dumps(payload), qos=0)

    # 删除临时文件
    def delete_tmp(self):
        for f in os.listdir(os.environ['HOME']):
            if 'Sensor_Fusion' in f and '.zip' in f:
                os.remove(os.path.join(os.environ['HOME'],f))

import cv2
class playVideoOnDemand:
    def __init__(self,pushVideo_dict):
        self.pushVideo_dict=pushVideo_dict
        self.flag_dict={}
        self.openVideo=[]
        while True:
            try:
                for cameraIp,para in list(self.pushVideo_dict.items()):
                    if para.get('set')==1:
                        if cameraIp in self.openVideo:
                            break
                        self.flag_dict[cameraIp]=1
                        self.openVideo.append(cameraIp)
                        pushType = para.get('pushType')
                        pushUri = para.get('pushUri')
                        try:
                            pushTimeMin = int(para.get('pushTimeMin'))
                        except:
                            pushTimeMin=10
                        if cameraIp and pushUri:
                            Thread(target=lambda:self.push_stream(cameraIp,pushType,pushUri,pushTimeMin)).start()
                    else:
                        self.flag_dict[cameraIp]=0
                        del self.pushVideo_dict[cameraIp]
            except:pass

    def push_stream(self,cameraIp,pushType,pushUri,pushTimeMin):
        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', "{}x{}".format(960, 540),
                   # '-r', str(fps),
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'ultrafast',
                   '-f', 'flv',
                   pushUri]
        # 管道配置
        p = subprocess.Popen(command, stdin=subprocess.PIPE)
        cap = cv2.VideoCapture('rtsp://admin:ehl1234.@{}:554/h264/ch1/main/av_stream'.format('10.130.210.30'))
        del self.pushVideo_dict[cameraIp]
        t0 = time.time()
        while self.flag_dict[cameraIp]:
            t = time.time()
            if t - t0 < int(pushTimeMin) * 60:
                ret, image = cap.read()
                if ret:
                    try:
                        img = cv2.resize(image, (960, 540))
                        p.stdin.write(img.tostring())
                    except:
                        continue
            else:
                break
        self.openVideo.remove(cameraIp)

if __name__ == '__main__':
    from multiprocessing import RawArray, Manager
    from config_operate import load_config, load_sys_config

    NetworkRate_dict = Manager().dict({'Total':0,'202.108.22.5':0})
    NetworkCount_dict = Manager().dict({'Total':0,'202.108.22.5':0})
    SensorOnline_dict = Manager().dict()
    pushVideo_dict = Manager().dict()
    pc_status_dict = Manager().dict({'CPU': None, 'Temperature': None, 'Memory': None, 'Swap': None, 'Disk': {'percent': None}})
    event_sent_raw = RawArray('d', 1024)  ## 交通事件数据

    keyword = 'tuanjielu'
    target_Raw = RawArray('d', 1024)  # np.float64
    edg_conf, gpu_conf, mqtt_conf = load_sys_config()

    for file_name in os.listdir("location_H"):
        if keyword.split('_')[0] in file_name:
            location = file_name
    local_list = []
    ip_list = []
    for file_name in os.listdir('config'):
        if location in file_name:
            local_list.append(file_name)
            IP_dict = load_config(file_name)
            for value in list(IP_dict.values()):
                ip_list.extend(value)

    p_mqtt=Process(target=Mqtt,args=(local_list,edg_conf,gpu_conf,mqtt_conf,NetworkRate_dict,SensorOnline_dict,pc_status_dict,target_Raw, event_sent_raw,pushVideo_dict,))
    p_pushVideo=Process(target=playVideoOnDemand,args=(pushVideo_dict,))
    p_mqtt.start()
    p_pushVideo.start()
    p_mqtt.join()
    p_pushVideo.join()

