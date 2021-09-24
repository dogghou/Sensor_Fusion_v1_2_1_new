# from funs import send_fusion_data, init_client_mqtt
import numpy as np
import os
from threading import Thread
import paho.mqtt.client as mqtt
import json
import time

# send_client = init_client_RSU('tuanjielu', '10.110.212.7', 1883)
# print(1)

def init_client_mqtt(client_ID, mqtt_IP='127.0.0.1', mqtt_Port=11883):
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

if __name__ == '__main__':
    client = init_client_mqtt('tuanjie')
    pushVideo_dict = dict()
    payload = {
        "inputs": [{
            "name": "pushUri",
            "value": "rtmp://47.105.107.143:1935/live/stream1/EDG-ota0419-10-130-210-30"
        }, {
            "name": "pushTimeMin",
            "value": 30
        }, {
            "name": "cameraIp",
            "value": "10.130.210.30"
        }, {
            "name": "pushType",
            "value": "rtmp"
        }],
        "function": "publishSet",
        "messageId": "1397742272259166208",
        "deviceId": "ota0419"
    }
    inputs = payload['inputs']
    result = {}
    for para in inputs:
        result[para['name']] = para['value']
    result['set'] = 1
    cameraIp = result['cameraIp']
    del result['cameraIp']
    pushVideo_dict[cameraIp] = result
    param = json.dumps(pushVideo_dict, sort_keys=False)
    topic = '/Caeri/Push_Stream'
    while True:
        client.publish(topic, payload=param, qos=0)
        print(param)
        time.sleep(1)
