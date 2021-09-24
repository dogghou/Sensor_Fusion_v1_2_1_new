import paho.mqtt.client as mqtt
import json

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))


def on_message(client, userdata, msg):
    a = json.loads(msg.payload.decode())
    print(msg.topic + " " + str(msg.payload))


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
# client.connect("127.0.0.1", 11883, 60)
client.connect("127.0.0.1", 11883, 60)
topic = '/CQXDQ1/v1/EDG010500007/DAFU/CNODE0000007/1'
topic = '/Caeri/Push_Stream'
client.subscribe(topic, qos=0)
# client.subscribe('/fengxianlu', qos=0)
# topic=/CQXDQ1/v1/EDG010500000/DAFU/CNODE0000002/1
# topic=/CQXDQ1/v1/EDG010500000/DAFU/CNODE0000002/1
client.loop_forever()
