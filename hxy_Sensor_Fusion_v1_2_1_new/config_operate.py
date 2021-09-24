import xml.etree.ElementTree as ET
import os


def load_config(location, ip=None):
    result = dict()
    IP_dict = {'radar': [], 'camera': [], 'lidar': []}
    for file in os.listdir('./config'):
        if location in file:
            ip_dict = {'radar': [], 'camera': [], 'lidar': []}
            filename = './config/{}'.format(file)
            tree = ET.parse(filename)
            root = tree.getroot()
            result['Location'] = root.get('name')
            if ip:
                result['IP'] = ip
            for node in root:
                if node.tag == 'Sensor':
                    ip_dict[node.get('type')].append(node.get('ip'))
                    if node.get('ip') != ip:
                        continue
                    for child_node in node:
                        if len(child_node) == 0:
                            result[child_node.tag] = child_node.text
                        else:
                            result['type'] = node.get('type')
                            for i in range(len(child_node)): result[child_node[i].tag] = child_node[i].text
            for k, v in ip_dict.items():
                IP_dict.setdefault(k, []).extend(v)
    if not ip:
        return IP_dict
    else:
        return result


def load_sys_config(string=''):
    edg_conf, gpu_conf, mqtt_conf = dict(), dict(), dict()
    filename = './config/system.xml'
    tree = ET.parse(filename)
    root = tree.getroot()
    for node in root:
        if node.tag == "edg":
            Dict = edg_conf
        elif node.tag == "gpu":
            Dict = gpu_conf
        elif node.tag == "mqtt":
            Dict = mqtt_conf
        element = root.find(node.tag)
        for child_node in element:
            Dict[child_node.tag] = child_node.text
    if 'edg' in string:
        return edg_conf
    elif 'gpu' in string:
        return gpu_conf
    elif 'mqtt' in string:
        return mqtt_conf
    else:
        return (edg_conf, gpu_conf, mqtt_conf)


if __name__ == '__main__':
    import numpy as np
    a = np.array([[2, 10, 6, 5, 1],
                 [2, 9, 8, 5, 0],
                 [2, 10, 6, 3, 1]])
    aa = a[:, 2::-1]
    aaa = np.where(a[:, 4] > 0)
    aa = a[np.where(a[:, 4] > 0)[0], :]
    b = a[np.lexsort(a[:, 2::-1].T)]
    result = load_config('lrlmX', "172.16.11.117")
    result2 = load_config('lqy')
    edg, gpu, mqtt = load_sys_config()

