import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET


def get_XY(filename,area_id):
    tree = ET.parse(filename)
    root = tree.getroot()
    area = dict()
    data = np.array([[1, 5], [1, 18], [18, 18], [18, 5]]).astype(np.float64)
    data1 = data
    element = root.find('area_{}'.format(area_id))
    for child_node in element:
        area[child_node.tag] = child_node.text
    data_x = np.array(area['point_x'][1:-2].split(' ')).reshape(-1, 1)
    data_y = np.array(area['point_y'][1:-2].split(' ')).reshape(-1, 1)
    data = np.column_stack((data_x[data_x != ''], data_y[data_y != '']))
    data1[0, 0], data1[0, 1] = float(data[0, 0]), float(data[0, 1])
    data1[1, 0], data1[1, 1] = float(data[1, 0]), float(data[1, 1])
    data1[2, 0], data1[2, 1] = float(data[2, 0]), float(data[2, 1])
    data1[3, 0], data1[3, 1] = float(data[3, 0]), float(data[3, 1])
    return data1