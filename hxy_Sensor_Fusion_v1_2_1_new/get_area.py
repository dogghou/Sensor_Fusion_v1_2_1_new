import matplotlib.pyplot as plt
import numpy as np
import xml.dom.minidom
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement


def draw_plt_event(area, color):
    plt.plot([area[0][0], area[1][0]], [area[0][1], area[1][1]], color=color)
    plt.plot([area[1][0], area[2][0]], [area[1][1], area[2][1]], color=color)
    plt.plot([area[2][0], area[3][0]], [area[2][1], area[3][1]], color=color)
    plt.plot([area[3][0], area[0][0]], [area[3][1], area[0][1]], color=color)


def creat_xml(area, num, location):
    book = ElementTree()
    purOrder = Element(location)
    book._setroot(purOrder)
    item = Element("item{}".format(num))
    SubElement(item, "x").text = area[:, 0]
    SubElement(item, "y").text = area[:, 1]
    purOrder.append(item)


def CreateXml(area, num, location):
    book = ElementTree()
    purOrder = Element(location)
    book._setroot(purOrder)
    for i in range(num):
        # area2 = np.array(area)
        data = np.array(area)[i, :].reshape(-1, 2)
        item = Element("area_{}".format(i))
        SubElement(item, "point_x").text = str(data[:, 0])
        SubElement(item, "point_y").text = str(data[:, 1])
        purOrder.append(item)

    indent(purOrder)
    return book


def indent(elem, level=0):
    i = "\n" + level * "  "
    # print(elem)
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            print(e)
            indent(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem


def draw_plt_event(area, color):
    plt.plot([area[0][0], area[1][0]], [area[0][1], area[1][1]], color=color)
    plt.plot([area[1][0], area[2][0]], [area[1][1], area[2][1]], color=color)
    plt.plot([area[2][0], area[3][0]], [area[2][1], area[3][1]], color=color)
    plt.plot([area[3][0], area[0][0]], [area[3][1], area[0][1]], color=color)


if __name__ == '__main__':
    fig = plt.Figure()
    intersection_list = ['jylh', 'lrlm', 'hxj','cdcs']
    intersection = intersection_list[3]
    scale = 'X'
    direction = '1'
    location = intersection + scale + '_' + direction
    location=intersection
    PATH = r'./Map/{}.txt'.format(location)  # 地图地址
    file_lines = open(PATH).readlines()
    x_list, y_list = [], []
    for line in file_lines:
        data = np.array(line.split(', '))
#        print(data)
#        break
        lat = float(data[0])
        lon = float(data[1])
        x_list.append(lat)
        y_list.append(lon)
    plt.scatter(x_list, y_list, s=10)
    color_list = ['b', 'g', 'r', 'c']
    area_list = list()
    num = 4  ##创建区域数
    for i in range(num):
        pos = plt.ginput(4)
        print(pos)
        # draw_plt_event(np.array(pos), color_list[i])
        area_list.append(pos)

    filename = "detect_area.xml"
    book = CreateXml(area_list, num, location)
    book.write(filename, "utf-8")

    # area = dict()
    # tree = ET.parse(filename)
    # root = tree.getroot()
    #
    # data = np.array([[1, 5], [1, 18], [18, 18], [18, 5]]).astype(np.float64)
    # data1 = data
    # # print(data)
    # # for node in root:
    # # if node.tag == 'area_1':
    # #     Dict = area
    # for i in range(num):
    #     element = root.find('area_{}'.format(i))
    #     for child_node in element:
    #         area[child_node.tag] = child_node.text
    #     data_x = np.array(area['point_x'][1:-2].split(' ')).reshape(-1, 1)
    #     data_y = np.array(area['point_y'][1:-2].split(' ')).reshape(-1, 1)
    #     data = np.column_stack((data_x[data_x != ''], data_y[data_y != '']))
    #     data1[0, 0], data1[0, 1] = float(data[0, 0]), float(data[0, 1])
    #     data1[1, 0], data1[1, 1] = float(data[1, 0]), float(data[1, 1])
    #     data1[2, 0], data1[2, 1] = float(data[2, 0]), float(data[2, 1])
    #     data1[3, 0], data1[3, 1] = float(data[3, 0]), float(data[3, 1])
    #     # data1[1, 0:2] = data[1, 0:2]
    #     # data1[2, 0:2] = data[2, 0:2]
    #     # data1[3, 0:2] = data[3, 0:2]
    #     print(data1)
    #     draw_plt_event(data1, 'b')
    # plt.show()
