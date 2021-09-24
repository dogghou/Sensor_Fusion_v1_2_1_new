# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:38:06 2019

@author: Administrator
"""

import numpy as np
import ctypes
lib2 = ctypes.cdll.LoadLibrary('WGS84.dll')

def gpgga2gps(x):
    y = np.zeros(2)
    for i in range(2):
        k = 1
        y[i] = x[i]
        if y[i] < 0:
            y[i] = -y[i]
            k = -1
        y[i] = k * (y[i]//100 + (y[i]%100)/60)
    return y

def W842UTM(P0): # P0为某点的(lat,lon)
    lat = ctypes.c_double(P0[0]);
    lon = ctypes.c_double(P0[1]);
    doubleArray = ctypes.c_double*2
    result = doubleArray()
    lib2.LatLonToUTM(lat, lon, result)
    return np.round([result[0], result[1]],3)

def find_location(filepath): # 从‘filepath’路径下找到目标点的UTM坐标
    f = open(filepath)
    gpgga = np.empty((0,2), np.float)
    UTM = np.empty((0,2), np.float)
    Line = f.readline()
    while Line!='':
        # print (Line)
        line = Line.split(',') # 按逗号分隔提取字符串
        if line[0] == '$GPGGA' and line[6] == '4':
            gps = gpgga2gps([float(line[2]),float(line[4])])
            gpgga = np.vstack((gpgga, gps));
            UTM = np.vstack((UTM, W842UTM(gps)))
        Line = f.readline()
    f.close()
           
    Num_limit = int(gpgga.shape[0] * 0.75) # PTNL中至少保留的数据量
    max_distance_UTM = 1
    sigma3 = 0

    while UTM.shape[0] > Num_limit and max_distance_UTM > sigma3:
        mean_UTM = np.mean(UTM,axis=0)
        # mean_gpgga = np.mean(gpgga,axis=0)
        error_UTM = UTM-mean_UTM
        # error_gpgga = gpgga-mean_gpgga
        distance_UTM = np.empty([0,1])
        # distance_gpgga = np.empty([0,1])
        for i in range(UTM.shape[0]):
            tmp = np.dot(error_UTM[i,:],error_UTM[i,:].T) ** 0.5
            distance_UTM = np.vstack((distance_UTM,tmp))  # 每个点到中心点的距离 
        sigma3 = 3 * np.std(distance_UTM) # 3倍西格玛准则，
        tmp = [x for x in range(UTM.shape[0]) if distance_UTM[x,:]<=sigma3 and x < len(gpgga)]
        if len(tmp) < 0.5 * Num_limit:
            break
        UTM = np.array(UTM[tmp,:])
        gpgga = np.array(gpgga[tmp,:])
        distance_UTM = np.array(distance_UTM[tmp,:])
        max_distance_UTM = np.max(distance_UTM)
        sigma3 = 3 * np.std(distance_UTM) # 3倍西格玛准则，
    return (np.mean(UTM,axis=0),np.mean(gpgga,axis=0))
    
if __name__ == '__main__':
    resultpath = 'D:/location10/12.7/'
    filedata = 'RTK/'
    filename = '1.log'
    filepath = resultpath + filedata + filename
    
    P0 = find_location(filepath);
    P0_UTM = np.round(P0[0],3)
    P0_gpgga = np.round(P0[1],9)
    print (filename)
    print (tuple(P0_UTM.tolist()));
    print (tuple(P0_gpgga.tolist()));