# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:28:09 2019

@author: Administrator
"""

import cv2, math, re
import numpy as np


filename = 'B11'
filepath = 'D:/location6_south/12.5/Radar/' + filename + '.txt'

with open(filepath , 'r') as fr:
    Lines = fr.readlines()
fr.close()

Radar_storage = list()
for i in range(len(Lines)):
    if Lines[i][0:4] == 'Time':
        radar_data = np.empty((0,4),np.int32);
    elif Lines[i] == '\n':         
        Radar_storage.append(radar_data);
    else:
        line = re.split(', ', Lines[i].strip());
        data = np.reshape(np.array(list(map(int, line))), (1,4))
        radar_data = np.vstack((radar_data, data))

radar_data = np.empty((0,4),np.int32);
for i in range(len(Radar_storage)):
    if Radar_storage[i].shape[0] == 1:
        radar_data = np.vstack((radar_data, Radar_storage[i]))

Pr = np.round(np.mean(radar_data, axis=0)).astype('int32')

print (filename)
print ((Pr[0], Pr[1]))
        
    