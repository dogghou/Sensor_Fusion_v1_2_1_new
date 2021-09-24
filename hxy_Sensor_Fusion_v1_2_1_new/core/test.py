# -*- coding: utf-8 -*-
"""
Created on Wed May 19 21:17:38 2021

@author: Administrator
"""

import numpy as np
from core.core_process2 import *

radar_output = np.array([[137001, 3, 36, 1458, 250, 160, 620],
                         [137002, 3, -248, 2120, 150, 160, 620]]).astype(np.int32)


camera_output = np.array([[127001, 4, 456, 1618, 210, 170, 620],
                          [127002, 3, -848, 4120, 110, 170, 620]]).astype(np.int32)


ID_form = -np.ones((7,6), np.int32) # ID, cameraID(s1), radar/lidarID(s2), class, heading, counter
ID_form[:,0] = np.arange(1, ID_form.shape[0]+1)
# ID_form[:,4] = 3600

ID_form, target_state = data_fusion(ID_form, camera_output, radar_output)
print (ID_form)
print (target_state)

radar_output = np.array([[137001, 3, 36, 1458, 250, 160, 620], # 同上
                         [137002, 3, -248, 2120, 150, 160, 620]]).astype(np.int32) # 同上

camera_output = np.array([[127001, 4, 456, 1618, 210, 170, 620], # 同上
                          [127002, 3, -848, 4120, 110, 170, 620], # 同上
                          [127011, 2, 200, 2560, 123, 170, 620]]).astype(np.int32) # 新目标
#####
radar_output = np.array([[137001, 3, 36, 1458, 250, 160, 620],  # 同上
                         [137002, 3, -248, 2120, 150, 160, 620], # 同上
                         [137007, 3, 600, 5955, 121, 121, 620]]).astype(np.int32) # 新目标

camera_output = np.array([[127001, 4, 456, 1618, 210, 170, 620],  # 同上
                          [127002, 3, -848, 4120, 110, 170, 620],  # 同上
                          [127011, 2, 200, 2560, 123, 170, 620]]).astype(np.int32)  # 同上
#####
radar_output = np.array([[137001, 3, 36, 1458, 250, 160, 620], # 同上
                         [137002, 3, -248, 2120, 150, 160, 620]]).astype(np.int32) # 同上

camera_output = np.array([[127001, 4, -240, 2118, 170, 170, 620],  # 更新
                          [127002, 3, -848, 4120, 110, 170, 620],  # 同上
                          [127011, 2, 200, 2560, 123, 170, 620]]).astype(np.int32)  # 同上
###
radar_output = np.array([[137001, 3, 200, 2458, 125, 160, 620],  # 更新
                         [137002, 3, -248, 2120, 150, 160, 620]]).astype(np.int32) # 同上

camera_output = np.array([[127001, 4, -240, 2118, 170, 170, 620], # 同上
                          [127002, 3, -848, 4120, 110, 170, 620], # 同上
                          [127011, 2, 200, 2560, 123, 170, 620]]).astype(np.int32) # 同上

###
radar_output = np.array([[137001, 3, 200, 2458, 125, 160, 620], # 同上
                         [137002, 3, -248, 2120, 150, 160, 620]]).astype(np.int32) # 同上

camera_output = np.array([[127002, 3, -848, 4120, 110, 170, 620], # 同上
                          [127011, 2, 200, 2560, 123, 170, 620]]).astype(np.int32) # 同上


"""
camera_output = np.array([[137001, 3, 250, 433, 150, 160, 30],
                         [137002, 3, 353, 353, 150, 160, 45],
                         [137003, 3, 433, 250, 150, 160, 60],
                         [137004, 3, 433, -250, 150, 160, 120],
                         [137005, 3, 353, -353, 150, 160, 135],
                         [137006, 3, 250, -433, 150, 160, 150],
                         [137007, 3, -250, -433, 150, 160, 210],
                         [137008, 3, -353, -353, 150, 160, 225],
                         [137009, 3, -433, -250, 150, 160, 240],
                         [137010, 3, -433, 250, 150, 160, 300],
                         [137011, 3, -353, 353, 150, 160, 315],
                         [137012, 3, -250, 433, 150, 160, 330],
                         [137013, 3, 500, 0, 150, 160, 90],
                         [137014, 3, -500, 0, 150, 160, 270]]).astype(np.int32)
radar_output = np.array([[127001, 4, 0, 0, 110, 170, 60]]).astype(np.int32)
"""

radar_output = np.zeros((0,7),np.int32)
camera_output = np.zeros((0,7), np.int32)

from radar_process import *
import numpy as np
import pickle
from config_operate import load_config
location = 'jylhX'
pickle_name = location.split('_')[0]
pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
radar_Ip_list = load_config(location)['radar']
data_pickle = pickle.load(pickle_file)
L0_UTM = data_pickle['L0']

Radar_IP = radar_Ip_list[0]
Radar_conf = load_config(location, radar_Ip_list[0])

V1 = int(Radar_conf['vmin']) if 'vmin' in Radar_conf else 0
ini_life = int(Radar_conf['ini_life']) if 'ini_life' in Radar_conf else 0    
full_life = int(Radar_conf['full_life']) if 'full_life' in Radar_conf else 2
usage = Radar_conf['usage']
radar_direction = Radar_conf['direction'] if 'lane' in usage and 'direction' in Radar_conf and \
                  Radar_conf['port'] else 'head'

IP_num = eval(Radar_IP.split('.')[-1])*1000


data = data_pickle[Radar_IP]
P0_UTM, P0_radar = np.int32(data['P0'][0] * 100), np.int32(data['P0'][1])
Homography = data['Calibration'][-1]['H']
radar_range_arr = np.array(eval(Radar_conf['area_radar']), np.int32) if Radar_conf['area_radar']  else np.empty(0)


last_radar_frame = 0
last_radar_state = np.empty((0, 8), dtype=np.int32)  # ID, class, Xr, Yr, Vx, Vy, PV, life
radar_position = np.zeros((2, 7, 6)) * np.nan
assign_ID = 1  # 即将使用的可分配的ID
radar_data = np.int32([[400, 2400, 1000, 321],
                       [1200, 900, 1000, 321],
                       [150, 3200, -1000, 321]])

# radar_target_data = radar_ori_data[np.logical_and(np.abs(radar_ori_data[:,2]) > V1, radar_ori_data[:,3] > 0)]
# radar_target_fusion_data = point_fusion(radar_target_data, usage)
# print ('radar_target_data: \n', radar_target_data)
# radar_target_fusion_data = np.int32([[100, 5549, 1300, 3],[200, 7130, 1230, 3],[500, 73234, -76,3]])
radar_state, assign_ID = radar_movement(radar_data, last_radar_state, radar_position, ini_life, full_life, usage, assign_ID)  # ID, class, Xr, Yr, Vx, Vy, PV, life
last_radar_state = np.copy(radar_state)  # last_target_state
print ('radar_movement_stat: \n', radar_state)
print ('radar_position:\n', radar_position)
radar_data[:,0] = radar_data[:,0] + 50
radar_data[:,1] = radar_data[:,1] + 0.05*radar_data[:,2]

radar_data[0,0] = radar_data[0,0] + 500

radar_UTM_state = radar2UTM(radar_state, radar_range_arr, IP_num, Homography, radar_direction, P0_radar, P0_UTM)



import numpy as np
from core.core_process2 import *
radar_output = np.int32([[137001, 3, 430, 1237, 30, 30, 123],
                         [137002, 3, 530, 900, 30, 30, 123],
                         [137003, 3, 530, 900, 30, 30, 123],
                         [138001, 3, 430, 1237, 30, 30, 223],
                         [138002, 3, -1234, 234, 30, 30, 223],
                         [138003, 3, -457, 3230, 30, 30, 223],
                         [140001, 3, 1758, 4523, 30, 30, 323],
                         [140002, 3, 430, 1240, 30, 40, 323],
                         [140005, 3, -457, 3230, 30, 30, 323],
                         [140004, 3, 1758, 4520, 30, 30, 323]])
final_output = same_type_fusion(radar_output)
print('final_output:\n', final_output)

#####################

import numpy as np
from core.core_process2 import *

dt_base, len_frame = 0.04, 1
camera_position = np.zeros((2, 7, 6)) * np.nan
last_camera_state = np.empty((0, 7), np.int32)  # ID, class, Xw, Yw, Vx, Vy, life
# last_camera_state = last_target_state
ini_life = 0
full_life = 2
camera_data = np.int32([[1, 3, 400, 1300],
                        [2, 4, 1200, 900],
                        [4, 3, 800, 2400]])

camera_movement_state = camera_movement(camera_data, last_camera_state, camera_position, ini_life, full_life, dt_base, len_frame)
last_camera_state = np.copy(camera_movement_state)     
print ('camera_state: \n', camera_movement_state)
print ('camera_position:\n', camera_position)
len_frame = 1
camera_data[:,2:] += len_frame*50

camera_data[0,0] = 6



