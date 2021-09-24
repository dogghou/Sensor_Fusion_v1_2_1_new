import time
import numpy as np
from funs import UTM2W84, sigmoid, tanh
from math import sqrt, pow, tanh
from core.core_process import camera_movement, data_fusion, heading
from math import atan, pi

if __name__ == "__main__":
    def heading(vxy):
        vx = vxy[0]
        vy = vxy[1]
        if vx >= 0 and vy > 0:
            heading = atan(vx / vy)
        elif vx > 0 and vy < 0:
            heading = -atan(vy / vx) + pi / 2.
        elif vx <= 0 and vy < 0:
            heading = atan(vx / vy) + pi
        elif vx < 0 and vy > 0:
            heading = -atan(vy / vx) + pi * (3 / 2.)
        elif vy == 0:
            if vx > 0:
                heading = pi / 2.
            elif vx < 0:
                heading = pi * (3 / 2.)
            else:
                heading = 2 * pi
        return np.int32(np.degrees(heading) * 10)  # 返回10倍角度值
    xy = np.array([-1, 0])
    angle = heading(xy)
    print(angle)
    camera_test = False
    fusion = False
    if camera_test:
        last_target_state = np.empty((0, 10), np.int32)  # ID, class, Xw, Yw, Vx, Vy, m_B, m_G, m_R, life
        camera_position = np.zeros((2, 30, 6)) * np.nan
        camera_output_list = list()
        camera_data_list = list((np.array([[3, -1374, 2533, 81, 132, 137],
                                           [3, -221, 626, 94, 120, 113]]),
                                 np.array([[3, -1368, 2534, 85, 133, 137],
                                           [3, -210, 609, 94, 121, 114]]),
                                 np.array([[3, -1345, 2492, 81, 131, 137],
                                           [3, -204, 591, 92, 119, 113]]),
                                 np.array([[3, -1341, 2493, 82, 133, 140],
                                           [3, -200, 592, 95, 121, 112]]),
                                 np.array([[3, -1348, 2491, 84, 134, 140],
                                           [3, -179, 560, 92, 119, 113]]),
                                 np.array([[3, -1348, 2491, 86, 136, 139],
                                           [3, -179, 560, 94, 120, 113]]),
                                 np.array([[3, -1348, 2491, 86, 137, 142],
                                           [3, -180, 559, 95, 121, 114]]),
                                 np.array([[3, -1325, 2449, 83, 135, 143],
                                           [3, -191, 575, 94, 122, 115]]),
                                 np.array([[3, -1300, 2409, 85, 134, 140],
                                           [3, -144, 513, 92, 118, 110]])))

        camera_data_list = list((np.array([[3, -972, 1905, 125, 163, 160]]),
                                 np.array([[3, -926, 1843, 123, 161, 159]]),
                                 np.array([[3, -974, 1905, 133, 174, 171]]),
                                 np.array([[3, -892, 1779, 129, 169, 167]]),
                                 np.array([[3, -897, 1778, 130, 169, 168]]),
                                 np.array([[3, -881, 1747, 130, 171, 172]]),
                                 np.array([[3, -840, 1688, 130, 169, 167]]),
                                 np.array([[3, -827, 1657, 131, 170, 167]]),
                                 np.array([[3, -791, 1600, 128, 166, 163]]),
                                 np.array([[3, -736, 1517, 128, 163, 159]]),
                                 np.array([[3, -716, 1491, 127, 160, 154]]),
                                 np.array([[3, -697, 1465, 127, 167, 167]])))
        camera_data_list = list((np.array([[3, 273, 1720, 149, 184, 183]]),
                                 np.array([[3, 256, 1745, 148, 183, 181]]),
                                 np.array([[3, 227, 1768, 146, 180, 176]]),
                                 np.array([[3, 189, 1850, 148, 183, 180]]),
                                 np.array([[3, 151, 1935, 149, 188, 186]]),
                                 np.array([[3, 132, 1963, 148, 185, 183]]),
                                 np.array([[3, 88, 2053, 152, 190, 188]]),
                                 np.array([[3, 66, 2082, 149, 187, 184]]),
                                 np.array([[3, 23, 2141, 149, 186, 183]]),
                                 np.array([[3, 13, 2139, 143, 182, 182]]),
                                 np.array([[3, -17, 2203, 147, 182, 179]]),
                                 np.array([[3, -17, 2203, 147, 182, 179]])))

        for camera_data in camera_data_list:
            camera_movement_data = camera_movement(camera_data, last_target_state, camera_position, dt=0.033)
            last_target_state = np.copy(camera_movement_data)
            camera_output_list.append(camera_movement_data[:, 1:6])
            # print(camera_movement_data)
            # print("position", camera_position[:, 1, :])
    if fusion:
        last_target_state = np.empty((0, 8), np.int32)
        radar_output = np.array([], dtype=np.int32).reshape(0, 4)
        for camera_output in camera_output_list:
            target_state = data_fusion(last_target_state, camera_output, radar_output, dt=0.1)
            last_target_state = np.copy(target_state)
            last_target_state = last_target_state
            print(heading(np.array([target_state[0, 4], target_state[0, 5]], dtype=np.float64)))
            print(target_state)
            # print(target_state)
