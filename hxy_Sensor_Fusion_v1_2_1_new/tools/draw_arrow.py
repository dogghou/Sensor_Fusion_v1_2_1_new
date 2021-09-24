import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.text import Text
import matplotlib.patches as mp
from math import radians, degrees, sin, cos, pi
import numpy as np
import time
from funs import draw_lane
from config_operate import load_sys_config
from multiprocessing import Process, Value, Lock, RawArray



def draw_process(location, Target_Send_Raw, mapshow):
    while True:
        tick = time.time()
        if mapshow.value < 0:
            plt.close()
            try:
                del fig
            except:
                pass
            continue
        try:
            fig
        except:
            fig = plt.figure(num=str(location), figsize=(8, 8))
            draw_lane(location)
            dot_fusion_vec = plt.plot(list(), list(), 'go')[0]
            dot_fusion_per = plt.plot(list(), list(), 'bo')[0]
            text_list = list()
            arrow_list = list()
            for i in range(50):
                text_list.append(plt.text(float(), float(), str(), fontsize=10, verticalalignment="bottom",
                                          horizontalalignment="left"))
                arrow_list.append(plt.arrow(642740, 3283470, 1, 1,
                                            length_includes_head=True, head_width=0.2, lw=2,
                                            color="r"))
                # arrow_list[i].set_xy(np.array([[642740, 3283470], [642741, 3283471]]))
        plt.pause(0.001)

        data = np.copy(np.ctypeslib.as_array(Target_Send_Raw))
        target_draw = data[2:int(2 + data[0])].reshape(-1, 6)  # np.float64: ID,class,Xw,Yw,Vx,Vy
        # print("target_send: \n", target_draw)
        # target_send = np.float64([[1, 2, 431159.844, 3394696.216, 0, 0], [1, 2, 431185.111, 3394702.886, 0, 0]])
        # 绘制目标
        print('fusion_UTM:\n', target_draw[:, 2:4])
        ID_list = (target_draw[:, 0].astype(int)).tolist()
        class_list = target_draw[:, 1].tolist()
        x_list = target_draw[:, 2].tolist()
        y_list = target_draw[:, 3].tolist()
        fusion_UTM = target_draw[target_draw[:, 1] > 1, 2:4]
        dot_fusion_vec.set_data(fusion_UTM[:, 0], fusion_UTM[:, 1])
        fusion_UTM = target_draw[target_draw[:, 1] == 1, 2:4]
        dot_fusion_per.set_data(fusion_UTM[:, 0], fusion_UTM[:, 1])

        for i in range(len(text_list)):
            if i in range(len(ID_list)):
                Xt, Yt = x_list[i], y_list[i]
                text_list[i].set_position((Xt, Yt))
                text_list[i].set_text(str(ID_list[i]))


            else:
                text_list[i].set_text(str())
        plt.pause(0.001)
        while time.time() - tick <= 0.09:
            continue


if __name__ == '__main__':
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
                             np.array([[3, -17, 2203, 147, 182, 179]])))

    from plt_draw import camera_movement

    Target_Send_Raw = np.array([[1, 3, -17, 2203, -651, 958, 3041, 1]],dtype=np.int32)
    draw_process("tuanjielu", Target_Send_Raw, mapshow=Value('i', 1))
    camera_position = np.zeros((2, 30, 6)) * np.nan

    for camera_data in camera_data_list:
        camera_movement_data = camera_movement(camera_data, last_target_state, camera_position, dt=0.033)
        last_target_state = np.copy(camera_movement_data)

        print(camera_movement_data)
