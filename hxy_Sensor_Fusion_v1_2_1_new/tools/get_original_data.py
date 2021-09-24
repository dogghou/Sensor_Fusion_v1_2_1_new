import numpy as np
import sys, os, pickle
from config_operate import load_sys_config, load_config


def get_original_data(Homography, Pw):
    Pp = np.array([[Pw[0]], [Pw[1]], [1]])
    tmp = np.dot(np.linalg.inv(Homography), Pp)
    Pc = tmp[0:2] / tmp[-1]
    return Pc.T[0]


def get_target_location(Homography, Xp, Yp):  # 利用单应性变换将某平面坐标转为世界坐标
    Pp = np.array([[Xp], [Yp], [1]])  # 图像齐次坐标系
    tmp = np.dot(Homography, Pp)
    Pw = tmp[0:2] / tmp[-1]  # 当前帧目标的位置坐标，该坐标Y轴指向正北方，X轴指向东方
    return Pw.T[0]  # 输出单位是（cm）


def radar2UTM(ori_data, Homography, P0_UTM):
    # 调用 get_target_location 将多组从雷达坐标系中得到的目标位置及速度旋转到世界坐标系
    target_UTM_state = np.zeros((ori_data.shape[0], 2), np.int32)  # class, Xr, Yr, Vx, Vy, PV, life
    for i in range(ori_data.shape[0]):
        Xr, Yr = ori_data[i]
        target_UTM_state[i] = get_target_location(Homography, Xr, Yr) + P0_UTM
    return target_UTM_state  # Xw, Yw


def UTM2sensor(UTM_data, Homography, P0_UTM):
    original_data = np.zeros((UTM_data.shape[0], 2), np.int32)  # Xr, Yr
    for i in range(UTM_data.shape[0]):
        Xw, Yw = UTM_data[i]
        original_data[i, ] = get_original_data(Homography, (Xw, Yw) - P0_UTM)
    return original_data  # Xr, Yr



if __name__ == '__main__':
    location = 'lrlmX_1'
    Sensor_IP = '172.16.11.120'
    import matplotlib.pyplot as plt
    from funs import draw_lane
    camera = True
    evaluate = False
    img_shape = np.array([640, 360], dtype=np.float64)

    draw_lane(location)
    if not evaluate:
        pos = plt.ginput(4)
        pos = np.array(pos)
        point_arr = pos
    if evaluate:
        radar_arr = np.array([[-613, 2921], [46, 2981], [-775, 3558], [146, 3743], [-901, 4393], [284, 4706]])

        point_arr = np.array([[644218.528, 3284564.428],[644211.445, 3284562.811],[644223.126, 3284555.229], [644216.044, 3284555.803], [644227.363, 3284547.792], [644221.197, 3284547.472]])
        # plt.scatter(point_arr[:, 0], point_arr[:, 1], s=20, c='b', marker='o',)
        b=point_arr[:, 0].flatten()
        for i in range(point_arr.shape[0]):
            plt.text(point_arr[i, 0], point_arr[i, 1], s=str(i+1))
        plt.pause(0)

    for file_name in os.listdir("location_H"):
        if location.split('_')[0] in file_name:
            pickle_name = file_name
            break
    pickle_file = open('location_H/{}'.format(pickle_name), 'rb')
    data_pickle = pickle.load(pickle_file)
    data = data_pickle[Sensor_IP]
    Homography = data['Calibration'][-1]['H']
    cali_shape = np.array([1920, 1080], dtype=np.float64)
    L0_UTM = data_pickle['L0']
    P0_UTM = np.int32(data['P0'][0] * 100)
    ori_data = UTM2sensor(point_arr * 100, Homography, P0_UTM)
    if evaluate:
        UTM_data = radar2UTM(radar_arr, Homography, P0_UTM)
        print(UTM_data/100)

    if camera:
        ori_data[:, 0] = ori_data[:, 0] * img_shape[0]/cali_shape[0]
        ori_data[:, 1] = ori_data[:, 1] * img_shape[1]/cali_shape[1]
    print(ori_data)
    # [[64420762 328453425]]
    # [[644207.62 3284534.25]
    #  [644207.77 3284535.33]
    # [644203.79
    # 3284532.28]]
    # (384, 77), (371, 110), (259,127), (5,126), (183,73)
    # (190, 45), (405, 45), (365, 145), (0, 150)

