import serial
import serial.tools.list_ports as sps
import numpy as np
import subprocess, os, datetime, time
# import win32api
from multiprocessing import Value,  Process



def gps_time_check(time_plus):
    while True:
        try:
            port = list(sps.comports())[-1][0]
        except:
            continue
        p = subprocess.Popen(f"echo '123456' | sudo -S chmod 777 {port}", stderr=subprocess.PIPE, shell=True)
        if p.wait() == 0:
            s = serial.Serial(port, 9600, timeout=60)
            while True:
                try:
                    recv = s.readline().decode()
                except:
                    s.close()
                    break
                data = recv.split(',')
                try:
                    if data[0] in ['$GPRMC', '$GNGGA'] and data[1]:
                        # print(data)
                        hour = int(data[1].split('.')[0][0:2])
                        hour += 8
                        minute = int(data[1].split('.')[0][2:4])
                        second = int(data[1].split('.')[0][4:6])
                        year = int('20' + data[9][4:6])
                        mouth = int(data[9][2:4])
                        day = int(data[9][0:2])
                        gps_time_format = datetime.datetime(year, mouth, day, hour, minute, second)
                        gps_time = time.strptime(str(gps_time_format), '%Y-%m-%d %H:%M:%S')
                        gps_time = time.mktime(gps_time)
                        system_time = time.time()
                        print(gps_time, system_time)
                        time_plus.value = gps_time - time.time()
                        if gps_time - system_time > 0.01 or system_time > gps_time:
                            print(f'系统时间: {system_time}\nGPS时间: {gps_time}')
                            update_time = str(datetime.datetime.fromtimestamp(gps_time + 0.1))
                            os.popen(f"timedatectl set-time '{update_time}'")
                            print(f'校正成功 at {time.time()}！\n')

                except:
                    continue


def GPS_calibrate(GPS_COM, time_plus):
    s= serial.Serial(GPS_COM, 9600, timeout=60)
    if s.isOpen():
        print('串口打开成功！\n')
    else:
        print('串口打开失败！\n')
    while True:
        GPS_data = np.array(s.readline().decode('utf-8').split(','))
        # print(GPS_data)
        if GPS_data[0] == '$GPRMC':
            if GPS_data[1]:
                hour = int(GPS_data[1].split('.')[0][0:2])
                hour += 8
                minute = int(GPS_data[1].split('.')[0][2:4])
                second = int(GPS_data[1].split('.')[0][4:6])
                year = int('20' + GPS_data[9][4:6])
                mouth = int(GPS_data[9][2:4])
                day = int(GPS_data[9][0:2])
                gps_time_format = datetime.datetime(year, mouth, day, hour, minute, second)
                gps_time = time.strptime(str(gps_time_format), '%Y-%m-%d %H:%M:%S')
                gps_time = time.mktime(gps_time)
                print(gps_time)
                time_plus.value = gps_time - time.time()
                # system_time = time.time()
                # if gps_time - system_time > 0.01 or system_time > gps_time:
                #     print(f'系统时间: {system_time}\nGPS时间: {gps_time}')
                #     update_time = str(datetime.datetime.fromtimestamp(gps_time + 0.1))
                #     win32api.SetSystemTime
                #     os.popen(f"timedatectl set-time '{update_time}'")
                #     print(f'校正成功 at {time.time()}！\n')


if __name__ == '__main__':
    time_plus = Value('d', 0)
    GPS_COM = 'com13'
    Process(target=gps_time_check, args=(time_plus,)).start()
    while True:
        print('*****')
        print(time_plus.value)
    # GPS_calibrate(GPS_COM,time_plus)
    # gps_time_check()
