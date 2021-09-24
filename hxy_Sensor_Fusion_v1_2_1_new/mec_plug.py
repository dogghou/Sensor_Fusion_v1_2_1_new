# -*- coding: utf-8 -*-
"""
content:
        MEC边缘计算单元插件
        1.网络监控
        2.看门狗控制
        3.运维信息一览

author:   zhanglian
time:     2021/02/02
"""
from tkinter import *
import os,platform,subprocess,psutil,time,datetime
import socket,pcap,dpkt
from pynput.keyboard import Listener,Key
from threading import Thread
from multiprocessing import Manager
from config_operate import load_config

class Mec:
    def __init__(self,location,ip_list,NetworkRate_dict,NetworkCount_dict,SensorOnline_dict):
        self.location=location
        self.ip_list=ip_list
        # self.ip_extra_list=[] # 部署后未使用的设备ip
        self.NetworkRate_dict=NetworkRate_dict
        self.NetworkCount_dict=NetworkCount_dict
        self.SensorOnline_dict=SensorOnline_dict
        self.networkrate_dict={}

        for ip in self.ip_list:
            self.NetworkRate_dict[ip]=0
            self.NetworkCount_dict[ip]=0
            self.networkrate_dict[ip]=0
        self.t1 = self.t0 = time.time()
        self.status_log='~/OPEX/log/status'
        self.sensor_dict={'camera':'视频检测器','radar':'毫米波雷达','lidar':'激光雷达 '}

        ### 启动键盘监听
        self.keyboard_listener = Listener(on_press=self.press)
        self.keyboard_listener.start()

        '''*********************************************窗口主界面****************************************************'''
        tag_font = ('',9,'bold')
        link_font = ('',9,'underline') #Verdana
        self.tk = Tk()
        self.tk.title('MEC(隐藏:退出或关闭,显示:INSERT键)')
        self.tk.geometry(f'380x480+100+100')
        self.tk.protocol("WM_DELETE_WINDOW", self.tk.withdraw) #重定义关闭或退出窗口为隐藏窗口

        self.system_info = platform.version() # 系统信息
        self.v_ethName, self.v_ethIP = StringVar(), StringVar()
        self.last_boot_time = os.popen('uptime -s').readline().strip() # 系统最近一次开机时间
        self.v_runningIime = StringVar()
        self.v_bootCount,self.v_shutdownCount,self.v_poweroffCount = IntVar(),IntVar(),IntVar()
        Label(text=f'系统： {self.system_info}').place(x=10, y=5, height=20)
        Label(text='Ethnet：',font=tag_font).place(x=10,y=30,height=20)
        Label(textvariable=self.v_ethName).place(x=70, y=30,height=20)
        Label(text='IPv4：',font=tag_font).place(x=180,y=30,height=20)
        Label(textvariable=self.v_ethIP).place(x=225, y=30,height=20)
        Label(text=f'启动： {self.last_boot_time}').place(x=10,y=55,height=20)
        Label(text='已运行：').place(x=10, y=80, height=20)
        Label(textvariable=self.v_runningIime).place(x=60, y=80, height=20)
        Label(text='今日开机次数：').place(x=10, y=105, height=20)
        Label(textvariable=self.v_bootCount).place(x=100, y=105, height=20)
        Label(text='今日关机次数：').place(x=160, y=105, height=20)
        Label(textvariable=self.v_shutdownCount).place(x=250, y=105, height=20)
        Label(text='今日异常断电次数：').place(x=10, y=130, height=20)
        Label(textvariable=self.v_poweroffCount).place(x=120, y=130, height=20)

        self.v_inner_disconnectCount,self.v_outer_disconnectCount=IntVar(),IntVar()
        y_Network=160
        Label(text='网络监控', relief=RAISED).place(x=10,y=y_Network,height=25,width=360)
        Label(text='今日内网掉线次数：').place(x=10,y=y_Network+30,height=20)
        Label(textvariable=self.v_inner_disconnectCount).place(x=130,y=y_Network+30,height=20)
        Label(text='今日外网掉线次数：').place(x=10,y=y_Network+50,height=20)
        Label(textvariable=self.v_outer_disconnectCount).place(x=130,y=y_Network+50,height=20)
        Label(text='IP',font=tag_font).place(x=10,y=y_Network+75,height=20)
        Label(text='类型').place(x=120,y=y_Network+75,height=20)
        Label(text='状态').place(x=220,y=y_Network+75,height=20)
        Label(text='传输速率M/s').place(x=300,y=y_Network+75,height=20)
        frame_scrollbar=Frame()
        frame_scrollbar.place(x=10,y=255,width=360,height=62)
        sb = Scrollbar(frame_scrollbar)
        sb.pack(side=RIGHT, fill=Y)
        self.v_lb=StringVar()
        self.lb = Listbox(listvariable=self.v_lb,yscrollcommand=sb.set)
        self.lb.place(x=10,y=255,width=350,height=62)
        sb.config(command=self.lb.yview)
        self.v_NetworkRate_baidu=StringVar()
        self.v_NetworkRate_baidu.set(0)
        self.v_online_baidu=StringVar()
        Label(text='202.108.22.5').place(x=10, y=y_Network+157, height=20)
        Label(text='百度').place(x=120, y=y_Network+157, height=20)
        Label(textvariable=self.v_online_baidu).place(x=218, y=y_Network+157, height=20)
        Label(textvariable=self.v_NetworkRate_baidu).place(x=305, y=y_Network+157, height=20)
        Label(text='-' * 70).place(x=10, y=y_Network+172,height=10)
        self.v_NetworkRate_total = StringVar()
        self.v_NetworkRate_total.set(0)
        Label(text='Total').place(x=10, y=y_Network+185)
        Label(textvariable=self.v_NetworkRate_total).place(x=305, y=y_Network+185)

        self.cmd,self.guarddog_state,self.guarddog_instruct,self.deadtime = '00','启动','暂停',10 # 看门狗初始参数设置
        y_Guarddog = y_Network+185
        Label(text='看门狗', relief=RAISED).place(x=10, y=y_Guarddog+30, height=25,width=360)
        Label(text='ID',font=tag_font).place(x=10, y=y_Guarddog + 60, height=20)
        Label(text='运行状态').place(x=170, y=y_Guarddog + 60, height=20)
        Label(text='控制').place(x=300, y=y_Guarddog + 60, height=20,width=35)
        Label(text='172.16.10.10').place(x=10, y=y_Guarddog+80, height=25)
        self.v_guarddog_package=StringVar()
        Label(textvariable=self.v_guarddog_package,justify=LEFT).place(x=170, y=y_Guarddog+80, height=25)
        self.btn_guarddog_control=Button(text=self.guarddog_instruct,command=lambda:self.guarddog_control(self.guarddog_instruct))
        self.btn_guarddog_control.place(x=300, y=y_Guarddog+80, height=20, width=35)
        Button(text=' 更多日志 ',relief=FLAT,font=link_font,cursor='cross',command=lambda:subprocess.Popen(['xdg-open','/home/user/OPEX/log'])).place(x=150,y=450)

        Thread(target=self.LAN_sock).start()
        Thread(target=self.outernet_listen).start()
        Thread(target=self.sniffer_flow_rate).start()
        Thread(target=self.info_update).start()
        self.tk.mainloop()
        '''**********************************************窗口主界面***************************************************'''

        # 键盘监听关闭
        self.keyboard_listener.join()

    # 按INSERT键显示隐藏窗口
    def press(self, key):
        # if key not in Key and key.char=='F8':
        if key == Key.insert:
            self.tk.deiconify()

    # 局域网监控
    def LAN_sock(self):
        lan_flag = 0
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            p_mec = psutil.Process(os.getppid())
            while True:
                t0 = time.time()
                if t0 - self.t0 >= 1:
                    break
            while p_mec.is_running():
                t1 = time.time()
                if t1 - t0 >= 1:
                    now = datetime.datetime.now()
                    hour = str(hex(now.hour))[2:]
                    if len(hour) == 1:
                        hour = '0' + hour
                    minute = str(hex(now.minute))[2:]
                    if len(minute) == 1:
                        minute = '0' + minute
                    second = str(hex(now.second))[2:]
                    if len(second) == 1:
                        second = '0' + second
                    self.package = 'bb' + self.cmd + '00' + hour + minute + second
                    try:
                        s.sendto(bytes.fromhex(self.package), ('172.16.10.10', 5000))
                        self.btn_guarddog_control['state'] = NORMAL
                        self.v_guarddog_package.set(self.package)
                        self.deadtime=10
                        if lan_flag == 0:
                            os.popen(f"echo innernet {self.v_ethIP.get()} connect at {now}! >> {self.status_log}")
                            lan_flag = 1
                    except:
                        print(f'断开!电脑在{self.deadtime}s后断电')
                        self.btn_guarddog_control['state']=DISABLED
                        if self.cmd != '01':
                            self.v_guarddog_package.set(f'断开!电脑在{self.deadtime}s后断电')
                            if self.deadtime == 0:
                                self.v_guarddog_package.set('')
                                break
                            self.deadtime -= 1
                        else:
                            self.v_guarddog_package.set('看门狗暂停监控！')
                        for ip in self.ip_list:
                            # if ip in self.ip_extra_list:
                            #     continue
                            self.SensorOnline_dict[ip]='已断开'
                        if lan_flag == 1:
                            os.popen(f"echo innernet {self.v_ethIP.get()} disconnect at {now}! >> {self.status_log}")
                            lan_flag = 0
                    t0 = t1

    # 外网监控
    def outernet_listen(self):
        outnet_flag = 0
        p_mec = psutil.Process(os.getppid())
        while p_mec.is_running():
            now = datetime.datetime.now()
            status,output=subprocess.getstatusoutput('ping -c 1 202.108.22.5')
            if 'connect: Network is unreachable' in output:
                self.v_online_baidu.set('Ping不通')
                if outnet_flag == 1:
                    os.popen(f"echo outernet disconnect at {now}! >> {self.status_log}")
                    outnet_flag = 0
            else:
                self.v_online_baidu.set('外网可用')
                if outnet_flag == 0:
                    os.popen(f"echo outernet disconnect at {now}! >> {self.status_log}")
                    outnet_flag = 1

    # 网络流量统计
    def sniffer_flow_rate(self):
        ethName = self.v_ethName.get()
        p_mec = psutil.Process(os.getppid())
        while p_mec.is_running():
        # if ethName in self.eth_list:
            if not ethName:
                sniffer = pcap.pcap(ethName)
                # sniffer.setfilter('tcp port 554')  # 设置监听过滤器
                while True:
                    t_count=t_rate = time.time()
                    if t_rate - self.t0 >= 1:
                        break
                for packet_time, packet_data in sniffer:
                    t = time.time()
                    # if self.v_ethName.get() != ethName:
                    #     sniffer.close()
                    #     del sniffer
                    #     time.sleep(1)
                    #     break
                    packet = dpkt.ethernet.Ethernet(packet_data)
                    try:
                        src_ip = "%d.%d.%d.%d" % tuple(list(packet.data.src))
                        # dst_ip = "%d.%d.%d.%d" % tuple(list(packet.data.dst))
                        if src_ip in self.NetworkRate_dict:
                            self.NetworkRate_dict[src_ip] += sys.getsizeof(packet_data)/1024/1024
                            self.NetworkCount_dict[src_ip] += sys.getsizeof(packet_data)/1024/1024
                        self.NetworkRate_dict['Total'] += sys.getsizeof(packet_data)/1024/1024
                        self.NetworkCount_dict['Total'] += sys.getsizeof(packet_data)/1024/1024
                    except:
                        pass
                    if t - t_count >=30:
                        for key,count in list(self.NetworkRate_dict.items()):
                            if key != 'Total':
                                if count:
                                    self.SensorOnline_dict[key]='已连接'
                                else:
                                    self.SensorOnline_dict[key]='已断开'
                            self.NetworkCount_dict[key] = 0
                        t_count=t
                    if t - t_rate >= 1:
                        for key,rate in list(self.NetworkRate_dict.items()):
                            # if sensor in self.ip_extra_list:
                            #     self.SensorOnline_dict[sensor]='未使用'
                            #     continue
                            if key == 'Total' or key == '202.108.22.5':
                                self.v_NetworkRate_total.set(str(round(rate, 3)))
                            else:
                                self.networkrate_dict[key]=round(rate, 3)
                            self.NetworkRate_dict[key] = 0
                        t_rate = t

                        v_lb=[]
                        for ip in self.ip_list:
                            ip_show=ip+' '*(13-len(ip))
                            try:
                                sensorType = self.sensor_dict[load_config(self.location, ip).get('type')]
                            except:
                                sensorType = '未设置类型'
                            sensorState = self.SensorOnline_dict.get(ip,'已连接')
                            sensorRate = self.networkrate_dict[ip]
                            text = f"{ip_show}{' '*3}{sensorType}{' '*5}{sensorState}{' '*8}{sensorRate}"
                            v_lb.append(text)
                        self.v_lb.set(v_lb)

    # 看门狗控制
    def guarddog_control(self,instruct):
        if instruct=='暂停':
            self.cmd='01'
            self.guarddog_instruct='启动'
            self.btn_guarddog_control['text']='启动'
            self.guarddog_state = '暂停'
        elif instruct=='启动':
            self.cmd='00'
            self.guarddog_instruct = '暂停'
            self.btn_guarddog_control['text'] = '暂停'
            self.guarddog_state = '启动'

    # 系统信息更新
    def info_update(self):
        eth_name,eth_ip='lo','127.0.0.1'
        p_mec=psutil.Process(os.getppid())
        while True:
            t0 = time.time()
            if t0 - self.t0 >= 1:
                break
        while p_mec.is_running():
            t1 = time.time()
            if t1 - t0 >= 1:
                self.v_runningIime.set(str(os.popen('uptime -p').readline().strip()[3:]))  # 系统运行时间
                self.eth_list =[]
                line_list = os.popen('ifconfig').readlines()
                for line in line_list:
                    if 'flags=' in line:
                        self.eth_list.append(line.split(' ')[0][:-1])
                    elif 'Link' in line:
                        self.eth_list.append(line.split(' ')[0])

                    if '广播' in line or 'broadcast' in line:
                        eth_ip = os.popen("echo '%s' | awk '{print $2}'" % line.strip()).readline()[:-1]
                        if 'Ubuntu' in self.system_info:
                            eth_ip = eth_ip[3:]
                        row = line_list.index(line)
                        eth_name = line_list[row - 1].split(' ')[0]
                        if 'docker' in eth_name:
                            continue
                        if ':' in eth_name:
                            eth_name=eth_name[:-1]
                self.v_ethName.set(eth_name)  # 网络接口
                self.v_ethIP.set(eth_ip)  # 本机IP

                today = datetime.datetime.now().date()
                status_today=os.popen(f'grep {today} {self.status_log}').readlines()
                bootCount,shutdownCount,inner_disconnectCount,outer_disconnectCount,=0,0,0,0
                for line in status_today:
                    if 'Boot' in line:
                        bootCount+=1
                    elif 'Shutdown' in line:
                        shutdownCount+=1
                    elif 'innernet' in line and 'disconnect' in line:
                        inner_disconnectCount+=1
                    elif 'outernet' in line and 'disconnect' in line:
                        outer_disconnectCount+=1
                poweroffCount = bootCount - shutdownCount
                self.v_bootCount.set(bootCount)
                self.v_shutdownCount.set(shutdownCount)
                self.v_poweroffCount.set(poweroffCount)
                self.v_inner_disconnectCount.set(inner_disconnectCount)
                self.v_outer_disconnectCount.set(outer_disconnectCount)
                t0 = t1

if __name__ == '__main__':
    location='tuanjielu'
    ip_list=['10.130.210.84','10.130.210.30']
    NetworkRate_dict = Manager().dict({'Total': 0,'202.108.22.5':0})
    NetworkCount_dict = Manager().dict({'Total': 0,'202.108.22.5':0})
    SensorOnline_dict = Manager().dict()
    Mec(location,ip_list,NetworkRate_dict,NetworkCount_dict,SensorOnline_dict)
