# -*- coding: utf-8 -*-
"""
Created on 2021-1-23

@author: Lxh

利用TCPServer, ThreadingMixIn来启动GPU线程，每多一个连接，多启一个线程
"""
import ctypes
from ctypes import *
from multiprocessing import Process, Value, Lock, RawArray
from socketserver import TCPServer, ThreadingMixIn, BaseRequestHandler
from pynput.keyboard import Key, Listener
from threading import Thread
import numpy as np
import time, os, torch, sys, cv2
from config_operate import load_sys_config
import json
import pickle

from sdk import cksdk
from push_screen import playVideoOnDemand
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from tracker import update_tracker

gpu_conf = load_sys_config('gpu')
connect_IP = gpu_conf['gpu_IP']
model_name = gpu_conf['model_name']  # Model_Weights = 'yolov3-tiny_xdq.pt'
# model_name = 'yolov3-tiny_xdq_test3.pt'
threshold = float(gpu_conf['threshold'])
port = 6161
sys.path.append("../pytorch-yolov3-ultralytics")
# print(os.popen('ls').readlines())
# os.chdir(os.path.join(os.environ['HOME'], 'pytorch-yolov3-ultralytics/'))

# os.chdir('../pytorch-yolov3-ultralytics')
# sys.path.insert(1, os.getcwd())
from models.common import Conv
from utils.torch_utils import select_device
from utils.general import xywh2xyxy, non_max_suppression, scale_coords

CUDA = torch.cuda.is_available()
Half = False  # 是否采用半精度
model = os.path.join('../pytorch-yolov3-ultralytics/weights/xdq', model_name)
imgsz = 512  # 图像的长边... 一般选为416, 512, 640
labels = [None, 'person', 'bike', 'car', 'bus', 'truck']
flag_push = False
flag_tracker = False
Device = '0' if CUDA else 'CPU'
device = select_device(Device)
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")

# np.set_printoptions(precision=3, suppress=True)
# np.set_printoptions(linewidth=400)


def Non_Max_Suppression(output, conf_thres=0.2, iou_thres=0.35):
    opc = output[output[..., 4] > conf_thres]
    if not opc.shape[0]:
        return np.zeros((0, 6), np.float64)
    opc[:, 5:] *= opc[:, 4:5]  # conf = obj_conf * cls_conf
    box = xywh2xyxy(opc[:, :4])
    max_idex = opc[:, 5:].argmax(1).reshape((-1, 1))
    conf = opc[np.arange(max_idex.size), (max_idex[:, 0] + 5)].reshape((-1, 1))
    x = np.hstack((box, conf, max_idex))[conf.ravel() > conf_thres]  # xmin, ymin, xmax, ymax, score, class
    if not x.shape[0]:
        return np.zeros((0, 6), np.float64)
    return ops_nms(x, x[:, 1], iou_thres)


def ops_nms(bbox, cf, iou_thres):
    bbox = bbox[np.flip(cf.argsort())]
    s = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    out = np.zeros((0, bbox.shape[1]))
    while bbox.shape[0]:
        IoUdex = np.zeros(bbox.shape[0])
        out = np.vstack((out, bbox[0]))
        overlap = np.minimum(bbox[0, 2:4], bbox[:, 2:4]) - np.maximum(bbox[0, 0:2], bbox[:, 0:2])
        idex = (overlap >= 0).all(axis=1)
        I = np.prod(overlap[idex], 1)
        U = s[0] + s[idex] - I
        IoUdex[idex] = np.divide(I, U)
        IoUdex[IoUdex < iou_thres] = 0
        idex = ~(np.logical_and(idex, IoUdex))
        bbox, s = bbox[idex], s[idex]
    return out


def imgshow_process(image_raw, camera_IP, imgshw, show_image_size=(640, 360)):
    while flag.value:
        tick = time.time()
        if imgshow.value < 0:
            cv2.destroyAllWindows()
            continue
        data = np.copy(np.ctypeslib.as_array(image_raw))
        height = data[0] + data[1] * 256
        if height == 0:
            continue
        width = data[2] + data[3] * 256
        image = data[4:height * width * 3 + 4].reshape(height, width, 3)
        try:
            show_image = cv2.resize(image, show_image_size)
            cv2.imshow(camera_IP, show_image)
            cv2.waitKey(1)
        except:
            pass
        while time.time() - tick <= 0.05:
            continue
    print('imgshow process end')


def get_targets_data(detections, human_arr, vehicle_arr):
    targets_data = np.empty((0, 6), np.float32)  # (x1, y1, x2, y2, score, cls)
    detections = detections[np.lexsort(detections[:, 4::-1].T)]   # (x1, y1, x2, y2, score, cls)
    for i in range(len(detections)):  # for each bounding box, do:
        score, cls = int(detections[i, 4]), detections[i, 5]
        xmin, ymin, xmax, ymax = detections[i, 0:4]
        X, Y = int((xmin + xmax) / 2), int(ymax)
        # 根据检测范围筛选目标
        if cls < 1:
            if human_arr.shape[0] > 0:
                if cv2.pointPolygonTest(human_arr, (X, Y), False) < 0:
                    continue
        elif cls == 1:
            if np.vstack((human_arr, vehicle_arr)).shape[0] > 0:
                if (cv2.pointPolygonTest(human_arr, (X, Y), False) < 0) and (cv2.pointPolygonTest(vehicle_arr, (X, Y), False) < 0):
                    continue
        else:
            if vehicle_arr.shape[0] > 0:
                if cv2.pointPolygonTest(vehicle_arr, (X, Y), False) < 0:
                    #print(detection[i])
                    continue
        targets_data = np.vstack((targets_data, detections[i]))
    # print ('prepare result_image: ', time.time()-t1)
    return targets_data


def get_targets_data_and_draw(ori_image, detections, human_arr, vehicle_arr, imgshow):
    image = np.copy(ori_image)
    targets_data = np.empty((0, 4), np.int32)  # ID, class, Xp, Yp
    height, width = image.shape[0:2]
    colors_list = [(255, 0, 0), (0, 255, 0), (0, 255, 255)]
    detections = detections[np.lexsort(detections[:, 4::-1].T)]   # (x1, y1, x2, y2, cls, track_id)
    # print ('detdctions: \n', detections)
    # t1 = time.time()
    for i in range(len(detections)):  # for each bounding box, do:
        if detections[i, 5] == 1000:
            targets_data = np.array([[1000, 0, 0, 0]], np.int32)
            break
        cls, track_id = int(detections[i, 4]), detections[i, 5]
        label = labels[cls]
        # get box coordinate in original image
        # ymin, xmin, ymax, xmax = detections[i, 2:6]
        xmin, ymin, xmax, ymax = detections[i, 0:4]
        # ymin = int(max(0, np.floor(ymin + 0.5).astype('int32')) / model_image_size[1] * height)
        # xmin = int(max(0, np.floor(xmin + 0.5).astype('int32')) / model_image_size[0] * width)
        # ymax = int(min(model_image_size[1], np.floor(ymax + 0.5).astype('int32')) / model_image_size[1] * height)
        # # print("ymax=", ymax)
        # xmax = int(min(model_image_size[0], np.floor(xmax + 0.5).astype('int32')) / model_image_size[0] * width)

        # 目标视作一个点，位置为图像坐标系（X， Y），位于box底部中心
        X, Y = int((xmin + xmax) / 2), int(ymax)
        # 需要画图才执行cv2绘制矩形框
        if imgshow.value > 0:
            bbox_text = "%s | %s" % (label, track_id)
            if "person" in label:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors_list[0], 2)
                image = cv2.putText(image, bbox_text, (xmin - 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    colors_list[0], 2)
            elif "bike" in label:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors_list[0], 2)
                image = cv2.putText(image, bbox_text, (xmin - 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    colors_list[0], 2)
            else:
                if (xmax - xmin) / (ymax - ymin) > 1.8:
                    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
                    image = cv2.putText(image, bbox_text, (xmin - 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        colors_list[2], 2)

                else:
                    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors_list[2], 2)
                    image = cv2.putText(image, bbox_text, (xmin - 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        colors_list[2], 2)

        # 根据检测范围筛选目标
        # if cls < 2:
        #     if human_arr.shape[0] > 0:
        #         if cv2.pointPolygonTest(human_arr, (X, Y), False) < 0:
        #             continue
        # elif cls == 2:
        #     if human_arr.shape[0] > 0 and vehicle_arr.shape[0] > 0:
        #         if cv2.pointPolygonTest(human_arr, (X, Y), False) < 0 and cv2.pointPolygonTest(vehicle_arr, (X, Y), False) < 0:
        #             continue
        # else:
        #     if vehicle_arr.shape[0] > 0:
        #         if cv2.pointPolygonTest(vehicle_arr, (X, Y), False) < 0:
        #             continue
        #         if imgshow.value > 0:
            image = cv2.circle(image, (int(X), int(Y)), 4, (0, 0, 255), -1)  # 在检测范围内绘制坐标点
        targets_data = np.vstack((targets_data, np.hstack((track_id, cls, X, Y))))

    if imgshow.value:
        if human_arr.shape[0] or vehicle_arr.shape[0]:
            image = cv2.polylines(image, [human_arr], True, (0, 127, 255), 2)
            image = cv2.polylines(image, [vehicle_arr], True, (255, 127, 0), 2)
    # print ('prepare result_image: ', time.time()-t1)
    return targets_data, image, height, width


def read_video(camera_IP, user, usage, frame_list):
    if 'tra' in usage:
        print('The Camera is for traffic monitoring')
        address = 'rtsp://{}@{}:554/h264/ch1/main/av_stream'.format(user, camera_IP)
        print(address)
        cap = cv2.VideoCapture(address)
        tick = time.time()
        while flag.value and (time.time() - tick < 3) and len(frame_list) < 20:
            ret, frame = cap.read()
            if ret:
                frame_list.append(frame)
                tick = time.time()
        #get_frame = HikCam(camera_IP, user)
        #tick = time.time()
        #while flag.value and (time.time() - tick < 3) and len(frame_list) < 20:
        #    image = get_frame.get_image()
        #    # cv2.imshow("1", image)
        #    if time.time() - tick < 1/25:
        #        time.sleep(1/25-(time.time() - tick))
        #    if image.shape[0] > 0:
        #        frame_list.append(image)
        #    else:
        #        continue
        #    tick = time.time()
        print("read traffic camera end")
        return
    elif 'ind' in usage:
        print('The Camera is for industrial monitoring')
        result = cksdk.CameraEnumerateDevice()
        if result[0] != 0:
            print("Don't find camera")
            return
        print("Find cameras number: %d" % result[1])
        # 初始化相机
        result = cksdk.CameraInitEx2(camera_IP)
        if result[0] != 0:
            print("open camera failed")
            return
        hCamera = result[1]
        cksdk.CameraReadParameterFromFile(hCamera, 'UGSMT200C_Cfg_A.bin')  # 载入相机配置参数360p
        cksdk.CameraSetIspOutFormat(hCamera, cksdk.CAMERA_MEDIA_TYPE_BGR8)  # 设置相机输出格式
        cksdk.CameraSetTriggerMode(hCamera, 0)  # 设置为连续拍照模式
        cksdk.CameraSetAeState(hCamera, True)  # 设置为自动曝光模式
        # 开启相机
        cksdk.CameraPlay(hCamera)
        tick = time.time()
        while flag.value and (time.time() - tick < 3) and len(frame_list) < 10:
            result = cksdk.CameraGetImageBufferEx(hCamera, 1000)
            img_data = result[0]
            if img_data is not None:
                img_info = result[1]
                bytes_count = img_info.iWidth * img_info.iHeight * 3
                img_array = cast(img_data, POINTER(c_char * bytes_count))
                frame = np.frombuffer(img_array.contents, dtype=np.uint8, count=bytes_count)
                frame.shape = (img_info.iHeight, img_info.iWidth, 3)
                frame_list.append(frame)
                tick = time.time()
        cksdk.CameraPause(hCamera)  # 暂停相机
        cksdk.CameraUnInit(hCamera)  # 去初始化相机
        print("read industrial camera end")
        return


class Server(ThreadingMixIn, TCPServer): pass


class Handler(BaseRequestHandler):
    def setup(self):
        self.model = model  # 此处从全局变量中读取模型
        self.shape = (imgsz, int(round(imgsz / 32. * 9 / 16)) * 32)
        self.conf_thres = threshold
        self.frame_list = list()
        self.Non_Max_Suppression = Non_Max_Suppression
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, max_age=cfg.DEEPSORT.MAX_AGE,
                                 n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    def handle(self):
        # 6.24 chenlei：
        # 添加截图检测，修改跟踪步骤，先筛选检测范围内目标再跟踪
        self.addr = self.request.getpeername()
        # self.request.setblocking(0)  # 非阻塞模式
        self.request.send(b'Thank you for connecting')
        data_conf = self.request.recv(1024)

        Camera_Conf = json.loads(data_conf)
        location = Camera_Conf['location']
        camera_IP = Camera_Conf['IP']
        user = Camera_Conf['user'] if 'user' in Camera_Conf and Camera_Conf['user'] else 'admin:ehl1234.'
        usage = Camera_Conf['usage']
        if 'tra' in usage:
            img_width, img_height = 1920, 1080
        elif 'ind' in usage:
            img_width, img_height = 640, 360
        K_wh = (img_width / 640, img_height / 360)
        human_arr = np.array(eval(Camera_Conf['area_human']), np.int32) if Camera_Conf['area_human'] else np.empty((0, 2))
        vehicle_arr = np.array(eval(Camera_Conf['area_vehicle']), np.int32) if Camera_Conf['area_vehicle'] else np.empty((0, 2))
        human_arr = (human_arr * K_wh).astype(np.int32) if human_arr.size > 0 else human_arr
        vehicle_arr = (vehicle_arr * K_wh).astype(np.int32) if vehicle_arr.size > 0 else vehicle_arr
        # print(human_arr, vehicle_arr)
        # camera_IP = "10.130.29.128"
        (x_offset, y_offset) = (int(int(Camera_Conf['x_offset']) * K_wh[0]), int(int(Camera_Conf['y_offset']) * K_wh[1]))
        ratio = float(Camera_Conf['ratio'])
        det_width = int(img_width * ratio)
        det_height = int(img_height * ratio)
        K_det = (img_width/1920, img_height/1080)
        Image_Raw = RawArray("B", 1920 * 1080 * 3 + 4)  # np.uint8
        # p_imgshow = Process(target=imgshow_process, args=(Image_Raw, camera_IP, imgshow))
        # p_imgshow.start()
        if flag_push:
            process_push = Process(target=playVideoOnDemand, args=(Image_Raw, camera_IP, flag))
            process_push.start()  # 推流
        # i = 0
        # data_list = []
        while flag.value:  # 1秒钟内接不到数据或接到错误数据，断开连接
            if imgshow.value > 0 and 'p_imgshow' not in vars():  # 开启画图进程
                p_imgshow = Process(target=imgshow_process, args=(Image_Raw, camera_IP, imgshow))
                p_imgshow.start()
            if imgshow.value < 0 and 'p_imgshow' in vars():
                p_imgshow.terminate()
                del p_imgshow
            if 'read_video_thread' not in vars():
                read_video_thread = Thread(target=read_video, args=(camera_IP, user, usage, self.frame_list))
                read_video_thread.start()
            if not read_video_thread.is_alive():
                del read_video_thread
                self.frame_list.clear()
            if not len(self.frame_list):
                time.sleep(0.0005)
                continue
            t0 = (time.time() * 1000)
            ori_image = np.copy(self.frame_list[-1])
            len_frame = len(self.frame_list)
            self.frame_list.clear()
            # img_height, img_width = ori_image.shape[0:2]
            # detect_width, detect_height = int(img_width * ratio), int(img_height * ratio)
            # t1 = time.time()
            det_image = ori_image[y_offset:min(y_offset+det_height, img_height), x_offset:min(x_offset+det_width, img_width)]
            boxed_image = cv2.resize(det_image, self.shape)
            # boxed_image = cv2.copy(detect_image)
            img = np.expand_dims(boxed_image.transpose(2, 0, 1), 0)
            img = torch.from_numpy(img / 255.).to(device)
            img = img.half() if (Half and CUDA) else img.float()

            output = self.model(img, augment=False)[0].cpu().numpy()
            detections = self.Non_Max_Suppression(output, self.conf_thres)  # (x1, y1, x2, y2, score, cls)
            detections[:, 0:4] = scale_coords(img.shape[2:], detections[:, 0:4], det_image.shape).round()
            # detections = detections.numpy()
            if detections.size:
                detections[:, 0] += x_offset
                detections[:, 2] += x_offset
                detections[:, 1] += y_offset
                detections[:, 3] += y_offset
            detections = get_targets_data(detections, human_arr, vehicle_arr)
            detections = np.array(update_tracker(detections, ori_image, self.deepsort))  # (x1, y1, x2, y2, cls, track_id)

            # detections = detections.cpu().numpy().astype(np.int32)
            # detections[:, 4] = detections[:, 5]
            # detections[:, 5] = np.arange(1, detections.shape[0] + 1)
            # print(time.time() - t1)
            if not detections.any():
                detections = np.float64([[0., 0., 0., 0., 0., 1000.]])
            else:
                detections[:, 4] += 1
            camera_target_data, result_image, i_height, i_width = get_targets_data_and_draw(ori_image, detections,
                                                                                            human_arr, vehicle_arr,
                                                                                            imgshow)
            # camera_target_data: ID, class, Xp, Yp

            # print(camera_target_data)
            # print(len_frame)
            # data_list.append((camera_target_data, len_frame))
            # output2 = open('data.pkl', 'wb')
            # pickle.dump(data_list, output2)
            # output2.close()
            # print(i)
            # i += 1

            i_packet_head = [i_height % 256, i_height // 256, i_width % 256, i_width // 256]
            i_packet_body = result_image.ravel()
            i_packet = np.insert(i_packet_body, 0, i_packet_head)
            memoryview(Image_Raw).cast('B')[:result_image.size + 4] = i_packet
            if camera_target_data.size % 4 > 0:
                print(camera_target_data)
                continue
            packet_head = (t0, len_frame, camera_target_data.size)
            packet_body = camera_target_data.ravel().astype(np.float64)
            packet = np.insert(packet_body, 0, packet_head)
            # print(packet)
            try:
                self.request.send(packet.tostring())
            except:
                break
        if 'p_imgshow' in vars(): 
            p_imgshow.terminate()
        # print(image.size)
        # except:
        #     # self.request.send(b'image not integrally received.')
        #     continue

    def finish(self):
        print('{} disconnected.'.format(self.addr))


def Load_model(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = torch.load(weights, map_location=device)['model'].float().fuse().eval()  # load FP32 model
    # Compatibility updates
    for m in model.modules():
        if type(m) in [torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ReLU6]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv and torch.__version__ >= '1.6.0':
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    model.half() if (Half and CUDA) else None  # to FP16转为半精度
    return model


def init_keyboard_listener():
    def on_press(key):
        if key not in Key and key.char == 'C':
            imgshow.value *= -1
        if key == Key.f3:
            flag.value = -1

    def on_release(key):
        return

    def start():
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    T_listen = Thread(target=start, name='t1', args=())
    T_listen.start()


if __name__ == '__main__':
    global flag, imgshow

    flag = Value('i', 1)
    imgshow = Value('i', -1)  # -1 指不显示image，1指正常显示
    init_keyboard_listener()
    model = Load_model(model, map_location=device)
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if Half else img) if device.type != 'cpu' else None  # 模型预热
    while True:
        try:
            server = Server(('127.0.0.1', port), Handler)
            break
        except OSError:
            print('OSError')
            port += 1 if port < 6164 else -4
    Thread(target=server.serve_forever).start()
