import time
import numpy as np


def send_traffic_event(flag, target_event_raw, OtherEvent, lock):
    target_count = np.zeros((0, 8), dtype=np.int32)
    while flag.value:
        t1 = time.time()
        data = np.ctypeslib.as_array(target_event_raw)
        if not data.size:
            target_count[:, 4:] -= 1  # 本帧没事件发生则事件计数减一
        else:
            target_event = data[1:data[0]+1].reshape((-1, 8)).astype(np.int32)  # n*8: id,class,Xw,Yw,弱势群体,超速,低速,压线
            # print(target_event)
            # event_index:事件计数和新事件数据交集、交集在target_count索引、交集在target_event索引
            event_index = np.intersect1d(target_count[:, 0], target_event[:, 0], return_indices=True)
            if not event_index[0].size:
                target_count[:, 4:] -= 1
                target_count = np.vstack((target_count, target_event))
            else:
                target_count[event_index[1], 4:] += target_event[event_index[2], 4:]
                target_count[event_index[1], 2:4] = target_event[event_index[2], 2:4]
                target_count_temp1 = target_count[event_index[1]]
                target_count_temp2 = np.delete(target_count, event_index[1], axis=0)
                target_count_temp2[:, 4:] -= 1
                target_count = np.vstack((target_count_temp1, target_count_temp2))
                target_event = np.delete(target_event, event_index[2], axis=0)
                target_count = np.vstack((target_count, target_event))
        target_count[:, 4:][target_count[:, 4:] > 10] = 10
        target_count[:, 4:][target_count[:, 4:] < 0] = 0  # 事件计数中最大计数为10，最小为0
        # print(target_count)
        target_count = np.delete(target_count, np.where(np.all(target_count[:, 4:] == 0, axis=1)), axis=0)  # 删除无事件计数的id
        target_send = np.where(target_count[:, 4:] >= 5, 1, 0)  # 累计5帧则视为事件发生
        target_send = np.hstack((target_count[:, 0:4], target_send))  # n*8: id,class,Xw,Yw,弱势群体,超速,低速,压线
        target_send = np.delete(target_send, np.where(np.all(target_send[:, 4:] == 0, axis=1)), axis=0)
        # 发送交通事件 target_send
        target_send = target_send.reshape(1, -1)
        packet_head = [target_send.size, 0]
        packet_body = target_send.ravel()
        packet = np.insert(packet_body, 0, packet_head)
        packet = packet.astype(np.float64)
        lock.acquire()
        memoryview(OtherEvent).cast('B').cast('d')[0:packet.size] = packet
        lock.release()
        # print(target_send)
        while time.time() - t1 < 1 / 10:
            continue
