"""
# @Time    : 2022/5/10 20:52
# @Author  : Wang
# @Email   :
# @File    : env.py
"""
import numpy as np
import math
import random

from collections import deque
from bus.Bus import Bus
from bus.Bus_stop import BusStop

INTERVAL = 2000  # 1km
mu, sigma = 200, 20
np.random.seed(0)


class Env:
    """
    # 环境中的智能体
    """

    def __init__(self, args):
        self.stops = None
        self.is_control = None
        self.control_queue = None
        self.busses = None
        self.trip = None
        self.now = None
        self.bus_num = 6  # 设置智能体(公交车)的个数，这里设置为6个
        self.stop_num = 12
        self.state_dim = args.state_dim  # 设置智能体的观测纬度 [h_for, h_back, load_rate, wait_rate, satisfy]
        self.action_dim = args.action_dim  # 设置智能体的动作纬度 [hold_time, refuse_num]

        self.sim_horizon = 8 * 60 * 60

    def reset(self):
        """
        # 初始化环境参数
        """
        self.now = 0
        self.trip = []
        self.busses = []
        self.control_queue = deque()  # holding queue
        self.is_control = True
        self.stops = []
        # arr_rates = [1 / 60 / 2, 1 / 60 / 2, 1 / 60 / 1.2, 1 / 60, 1 / 60, 1 / 60 * 3, 1 / 60 * 4, 1 / 60 * 2, 1 / 60,
        #              1 / 60 / 1.5, 1 / 60 / 1.8, 1 / 60 / 2]
        arr_rates = [1 / 60 * 4, 1 / 60 / 2, 1 / 60 * 1, 1 / 60 / 1.5, 1 / 60 / 1.8, 1 / 60 / 2, 1 / 60 * 2, 1 / 60 / 2,
                     1 / 60 / 1.2, 1 / 60, 1 / 60 * 1, 1 / 60 * 3]

        # 初始化公交站的设置
        for i in range(self.stop_num):
            stop = BusStop(stop_id=i, arr_rate=arr_rates[i], loc=i * INTERVAL, wait_num=2)

            if i == self.stop_num - 1:
                stop.next_stop = 0
            else:
                stop.next_stop = i + 1
            self.stops.append(stop)

        # 初始化公交车
        for i in range(self.bus_num):
            bus = Bus(bus_id=i, loc=(i * 2) * INTERVAL + 200, pass_stop=i * 2)
            # 定义前后车关系
            if i == 0:
                bus.forward = i + 1
                bus.backward = self.bus_num - 1
            elif i == self.bus_num - 1:
                bus.forward = 0
                bus.backward = i - 1
            else:
                bus.forward = i + 1
                bus.backward = i - 1
            if bus.pass_stop == self.stop_num - 1:
                bus.next_stop = 0
            else:
                bus.next_stop = bus.pass_stop + 1
            bus.decide_time = self.now
            # log
            bus.log_time.append(self.now)
            bus.log_loc.append(bus.loc / INTERVAL)
            bus.log_load.append(bus.load)
            self.busses.append(bus)

        # 初始化下次事件表
        # 检查每辆公交车的状态: 初始的时候每个公交车下一个状态都是到站，都是正在运行
        for b in self.busses:
            if b.next_stop == 0:
                distance = INTERVAL * self.stop_num - b.loc
            else:
                distance = INTERVAL * b.next_stop - b.loc
            delta_t = round(distance / b.speed, 2)  # 到下一站还需要的时间
            b.waiting_event = self.now + delta_t

    """
        bus_serve_state_list:
        0: 等待到站事件，随后完成驻站，等待滞站开始事件
        1：等待滞站开始事件后开始计算action，随后完成滞站，等待离站事件
        2：等待离站事件，离站开始后进入running,等待到站事件
    """

    def dy_factor(self, arr_rate):
        if 3 * 60 * 60 <= self.now <= 3.5 * 60 * 60:
            arr_rate *= (self.now - 3 * 60 * 60) / (0.5 * 60 * 60) + 1
        elif 3.5 * 60 * 60 <= self.now <= 4 * 60 * 60:
            arr_rate *= (4 * 60 * 60 - self.now) / (0.5 * 60 * 60) + 1

        return arr_rate

    def sim(self):
        if self.now > self.sim_horizon:
            return -1
        waiting_events = []
        # 扫描公交车，确定下一个等待的事件
        for b in self.busses:
            waiting_events.append(b.waiting_event)
        # 找到时间最早的事件推进
        bus_id = np.argmin(waiting_events)
        bus = self.busses[bus_id]
        last_time = self.now
        self.now = min(waiting_events)
        delta_time = round(self.now - last_time, 2)
        # update other buses info, 后面会更新当前事件的bus
        for b_ in self.busses:
            if bus_id == b_.bus_id:
                continue
            else:
                if bus.next_serve == 0:
                    b_.loc += round(delta_time * b_.speed, 2)
        # arrive
        if bus.next_serve == 0:
            # 计算状态信息  马上到哪个站
            stop_id = self.busses[bus_id].next_stop
            stop = self.stops[stop_id]
            bus.loc = stop.loc
            al_bus_num = len(stop.serve_bus)
            # log arrive
            bus.log_time.append(self.now)
            if stop_id == 0:
                bus.log_loc.append(self.stop_num)
            else:
                bus.log_loc.append(bus.loc / INTERVAL)
            bus.log_load.append(bus.load)

            arr_rate = stop.arr_rate
            if stop_id == 1:
                arr_rate = self.dy_factor(arr_rate)
            add_wait_num = math.floor((self.now - stop.over_dwell_time) * arr_rate / 2)
            add_wait_num = max(0, add_wait_num)
            aver_wait_time = round((self.now - stop.over_dwell_time) * add_wait_num / 2
                                   + stop.wait_num * (self.now - stop.over_dwell_time), 2)
            stop.wait_num += add_wait_num  # 在站点等待乘客数
            bus.log_wait.append(stop.wait_num)
            bus.log_wait_time.append(aver_wait_time)
            # 计算前后发车间隔
            f_loc = self.busses[bus.forward].loc
            b_loc = self.busses[bus.backward].loc
            if f_loc < bus.loc:
                bus.forward_h = round((INTERVAL * self.stop_num - bus.loc) + f_loc, 1)
            else:
                bus.forward_h = round(abs(f_loc - bus.loc), 1)
            if b_loc > bus.loc:
                bus.backward_h = round((INTERVAL * self.stop_num - b_loc) + bus.loc, 1)
            else:
                bus.backward_h = round(abs(bus.loc - b_loc), 1)

            # 计算state
            bus.state[0] = bus.forward_h/200
            bus.state[1] = bus.backward_h/200
            bus.state[2] = round(bus.load / 10, 2)
            bus.state[3] = round(stop.wait_num / 10, 1)  # 状态：到站等待乘客数量
            bus.state[4] = stop_id

            # # 计算reward: 里面的指标考虑归一化处理  是r(t-1)
            # reward1 = math.exp(-abs(bus.state[0] - bus.state[1])) * 2
            # reward2 = math.exp(-bus.hold_time / 60)
            # reward3 = math.exp(-bus.refuse_num/10)
            # reward4 = math.exp(-bus.crowded)
            #
            # reward = reward1 + reward2 + reward3 + reward4
            # # reward = reward1 + reward2
            # bus.reward = np.array(reward)

            sig_mean_f = 0
            sig_mean_b = 0
            sig_var_f = 0
            sig_var_b = 0
            sig_f = []
            sig_b = []

            # for bus0 in self.busses:
            #     sig += abs(
            #         (bus0.forward_h - bus0.backward_h) * INTERVAL / 10 / (bus0.forward_h + bus0.backward_h + 1) ** 2)
            for bus0 in self.busses:
                sig_f.append(bus0.forward_h / INTERVAL)
                sig_b.append(bus0.backward_h / INTERVAL)
            sig_mean_f, sig_var_f = np.mean(sig_f), np.var(sig_f)
            sig_mean_b, sig_var_b = np.mean(sig_b), np.var(sig_b)
            loads = 0
            for bus1 in self.busses:
                if 15 <= bus1.load <= 40:
                    score = 1
                elif 40 < bus1.load <= 60:
                    score = 2
                elif bus1.load < 15:
                    score = 4
                else:
                    score = 3
                loads += score
            if 20 <= bus.load <= 40:
                load = 1
            elif 40 < bus.load <= 60:
                load = 2
            elif bus.load < 20:
                load = 4
            else:
                load = 3

            refuse = (bus.refuse_num + 1)
            reward1 = math.exp(-abs(bus.forward_h - bus.backward_h) / abs(bus.forward_h + bus.backward_h + 1)
                               * abs(bus.forward_h - 2 * INTERVAL) * abs(bus.backward_h - 2 * INTERVAL) / INTERVAL ** 2/2)
            reward2 = math.exp(-30/(bus.hold_time+1))
            reward3 = math.exp(-1 / refuse)
            reward4 = math.exp(-load / 72)
            reward5 = math.exp(
                -(abs(sig_var_f / sig_mean_f + sig_mean_f) * abs(sig_var_b / sig_mean_b + sig_mean_b)) ** 0.5 / 10)
            reward6 = math.exp(-1 / (al_bus_num + 1))

            bus.reward1 = reward1
            bus.reward2 = reward2
            bus.reward3 = reward3

            bus.reward4 = reward4
            bus.reward5 = reward5
            bus.reward6 = reward6
            # reward5 = math.exp()

            # reward = 20 * reward1 + 10 * reward2 + reward5  # + 2 * reward3 + reward4
            # reward = reward1 + reward2

            # reward = reward5 * reward1 * reward3 * reward4 * 100  # * reward2
            # 10 * (5 * reward1 + 3 * reward5 - 3 * reward3 - 3 * reward6 - 0.5 * reward2)
            reward = 20 * (5 * reward1 + 3 * reward5 - 3 * reward3 - 3 * reward6 - 0.7 * reward2)  # + 20 * reward6
            if reward <= 0:
                reward = 0.01
            reward_ = np.array([reward])
            bus.reward = reward_

            self.control_queue.append(bus_id)
            bus.waiting_event = self.now + 0.1
            bus.next_serve = 1

            return 1

        if bus.next_serve == 1:
            stop_id = self.busses[bus_id].next_stop
            stop = self.stops[stop_id]
            stop.serve_bus.append(bus_id)
            serve_bus_num = len(stop.serve_bus)
            hold_t = round(bus.action[0] * 120, 2)
            if hold_t < 10:
                bus.hold_time = 0
            else:
                bus.hold_time = hold_t

            bus.load_rate = 1

            if serve_bus_num > 1:
                stop.bunching = True
            # 下一站是终点站和起始站
            if stop_id == 0:
                alight_num = bus.load  # 全部下车
                bus.alight_list = [0 for _ in range(12)]  # 目的地列表清零
                self.trip.append([bus_id, bus.log_time, bus.log_loc, bus.log_load, bus.log_hold, bus.log_for
                                     , bus.log_back, bus.log_wait, bus.log_wait_time])
                bus.log_time = []
                bus.log_loc = []
                bus.log_load = []
                bus.log_hold = []
                bus.log_for = []
                bus.log_back = []
                bus.log_wait = []
                bus.log_wait_time = []
                bus.load = 0
                alight_time = round(alight_num * bus.alight_rate)  # 计算下车时间
                bunching_rate = 1 if not stop.bunching else (1 / serve_bus_num)
                # 新来乘客取整,如果站点还有其他车，只有一半人上车
                # add_wait_num = math.floor((self.now - stop.over_dwell_time) * stop.arr_rate)
                # add_wait_num = max(0, add_wait_num)
                wait_num = stop.wait_num  # 在站点等待乘客数
                part_pax_num = math.floor(wait_num * bunching_rate)
                allow_num = math.floor(bus.capacity * bus.load_rate)
                # 为每个乘客随机选择目的地
                if part_pax_num <= allow_num:
                    board_num = part_pax_num
                    wait_num -= part_pax_num
                    bus.refuse_num = 0
                else:
                    board_num = allow_num
                    refuse_num = part_pax_num - allow_num  # decision variable
                    wait_num = wait_num - part_pax_num + refuse_num
                    bus.refuse_num = refuse_num
                for i in range(board_num):
                    stop_index = random.randint(stop_id + 1, 12)  # 后面的N/2个站点
                    if stop_index == 12:  # 目的地是终点站
                        stop_index = 0
                    bus.alight_list[stop_index] += 1
                bus.load = board_num
                board_time = round(board_num * stop.board_rate)
                dwell_time = round(max(0, alight_time + board_time))  # 到终点站是先下车再上车
                stop.wait_num = wait_num
                if bus.load < 0:
                    print("neg error2")
            else:
                # 计算上车人数
                bunching_rate = 1 if not stop.bunching else (1 / serve_bus_num)
                # add_wait_num = math.floor((self.now - stop.over_dwell_time) * stop.arr_rate)  # 取整
                # add_wait_num = max(0, add_wait_num)
                wait_num = stop.wait_num
                part_pax_num = math.floor(wait_num * bunching_rate)
                alight_num = bus.alight_list[stop_id]
                bus.alight_list[stop_id] = 0
                bus.load -= alight_num
                # allow_num = min(bus.capacity - bus.load, math.floor(bus.capacity * bus.load_rate))
                allow_num = min(bus.capacity - bus.load, bus.capacity)
                # 计算允许上车人数和拒载人数 ..没计算alight_num
                if part_pax_num <= allow_num:
                    board_num = part_pax_num
                    wait_num -= part_pax_num
                    bus.refuse_num = 0
                else:
                    board_num = allow_num
                    refuse_num = part_pax_num - allow_num  # decision variable
                    bus.refuse_num = refuse_num
                    if refuse_num > 0:
                        print("refuse:" + str(refuse_num))
                    wait_num = wait_num - part_pax_num + refuse_num
                for i in range(board_num):
                    stop_index = random.randint(stop_id + 1, 12)
                    if stop_index == 12:  # 目的地是终点站
                        stop_index = 0
                    bus.alight_list[stop_index] += 1
                bus.load = bus.load + board_num
                board_time = round(board_num * stop.board_rate, 2)
                dwell_time = max(0, board_time)
                stop.wait_num = wait_num
                if bus.load < 0:
                    print("neg error1")
            bus.crowded = round(max(0, bus.load - bus.capacity * 0.8) / 10)
            over_dwell_time = round(dwell_time + self.now, 2)
            stop.over_dwell_time = over_dwell_time
            # bus.waiting_event = round(max(over_dwell_time, bus.hold_time), 2)
            bus.waiting_event = round(over_dwell_time + bus.hold_time, 2)
            # 公交车下一个事件
            bus.next_serve = 2
            # bus.hold_time = 0
            return 2

        # 离站，更新站点和公交车的信息，将arr_list[i]置为下次到站的时间
        if bus.next_serve == 2:
            stop_id = bus.next_stop
            bus.pass_stop = stop_id
            stop = self.stops[stop_id]
            bus.loc = stop.loc
            if stop.serve_bus[0] == bus_id:
                stop.serve_bus.popleft()
            else:
                bus.waiting_event += 0.01
                return 3
            if stop_id == self.stop_num - 1:
                bus.next_stop = 0
            else:
                bus.next_stop += 1
            stop.bus_dep_time = self.now

            # 记录时间和位置、load + 公交站点的记录
            bus.log_loc.append(bus.loc / INTERVAL)
            bus.log_time.append(self.now)
            bus.log_load.append(bus.load)
            bus.log_hold.append(bus.hold_time)
            bus.log_for.append(bus.forward_h)
            bus.log_back.append(bus.backward_h)
            interval_time = np.array(np.random.normal(mu, sigma, 1))
            bus.waiting_event = self.now + round(interval_time[0], 2)
            # bus.waiting_event = self.now + INTERVAL / bus.speed
            bus.next_serve = 0
            return 3
