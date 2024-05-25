import numpy as np
from collections import deque

INTERVAL = 500  # 500m


class BusStop:
    """
    stop_id:          bus stop id
    arr_dis:          arrival distribution 0 for poisson 1 for uniform
    schedule_hw:      set schedule arrival interval
    bus_arr_time:     actual bus arrival time list
    bus_arr_interval: actual bus arrival interval list
    arr_rate:         arrival rate
    board_rate:       boarding rate
    tor_cap:          passenger tolerance on the maximum loading number on bus
    wait_num:         number of passenger waiting on the bus stop
    corr_radius:      radius of bus corridor
    loc:              location of bus stop in angle

    wait_time_sum:    total waiting time
    wait_num_sep:     accumulated waiting number between two consecutive arrival of buses
    """

    # arr_dis : arrive distribution. 0 for uniform, 1 for poisson
    def __init__(self, stop_id, arr_rate, loc, board_rate=2.5, arr_dis=0, wait_num=5):
        self.arr_rate = arr_rate
        self.arr_dis = arr_dis

        self.board_rate = board_rate  # person / s
        self.bunching = False  # if len(serve_que)>1: bunching = True,  default=False

        self.loc = loc
        self.next_stop = None
        self.over_dwell_time = 0

        self.stop_id = stop_id
        self.serve_bus = deque()  # 进入公交站的车队列，前一辆车走了，后一辆车才能走

        self.wait_num = wait_num

