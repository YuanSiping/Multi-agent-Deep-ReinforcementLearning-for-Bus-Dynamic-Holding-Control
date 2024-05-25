import numpy as np

INTERVAL = 2000  # 1km
MAX_ARR = 4 * 60 * 60


class Bus:
    """
    bus_id:            bus id
    is_serving:        whether bus is serving a stop  1:stop for service 0:on route -1:holding -2: not emit
    next_serve:        arrive: 0 holding:1 dep:2
    """
    """
            bus_serve_state_list:
            0: 等待到站事件，随后完成驻站，等待滞站开始事件
            1：等待滞站开始事件后开始计算action，随后完成滞站，等待离站事件
            2：等待离站事件，离站开始后进入running,等待到站事件
    """

    def __init__(self, bus_id, loc, pass_stop, capacity=72, state_dim=5, action_dim=2, alight_rate=1.5,
                 dep_time=0):
        self.action = np.zeros(action_dim)
        self.action_dim = action_dim
        self.a_logprob = np.zeros(action_dim)
        self.alight_rate = alight_rate
        self.alight_list = [0 for _ in range(12)]

        self.bus_id = bus_id

        self.capacity = capacity
        self.crowded = 0  # 拥挤度【0,0.25,0.5,1】

        self.dep_time = dep_time  # 发车时间
        self.decide_time = 0

        # headway
        self.headway = 2 * INTERVAL
        self.forward_h = 2 * INTERVAL
        self.backward_h = 2 * INTERVAL
        self.forward = None
        self.backward = None
        self.hold_time = 0
        self.hold_time_sum = 0  # 在horizon里hold time总和

        self.load = 0
        self.load_rate = 1
        self.loc = loc  # 公交车的位置
        self.log_time = []
        self.log_loc = []
        self.log_load = []
        self.log_hold = []
        self.log_for = []
        self.log_back = []
        self.log_wait = []
        self.log_wait_time = []
        self.log_hold = []
        self.next_serve = 0

        self.reward = np.zeros(1)
        self.reward1 = 0
        self.reward2 = 0
        self.reward3 = 0
        self.reward4 = 0
        self.reward5 = 0
        self.reward6 = 0

        self.refuse = []  # [num, loc]
        self.refuse_num = 0
        self.rl_first = 1
        self.speed = 10  # m/s  换算成km/h*3.6
        self.serve_level = []  # 考虑要不要记录
        # self.serve_state = 0
        self.sim_state = 0
        self.state = np.zeros(state_dim)
        self.state_last = np.zeros(state_dim)
        self.state_dim = state_dim  # [for_headway,back_headway, load, waiting][satisfy]

        self.pass_stop = pass_stop
        self.next_stop = None

        self.waiting_event = 0

        # self.next_stop = next_stop
        # self.next_stop_loc = next_stop  # 位置可以说是1.5站 即1站和2站之间
        #
        # self.waiting_event = 0  # 下一个事件时间
