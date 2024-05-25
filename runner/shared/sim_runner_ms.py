"""
# @Time    : 2022/05/17 16:38
# @Author  : Wang
# @File    : sim_runner.py
"""
import csv
import matplotlib.pyplot as plt
import mindspore
import numpy as np
import torch

from envs.env import Env
from utils.replay_buffer import ReplayBuffer

from algorithms_ms.algorithm_ms.actor_critic_ms import PPO
from algorithms_ms.algorithm_ms.normalization import Normalization, RewardScaling, StateNorm

# from algorithms.algorithm.actor_critic import PPO
# from algorithms.algorithm.normalization import Normalization, RewardScaling, StateNorm


# from torch.utils.tensorboard import SummaryWriter

# plt.rcParams["figure.figsize"] = (6, 6)
plt.rc("font", family="Times New Roman", size=12)

# RL
# num_episodes = 20
updateStep = 1

color = {
    0: "r",
    1: "y",
    2: "b",
    3: "g",
    4: "m",
    5: "c"
}


class Runner:
    def __init__(self, args):
        self.args = args
        self.env = None

    def run(self, warm_time):
        # 初始化环境
        env = Env(self.args)
        agent = PPO(self.args)
        replay_buffer = ReplayBuffer(self.args)
        episode_rewards = []
        train_flag = True
        state_memory = []
        # state_norm = Normalization(shape=self.args.state_dim)
        state_norm = None
        reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)
        # runtime = list()
        norm_flag = 1
        if train_flag:

            for i in range(self.args.num_episodes):
            # for i in range(2):
                env.reset()
                if self.args.use_reward_scaling:
                    reward_scaling.reset()
                # self.reset_memory()
                episode_reward = 0
                episode_reward1 = 0
                episode_reward2 = 0
                episode_reward3 = 0
                episode_reward4 = 0
                episode_reward5 = 0
                episode_reward6 = 0
                print(str(i) + "-----------")
                while True:
                    # print("Sim start ====")
                    is_sim_over = env.sim()
                    # hold control
                    if is_sim_over == 1:
                        # record state i/reward i-1
                        # 下一次到站才能知道 reward and next state
                        if not env.control_queue:
                            print("error hold empty")
                        if env.now < warm_time:
                            bus_id = env.control_queue.popleft()
                            bus = env.busses[bus_id]
                            state_memory.append(bus.state)
                            continue
                        else:
                            if norm_flag:
                                state_norm = StateNorm(state_memory)
                                norm_flag = 0
                            bus_id = env.control_queue.popleft()
                            bus = env.busses[bus_id]
                            # append 上一次的buffer,清空并记录这次的a,a_logprob
                            s_ = bus.state
                            # r = reward_scaling(bus.reward)
                            r = bus.reward
                            s_ = state_norm.norm(s_)
                            # start = time.perf_counter()
                            if bus.rl_first == 0:
                                # bus.buffer[3], bus.buffer[4] = r, s
                                replay_buffer.store(bus.state_last, bus.action, bus.a_logprob, r, s_)
                                episode_reward += r[0]
                                episode_reward1 += bus.reward1
                                episode_reward2 += bus.reward2
                                episode_reward3 += bus.reward3
                                episode_reward4 += bus.reward4
                                episode_reward5 += bus.reward5
                                episode_reward6 += bus.reward6
                            # 开始下一次的记录s a

                            # s = state_norm(s)
                            s = s_
                            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
                            bus.state_last = s
                            bus.action = a
                            bus.a_logprob = a_logprob
                            bus.rl_first = 0

                            # When the number of transitions in buffer reaches batch_size,then update
                            if replay_buffer.count == self.args.batch_size:
                                agent.update(replay_buffer, bus.decide_time)
                                replay_buffer.count = 0
                            # end = time.perf_counter()
                            # runtime.append(["time", round((end - start) * 1000)])

                            # Evaluate the policy every 'evaluate_freq' steps
                            # 后面改
                    if is_sim_over < 0:
                        break
                episode_rewards.append(episode_reward)
                print("episode" + str(i) + " reward:-------" + str(episode_reward) + "--\n")
                print("episode" + str(i) + " reward1:-------" + str(episode_reward1) + "--\n")
                print("episode" + str(i) + " reward2:-------" + str(episode_reward2) + "--\n")
                print("episode" + str(i) + " reward3:-------" + str(episode_reward3) + "--\n")
                print("episode" + str(i) + " reward4:-------" + str(episode_reward4) + "--\n")
                print("episode" + str(i) + " reward5:-------" + str(episode_reward5) + "--\n")
                print("episode" + str(i) + " reward6:-------" + str(episode_reward6) + "--\n")

                if i == self.args.num_episodes - 1:
                # if i == 1:
                    plt.figure(dpi=300, figsize=(5, 5))
                    # 轨迹图
                    for trip in env.trip:
                        tr_color = color[trip[0]]
                        plt.plot(trip[1], trip[2], tr_color + "-", label="loc", linewidth=1)
                        # plt.subplot(3, 3, trip[0]+4)
                        # plt.plot(trip[1], trip[3], tr_color + "-", label="load"+str(trip[0]+1))
                    plt.xlabel("Time/s")
                    plt.ylabel("Stop")
                    plt.title("(d)", fontsize=16)
                    plt.savefig('../results/picture/MH.png')

                    # 载客量图
                    plt.cla()
                    for trip in env.trip:
                        tr_color = color[trip[0]]
                        # plt.plot(trip[2], trip[3], tr_color + "-", label="loc", linewidth=1)
                        plt.plot(trip[2][::2], trip[3][::2], "b-", label="loc", linewidth=1)
                        # plt.subplot(3, 3, trip[0]+4)
                        # plt.plot(trip[1], trip[3], tr_color + "-", label="load"+str(trip[0]+1))
                    plt.plot([0, 12], [72, 72], "r-", linewidth=1)
                    plt.xlabel("Distance")
                    plt.ylabel("Load (pax.)")
                    plt.title("(d)", fontsize=16)
                    plt.savefig('../results/picture/MH_load.png')

                    topic = ["bus_id", "bus_time", "bus_loc", "bus_load", "bus_hold", "bus_for",
                             "bus_back", "bus_wait", "bus_wait_time"]
                    with open('../results/data/csv_Save.csv', 'w', newline='') as fp:
                        wr = csv.writer(fp)
                        wr.writerow(topic)
                        wr.writerows(env.trip)
                    np.save('../results/data/reward.npy', episode_rewards)
                    np.save('../results/data/critic_loss.npy', agent.critic_loss)
                    np.save('../results/data/state_mean.npy', state_norm.mean)
                    np.save('../results/data/state_std.npy', state_norm.std)
                    print("state mean: "+str(state_norm.mean))
                    print("state std: " + str(state_norm.std))
                    # b=np.load('a.npy')
                    # critic loss
                    plt.cla()
                    index_c = [i + 1 for i in range(len(agent.critic_loss))]
                    plt.plot(index_c, agent.critic_loss, "b-", label="critic loss", linewidth=1)
                    # plt.subplot(3, 3, trip[0]+4)
                    # plt.plot(trip[1], trip[3], tr_color + "-", label="load"+str(trip[0]+1))
                    plt.xlabel("iter")
                    plt.ylabel("Loss")
                    plt.savefig('../results/picture/critic_loss.png')
                    # reward
                    plt.cla()
                    index_r = [i + 1 for i in range(len(episode_rewards))]
                    plt.plot(index_r, episode_rewards, "b-", label="reward", linewidth=1)
                    plt.xlabel("iter")
                    plt.ylabel("reward")
                    plt.savefig('../results/picture/reward.png')

            # torch.save(agent.actor.state_dict(), '../results/parse/actor.params')
            # torch.save(agent.critic.state_dict(), '../results/parse/critic.params')

            mindspore.save_checkpoint(agent.actor, "../results/parse/actor_ms.ckpt")
            mindspore.save_checkpoint(agent.critic, "../results/parse/critic_ms.ckpt")

        else:
            # agent.actor.load_state_dict(torch.load('../results/parse/actor.params'))
            # agent.critic.load_state_dict(torch.load('../results/parse/critic.params'))

            param_dict_actor = mindspore.load_checkpoint("../results/parse/actor_ms.ckpt")
            param_dict_critic = mindspore.load_checkpoint("../results/parse/critic_ms.ckpt")
            mindspore.load_param_into_net(agent.actor, param_dict_actor)
            mindspore.load_param_into_net(agent.critic, param_dict_critic)

            env.reset()
            if self.args.use_reward_scaling:
                reward_scaling.reset()
            episode_reward = 0
            episode_reward1 = 0
            episode_reward2 = 0
            episode_reward3 = 0
            episode_reward4 = 0
            episode_reward5 = 0
            episode_reward6 = 0
            mean = np.load('../results/data/state_mean.npy')
            std = np.load('../results/data/state_std.npy')
            print(str(0) + "-----------")
            while True:
                # print("Sim start ====")
                is_sim_over = env.sim()
                # hold control
                if is_sim_over == 1:
                    # record state i/reward i-1
                    # 下一次到站才能知道 reward and next state
                    if not env.control_queue:
                        print("error hold empty")
                    if env.now < warm_time:
                        continue
                    else:
                        bus_id = env.control_queue.popleft()
                        bus = env.busses[bus_id]
                        # append 上一次的buffer,清空并记录这次的a,a_logprob
                        s_ = bus.state
                        r = reward_scaling(bus.reward)
                        # start = time.perf_counter()
                        if bus.rl_first == 0:
                            # bus.buffer[3], bus.buffer[4] = r, s
                            episode_reward += r[0]
                            episode_reward1 += bus.reward1
                            episode_reward2 += bus.reward2
                            episode_reward3 += bus.reward3
                            episode_reward4 += bus.reward4
                            episode_reward5 += bus.reward5
                            episode_reward6 += bus.reward6
                        # 开始下一次的记录s a
                        s = (s_ - mean) / (std + 1e-5)
                        a = agent.evaluate(s)  # Action and the corresponding log probability
                        bus.state_last = s
                        bus.action = a

                        bus.rl_first = 0

                        # Evaluate the policy every 'evaluate_freq' steps
                        # 后面改
                if is_sim_over < 0:
                    break
            episode_rewards.append(["reward", episode_reward])
            print("episode" + "0" + " reward:-------" + str(episode_reward) + "--\n")
            print("episode" + "0" + " reward1:-------" + str(episode_reward1) + "--\n")
            print("episode" + "0" + " reward2:-------" + str(episode_reward2) + "--\n")
            print("episode" + "0" + " reward3:-------" + str(episode_reward3) + "--\n")
            print("episode" + "0" + " reward4:-------" + str(episode_reward4) + "--\n")
            print("episode" + "0" + " reward5:-------" + str(episode_reward5) + "--\n")
            print("episode" + "0" + " reward6:-------" + str(episode_reward6) + "--\n")

            plt.figure(dpi=300, figsize=(5, 5))
            # 轨迹图
            for trip in env.trip:
                tr_color = color[trip[0]]
                plt.plot(trip[1], trip[2], tr_color + "-", label="loc", linewidth=1)
                # plt.subplot(3, 3, trip[0]+4)
                # plt.plot(trip[1], trip[3], tr_color + "-", label="load"+str(trip[0]+1))
            plt.xlabel("Time/s")
            plt.ylabel("Stop")
            plt.title("(d) MH", fontsize=16)
            plt.savefig('../results/picture/MH2.png')

            # 载客量图
            plt.cla()
            for trip in env.trip[6:]:
                tr_color = color[trip[0]]
                # plt.plot(trip[2], trip[3], tr_color + "-", label="loc", linewidth=1)
                plt.plot(trip[2][::2], trip[3][::2], "b-", label="loc", linewidth=1)
                # plt.subplot(3, 3, trip[0]+4)
                # plt.plot(trip[1], trip[3], tr_color + "-", label="load"+str(trip[0]+1))
            plt.plot([0, 11], [72, 72], "r-", linewidth=1)
            plt.xlabel("Distance")
            plt.ylabel("Load (pax.)")
            plt.title("(d) MH", fontsize=16)
            plt.savefig('../results/picture/MH2_load.png')

            topic = ["bus_id", "bus_time", "bus_loc", "bus_load", "bus_hold", "bus_for",
                     "bus_back", "bus_wait", "bus_wait_time"]
            with open('../results/data/MH2_csv_Save.csv', 'w', newline='') as fp:
                wr = csv.writer(fp)
                wr.writerow(topic)
                wr.writerows(env.trip)
            # critic loss
            plt.cla()
            for j, loss in enumerate(agent.critic_loss):
                plt.plot(j + 1, loss, "bo", label="loss", ms=0.5)
                # plt.subplot(3, 3, trip[0]+4)
                # plt.plot(trip[1], trip[3], tr_color + "-", label="load"+str(trip[0]+1))
            plt.xlabel("iter")
            plt.ylabel("Loss")
            plt.savefig('../results/picture/critic_loss_t.png')
            # reward
            plt.cla()
            for k, reward in enumerate(episode_rewards):
                plt.plot(k + 1, reward[1], "bo", label="reward")
            plt.xlabel("iter")
            plt.ylabel("reward")
            plt.savefig('../results/picture/reward_t.png')
