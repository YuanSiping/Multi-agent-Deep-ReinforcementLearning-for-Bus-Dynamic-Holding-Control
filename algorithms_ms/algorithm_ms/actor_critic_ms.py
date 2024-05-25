import math
import numpy as np
import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.experimental.optim as optim
from mindspore.common.initializer import initializer, Orthogonal
from mindspore.dataset import SubsetRandomSampler
import mindspore.ops as ops
from mindspore import Parameter


def orthogonal_init(layer, gain=1.0):
    layer.weight = initializer(Orthogonal(gain=gain), layer.weight.shape, mindspore.float32)
    layer.bias = initializer('zeros', layer.bias.shape, mindspore.float32)


class Actor_Beta(nn.Cell):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Dense(args.state_dim, args.hidden_width)
        self.fc2 = nn.Dense(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Dense(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Dense(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        self.activate_func1 = nn.Tanh()
        self.activate_func2 = nn.Sigmoid()
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def construct(self, s):  # 注意转换成tensor类型
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = ops.softplus(self.alpha_layer(s)) + 1.0
        beta = ops.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.construct(s)
        dist = nn.probability.distribution.Beta(alpha, beta, dtype=mindspore.float32)
        return dist

    def mean(self, s):
        alpha, beta = self.construct(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution

        return mean


class Actor_Gaussian(nn.Cell):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Dense(args.state_dim, args.hidden_width)
        self.fc2 = nn.Dense(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Dense(args.hidden_width, args.action_dim)
        self.log_std = Parameter(
            mindspore.ops.zeros((1, args.action_dim))) # We use 'nn.Paremeter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        # tanh  *2 -1

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def construct(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # 可以换成[0,1]的
        mean = self.max_action * ops.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.construct(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = ops.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = nn.probability.distribution.Normal(mean, std, dtype=mindspore.float32)  # Get the Gaussian distribution
        return dist

class Critic(nn.Cell):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Dense(args.state_dim + args.action_dim, args.hidden_width)
        self.fc2 = nn.Dense(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Dense(args.hidden_width, 1)
        self.activate_func1 = nn.Tanh()  # Trick10: use tanh
        self.activate_func2 = ops.Softplus()

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def construct(self, s, a):
        s_a = ops.cat((s, a), axis=1)
        s_a = self.activate_func1(self.fc1(s_a))
        s_a = self.activate_func1(self.fc2(s_a))
        v_s = self.fc3(s_a)
        return v_s


class CustomSampler(ds.Sampler):
    def __init__(self, num_samples, batch_size):
        self.num_samples = num_samples
        self.batch_size = batch_size

    def __iter__(self):
        indices = np.random.choice(range(self.num_samples), self.num_samples, replace=False).tolist()
        for i in range(0, self.num_samples, self.batch_size):
            yield indices[i:i+self.batch_size]

    def __len__(self):
        return self.num_samples // self.batch_size


class PPO:
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        # self.max_train_steps = args.max_train_steps
        self.sim_horizon = args.sim_horizon
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.num_episodes = args.num_episodes
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)
        self.critic_loss = []

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = optim.Adam(self.actor.trainable_params(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = mindspore.experimental.optim.Adam(self.critic.trainable_params(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = optim.Adam(self.actor.trainable_params(), lr=self.lr_a)
            self.optimizer_critic = optim.Adam(self.critic.trainable_params(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = ops.unsqueeze(mindspore.tensor(s, dtype=mindspore.float32), 0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).asnumpy().flatten()
            a = np.array(a)
        else:
            a = self.actor(s).asnumpy().flatten()
            a = np.array(a)
        return a

    def choose_action(self, s):
        s = ops.unsqueeze(mindspore.tensor(s, dtype=mindspore.float32), 0)
        if self.policy_dist == "Beta":
            dist = self.actor.get_dist(s)
            a = dist._sample()  # Sample the action according to the probability distribution
            a_logprob = dist._log_prob(a)  # The log probability density of the action


        else:
            dist = self.actor.get_dist(s)
            a = dist.sample()  # Sample the action according to the probability distribution

            # a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
            # 这里生成的动作的两个元素都是0-1
            a_logprob = dist.log_prob(a)  # The log probability density of the action

        # flatten()  把数组降到一维，按行排列
        return a.asnumpy().flatten(), a_logprob.asnumpy().flatten()

    def update(self, replay_buffer, episode):
        s, a, a_logprob, r, s_ = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0

        vs = self.critic.construct(s, a)
        alpha, beta = self.actor.construct(s_)
        a_ = alpha / (alpha + beta)
        vs_ = self.critic(s_, a_)
        deltas = r + self.gamma * 1.0 * vs_ - vs
        for delta in reversed(deltas.flatten().asnumpy()):
            gae = delta + self.gamma * self.lamda * gae * 1.0
            adv.insert(0, gae)
        adv = mindspore.Tensor(adv, dtype=mindspore.float32).view(-1, 1)
        v_target = adv + vs
        if self.use_adv_norm:  # Trick 1:advantage normalization
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        def forward_fn1(s, a, a_logprob, adv):
            dist_now = self.actor.get_dist(s)
            dist_entropy = dist_now.entropy().sum(1, keepdims=True)
            a_logprob_now = dist_now.log_prob(a)
            # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
            ratios = ops.exp(a_logprob_now.sum(1, keepdims=True) - a_logprob.sum(1, keepdims=True))  # shape(mini_batch_size X1
            surr1 = ratios * adv  # Only calculate the gradient of 'a_logprob_now' in ratios
            surr2 = ops.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv

            surr = ops.concat((surr1, surr2))
            actor_loss = -mindspore.Tensor.min(surr, axis=0) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
            return actor_loss.mean()

        def forward_fn2(s, a, v_target):
            v_s = self.critic(s, a)
            critic_loss = ops.mse_loss(v_target, v_s)
            return critic_loss

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in CustomSampler(self.batch_size, self.mini_batch_size):
                grad_fn1 = mindspore.value_and_grad(forward_fn1, None, self.optimizer_actor.parameters)
                actor_loss, grads1 = grad_fn1(s[index], a[index], a_logprob[index], adv[index])
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    grads1 = ops.clip_by_norm(grads1, 0.5)
                self.optimizer_actor(grads1)


                # Update critic
                grad_fn2 = mindspore.value_and_grad(forward_fn2, None, self.optimizer_critic.parameters)
                critic_loss, grads2 = grad_fn2(s[index], a[index], v_target[index])
                self.critic_loss.append(critic_loss.item())
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    grads2 = ops.clip_by_norm(grads2, 0.5)
                self.optimizer_critic(grads2)

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(episode)

    # def lr_decay(self, total_steps):  # 改成时刻的也一样
    #     lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
    #     lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
    #     for p in self.optimizer_actor.param_groups:
    #         p['lr'] = lr_a_now
    #     for p in self.optimizer_critic.param_groups:
    #         p['lr'] = lr_c_now

    def lr_decay(self, episode):  # 改成时刻的也一样
        lr_a_now = self.lr_a * (math.floor((1 - episode / self.num_episodes) * 4) + 1) / 4
        lr_c_now = self.lr_c * (math.floor((1 - episode / self.num_episodes) * 4) + 1) / 4
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now