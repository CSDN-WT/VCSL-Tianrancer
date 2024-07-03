# 自定义环境实现racetrack，DDPG+Pytorch，没采用stable-baselines3库
'''
import math
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import pprint
import highway_env

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        # uniform_将tensor用从均匀分布中抽样得到的值填充。参数初始化
        self.linear3.weight.data.uniform_(-init_w, init_w)
        # 也用用normal_(0, 0.1) 来初始化的，高斯分布中抽样填充，这两种都是比较有效的初始化方式
        self.linear3.bias.data.uniform_(-init_w, init_w)
        # 其意义在于我们尽可能保持 每个神经元的输入和输出的方差一致。
        # 使用 RELU（without BN） 激活函数时，最好选用 He 初始化方法，将参数初始化为服从高斯分布或者均匀分布的较小随机数
        # 使用 BN 时，减少了网络对参数初始值尺度的依赖，此时使用较小的标准差(eg：0.01)进行初始化即可

        # 但是注意DRL中不建议使用BN

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3,
                 decay_period=10000):  # decay_period要根据迭代次数合理设置
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        # 将经过tanh输出的值重新映射回环境的真实值内
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        # 因为激活函数使用的是tanh，这里将环境输出的动作正则化到（-1，1）

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


class DDPG(object):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(DDPG, self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 28
        self.gamma = 0.99
        self.min_value = -np.inf
        self.max_value = np.inf
        self.soft_tau = 2e-2
        self.replay_buffer_size = 8000
        self.value_lr = 5e-3
        self.policy_lr = 5e-4

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.value_criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def ddpg_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, self.min_value, self.max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )


env = gym.make("Myracetrack-v0")  # 自定义的环境，与自带的racetrack环境相同，目前在学习如何自定义自己的环境
env.configure(
    {
        "observation": {
            "type": "OccupancyGrid",
            "features": ['presence', 'on_road', "vx", "vy"],
            # "features_range": {
            # "x": [-100, 100],
            # "y": [-100, 100],
            # "vx": [-20, 20],
            # "vy": [-20, 20]},
            "grid_size": [[-6, 6], [-9, 9]],
            "grid_step": [3, 3],  # 每个网格的大小
            "as_image": False,
            "align_to_vehicle_axes": True
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True
        },
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "duration": 500,
        "collision_reward": -10,
        "lane_centering_cost": 6,
        "action_reward": -0.3,
        "controlled_vehicles": 1,
        "other_vehicles": 5,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.5, 0.5],
        "scaling": 7,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    })

env.reset()
env = NormalizedActions(env)

ou_noise = OUNoise(env.action_space)

state_dim = env.observation_space.shape[2] * env.observation_space.shape[1] * env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print("状态维度" + str(state_dim))
print("动作维度" + str(action_dim))
# print(env.action_space)
hidden_dim = 256

ddpg = DDPG(action_dim, state_dim, hidden_dim)

max_steps = 250
rewards = []
batch_size = 32
VAR = 1  # control exploration

for step in range(max_steps):
    print("================第{}回合======================================".format(step + 1))
    state = env.reset()
    state = torch.flatten(torch.tensor(state))
    ou_noise.reset()
    episode_reward = 0
    done = False
    st = 0

    while not done:
        action = ddpg.policy_net.get_action(state)
        # print(action)
        action[0] = np.clip(np.random.normal(action[0], VAR), -1, 1)  # 在动作选择上添加随机噪声
        action[1] = np.clip(np.random.normal(action[1], VAR), -1, 1)  # 在动作选择上添加随机噪声
        # action = ou_noise.get_action(action, st)
        next_state, reward, done, _ = env.step(action)  # 奖励函数的更改需要自行打开安装的库在本地的位置进行修改
        next_state = torch.flatten(torch.tensor(next_state))
        if reward == 0.0:  # 车辆出界，回合结束
            reward = -10
            done = True
        ddpg.replay_buffer.push(state, action, reward, next_state, done)

        if len(ddpg.replay_buffer) > batch_size:
            VAR *= .9995  # decay the action randomness
            ddpg.ddpg_update()

        state = next_state
        episode_reward += reward
        env.render()
        st = st + 1

    rewards.append(episode_reward)
    print("回合奖励为：{}".format(episode_reward))
env.close()

plt.plot(rewards)
plt.savefig(r'E:\highway1\DDPG+race\reward.jpg')
'''


# 用了stable-baselines3库

import numpy as np
import pandas as pd
import csv
import gym
import highway_env
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image,Video
from highway_env.road.lane import CircularLane

csv_file_path = r"E:\highway1\Myracetrack_v1\part_2\train\value\DDPG.csv"
csv_file_path1 = "E:\highway1\Myracetrack_v1\part_2\evaluation\DDPG(1).csv"


env = gym.make("Myracetrack-v0")
env.configure(
    {
        "observation": {
            "type": "OccupancyGrid",
            "features": ['presence', 'on_road', "vx", "vy"],
            # "features_range": {
            # "x": [-100, 100],
            # "y": [-100, 100],
            # "vx": [-20, 20],
            # "vy": [-20, 20]},
            "grid_size": [[-6, 6], [-9, 9]],
            "grid_step": [3, 3],  # 每个网格的大小
            "as_image": False,
            "align_to_vehicle_axes": True
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True
        },
        "controlled_vehicles": 1,
        "other_vehicles": 1,
        "screen_width": 1600,  # 屏幕宽度
        "screen_height": 1000,  # 屏幕高度
        "duration": 300,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    })

env.reset()

class ImageRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ImageRecorderCallback, self).__init__(verbose)

    def _on_step(self):
        image = self.training_env.render(mode="rgb_array")
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        self.logger.record("trajectory/image", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))
        return True
ImageRecorderCallback = ImageRecorderCallback()

# 存储车辆角度、速度、位置等信息

class InfoCallback:
    def __init__(self):
        # 初始化列表以存储速度和转向角信息
        self.speed_list = [] # 速度
        self.heading_list = [] # 横摆角
        self.steering_list = [] # 转向角
        self.accelerate_list = [] # 加速度
        self.x_list = [] # x位置
        self.y_list = [] # y位置
        self.lateral_list = [] # 距离中心线的偏移
        self.beta_list = [] # 滑移角
        self.yaw_list = [] # 航向角
        self.rel_yaw_list = [] #航向对比
        self.lane_heading_list = [] # 车道航向


    def callback(self, local, global_):
        # 获取当前环境的信息
        info = local["info"]

        # 获取车辆速度和转向角信息并存储
        speed = info["speed"]
        heading = info["vehicle heading"]
        steering = info["steering"]
        accelerate = info["accelerate"]
        x = info["x"]
        y = info["y"]
        lateral = info["lateral"]
        beta = info["beta"]
        yaw = info["yaw"]
        rel_yaw = info["rel_yaw"]
        lane_heading = info["lane_heading"]

        self.speed_list.append(speed)
        self.heading_list.append(heading)
        self.steering_list.append(steering)
        self.accelerate_list.append(accelerate)
        self.x_list.append(x)
        self.y_list.append(y)
        self.lateral_list.append(lateral)
        self.beta_list.append(beta)
        self.yaw_list.append(yaw)
        self.rel_yaw_list.append(rel_yaw)
        self.lane_heading_list.append(lane_heading)


Infocallback1 = InfoCallback()


######## 模型训练 ##########

# model = DDPG("MlpPolicy",
#              env,
#              verbose=1,
#              tensorboard_log = "E:\highway1\Myracetrack_v2\DDPG"
#              )
# model.learn(total_timesteps=100000,callback=ImageRecorderCallback)
# model.save("E:\highway1\Myracetrack_v2\model\DDPG_obstacle_model")

###### 模型测试 #######

# model = DDPG.load("E:\highway1\Myracetrack_v3\model\DDPG_model", env=env)
#
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render()

####### 模型评估 #######

model = DDPG.load("E:\highway1\Myracetrack_v1\model\DDPG_model.zip", env=env)
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    deterministic=True,
    callback=Infocallback1.callback,
    render=True,
    n_eval_episodes=1)
print('mean_reward:', mean_reward, 'std_reward:', std_reward)

### 存储数据

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Speed", "Heading", "Steering", "Accelerate", "X", "Y", "Lateral", "Beta", "Yaw", "Rel_Yaw","Lane_heading"])
    writer.writerows(zip(Infocallback1.speed_list, Infocallback1.heading_list,
                         Infocallback1.steering_list, Infocallback1.accelerate_list,
                         Infocallback1.x_list, Infocallback1.y_list,
                         Infocallback1.lateral_list, Infocallback1.beta_list,
                         Infocallback1.yaw_list, Infocallback1.rel_yaw_list,Infocallback1.lane_heading_list))

print(f"Data saved to {csv_file_path}")

with open(csv_file_path1, mode='w', newline='') as file1:
    writer1 = csv.writer(file1)
    writer1.writerow(["mean_reward","std_reward"])
    writer1.writerow([mean_reward,std_reward])

print(f"Data saved to {csv_file_path1}")






