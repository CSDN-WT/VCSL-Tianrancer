import gym
import highway_env
from stable_baselines3 import PPO
import numpy as np
import gym
import highway_env
from stable_baselines3 import TD3
import gym
import highway_env
from stable_baselines3 import SAC
import numpy as np
import pandas as pd
import csv
import gym
import highway_env
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image

csv_file_path = r"E:\highway1\Myracetrack_v1\part_2\train\value\PPO.csv"
csv_file_path1 = "E:\highway1\Myracetrack_v1\part_2\evaluation\PPO(1).csv"

# Create environment
env = gym.make("Myracetrack-v0")

env.configure({
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
        "lateral": True,

    },
    "controlled_vehicles": 1,
    "other_vehicles": 1,
    # "screen_width": 1600,  # 屏幕宽度
    # "screen_height": 1000,  # 屏幕高度
    "duration": 300,
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
        ##########航向对比



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

        self.speed_list.append(speed)
        self.heading_list.append(heading)
        self.steering_list.append(steering)
        self.accelerate_list.append(accelerate)
        self.x_list.append(x)
        self.y_list.append(y)
        self.lateral_list.append(lateral)
        self.beta_list.append(beta)
        self.yaw_list.append(yaw)

Infocallback1 = InfoCallback()

###### 模型训练 ######

# model = PPO('MlpPolicy',
#             env,
#             verbose=1,
#             tensorboard_log="E:\highway1\Myracetrack_v3\PPO")
#
# model.learn(total_timesteps=100000,callback=ImageRecorderCallback)
# model.save("E:\highway1\Myracetrack_v3\model\PPO_obstacle_model")

####### 模型测试 #######
'''
model = PPO.load("E:\highway1\Myracetrack_v3\model\PPO_model", env=env)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    print(reward)
'''
####### 模型评估 #######

model = PPO.load("E:\highway1\Myracetrack_v1\model\PPO_model.zip", env=env)
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    deterministic=True,
    callback=Infocallback1.callback,
    render=True,
    n_eval_episodes=1)
print('mean_reward:', mean_reward, 'std_reward:', std_reward)

# 存储数据

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Speed", "Heading", "Steering", "Accelerate", "X", "Y", "Lateral", "Beta", "Yaw" ])
    writer.writerows(zip(Infocallback1.speed_list, Infocallback1.heading_list,
                         Infocallback1.steering_list, Infocallback1.accelerate_list,
                         Infocallback1.x_list, Infocallback1.y_list,
                         Infocallback1.lateral_list, Infocallback1.beta_list,
                         Infocallback1.yaw_list))

print(f"Data saved to {csv_file_path}")

with open(csv_file_path1, mode='w', newline='') as file1:
    writer1 = csv.writer(file1)
    writer1.writerow(["mean_reward","std_reward"])
    writer1.writerow([mean_reward,std_reward])


print(f"Data saved to {csv_file_path1}")