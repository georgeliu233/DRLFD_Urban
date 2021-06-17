#!/usr/bin/env python
import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import time
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass
import gym
import gym_carla
#import carla
import PIL
import gym
import numpy as np

from stable_baselines import DQN
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from PIL import Image
import matplotlib.pyplot as plt

#from stable_baselines.common import SubprocVecEnv
def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 50,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 5,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'roundabout',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.5,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 10,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
    'obs_single':True, #return a single key of the obs_space dict for obs space
    'obs_name':'birdeye'#basic options have 'camera' ; 'lidar' ; 'birdeye' ; 'state'
  }

  # Set gym-carla environment
  env = gym.make('carla-v0',params=params)
  model = DQN.load("C:/Users/24829/Desktop/gym-carla-master/dqn_logs_xunhuan3/rl_model_100000_steps.zip")
  print('Load model...')
  obs = env.reset()

  #plt.figure()
  
  for _ in range(10000):
    #env.render(mode='human')
    action,_states = model.predict(obs)
    #print(action)
    #action = [2.0,0.0]
    #action = 2
    obs,r,done,info = env.step(action)
    print(r)
    #plt.ion()
    #if i%10==0:
      #img = Image.fromarray((obs*255).astype(np.uint8))
    #plt.imshow((obs*255).astype(np.uint8))
    #plt.show()
    #plt.pause(0.05)
    #plt.clf()
    
    if done:
      obs = env.reset()
 
  
  
if __name__ == '__main__':
  main()