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
import carla

import numpy as np

from stable_baselines.common.policies import CnnPolicy,CnnLstmPolicy,MlpLstmPolicy
from stable_baselines import TRPO,PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

#from stable_baselines.common import SubprocVecEnv
def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 150,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 5,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'roundabout',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 800,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.5,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 3.0,  # threshold for out of lane
    'desired_speed': 5,  # desired speed (m/s)
    'max_ego_spawn_times': 10,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
    'obs_single':True, #return a single key of the obs_space dict for obs space
    'obs_name':'state',#basic options have 'camera' ; 'lidar' ; 'birdeye' ; 'state'
    'use_control':True,
    'add_state':False,
    'random_seed':10
  }

  # Set gym-carla environment
  env = gym.make('carla-v0',params=params)
  env = env.unwrapped
  model = PPO2(MlpLstmPolicy,env,gamma=0.995,nminibatches=1)
  #model = DQN.load("C:/Users/24829/Desktop/rl_model_50000_steps.zip",env=env)
  checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_logs_xunhuan_ppo/')
  #callback = CallbackList([checkpoint_callback, eval_callback])
  model.tensorboard_log="./tensorboard_dqn_log/PPO2-MLP/"
  model.learn(total_timesteps=100000,callback=checkpoint_callback,reset_num_timesteps=False) 
  '''
  car_num = [50,100,200] #generate vehicles for each curriculum
  training_step =[(1,10),(4,233),(10,200)]
  for i in range(3):
    vehicle = car_num[i]
    step,seed =training_step[i]
    print("=======Iteration {},car number:{}========".format(i,vehicle))
    model.env.number_of_vehicles = 100
    #model.env.random_seed = seed

    #model.tensorboard_log="./tensorboard_dqn_log/DQN5/"
    model.learn(total_timesteps=1000*step,reset_num_timesteps=False) 
  #model.save("dqn_carla_7")
  
  
  #model.save("dqn_carla_6")
  print('Model Saved!!!')

 # del model # remove to demonstrate saving and loading
  
  model = DQN.load("dqn_carla_4")
  print('Load model...')
  obs = env.reset()


  while True:
    #env.render(mode='human')
    action,_states = model.predict(obs)
    #print(action)
    #action = [2.0,0.0]
    obs,r,done,info = env.step(action)
    #print(r,info)

    if done:
      obs = env.reset()

'''
if __name__ == '__main__':
  main()