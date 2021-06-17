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
import pickle

import numpy as np
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

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

from stable_baselines.common.policies import MlpLstmPolicy,MlpPolicy,MlpLnLstmPolicy
from stable_baselines.td3.policies import CnnPolicy
from stable_baselines import SAC,GAIL,DQN,PPO2,TD3,A2C
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.gail import generate_expert_traj,ExpertDataset
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from SQIL import SQIL_DQN
#from stable_baselines.common import SubprocVecEnv
def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles':100,
    'number_of_walkers':0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 5,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-6.0,-3.0, 0.0, 3.0],  # discrete value of accelerations
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
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
    'obs_single':True, #return a single key of the obs_space dict for obs space
    'obs_name':'birdeye',#basic options have 'camera' ; 'lidar' ; 'birdeye' ; 'state'
    'use_control':True,
    'add_state':False,
    'random_seed':200
  }

  TEST_TIMES=50
  env = gym.make('carla-v0',params=params)
  params_dict={
    'episode_rewards':[],
    'ends':[],
    'epi_length':[],
    'speed':[]
  }
  buf_reward=[]
  buf_speed=[]
  epi=0
  full_epis=0
  obs=env.reset()
  while True:
    obs,r,done,info = env.step(1.0)
    epi+=1
    buf_reward.append(r)
    if epi<=200:
      #print(obs['state'])
      buf_speed.append(info['states'][2])
    if done!=0:
      
      full_epis+=1
      params_dict['episode_rewards'].append(np.sum(buf_reward))
      params_dict['ends'].append(done)
      params_dict['epi_length'].append(epi)
      params_dict['speed'].append(buf_speed)
      buf_reward=[]
      buf_speed=[]
      epi=0
      obs=env.reset()
      if full_epis>=TEST_TIMES:
        break
  print('reward:',np.mean(params_dict['episode_rewards']),np.std(params_dict['episode_rewards']))
  print('steps:',np.mean(params_dict['epi_length']),np.std(params_dict['epi_length']))
  d = np.array(params_dict['ends'])
  print(len(np.where(d==1)[0]),len(np.where(d==2)[0]),len(np.where(d==3)[0]))
  with open('res_rule.pkl','wb') as writer:
    pickle.dump(params_dict,writer)  
def curve():
  import matplotlib.pyplot as plt
  with open('res_rule.pkl','rb') as reader:
    params_dict=pickle.load(reader)
  l = []
  for s in params_dict['speed']:
    if len(s)==200:
      l.append(s)

  plt.figure()
  plt.plot(np.mean(np.array(l),axis=0))
  plt.fill_between(np.arange(200),np.mean(np.array(l),axis=0)+np.std(np.array(l),axis=0),np.mean(np.array(l),axis=0)-np.std(np.array(l),axis=0))
  plt.show()
  


if __name__=="__main__":
  #main()
  curve()