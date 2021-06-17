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

from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import SAC,GAIL,DQN
from stable_baselines.gail import generate_expert_traj,ExpertDataset
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.deepq.DQFD import DQFD
#from SQIL import SQIL_DQN
#from stable_baselines.common import SubprocVecEnv
def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 150,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 5,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
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
    'desired_speed': 5,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
    'obs_single':True, #return a single key of the obs_space dict for obs space
    'obs_name':'birdeye',#basic options have 'camera' ; 'lidar' ; 'birdeye' ; 'state'
    'use_control':True,
    'add_state':False,
    'random_seed':10
  }

  # Set gym-carla environment
  env = gym.make('carla-v0',params=params)
  env = env.unwrapped
  '''
  def dummy_expert(_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """
    c = env.ego.get_control()
    #print("throttle:",c.throttle,"steer:",c.steer,"brake:",c.brake)
    acc = c.throttle if c.throttle >0 else -c.brake
    acc = (acc+1)/2
    steer = np.clip(c.steer,-0.2,0.2)
    action = np.array([acc,steer])
    #print(action)
    return action
    '''
  #model = SAC.load("./dqn_logs_xunhuan_sac2/rl_model_30000_steps.zip",env)
  #model = DQN.load("dqn_carla_new")
  #generate_expert_traj(model, 'expert_carla_dqn_1',env,n_timesteps=0,n_episodes=5,limit_return=200)
  print("expert saved!")
  
  dataset = ExpertDataset(expert_path='expert_carla_dqn_1.npz')
  #dataset.plot()
  model = DQFD(CnnPolicy, env,gamma=0.995,buffer_size=dataset.observations.shape[0]+1,prioritized_replay=True,batch_size=3,learning_starts=500)
  model.tensorboard_log = './tensorboard_dqn_log/DQFD_2/'
  # Note: in practice, you need to train for 1M steps to have a working policy
  checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_logs_xunhuan_DQfD2/')
  #model.pretrain(dataset,n_epochs=10000)
  #model.save("BC_gail_CARLA")
  model.initializeExpertBuffer(dataset.observations,dataset.observations.shape[0]+1,dataset.actions,dataset.rewards,dataset.dones)
  print("expert_buffer initiated!")
  model.learn(total_timesteps=100000,callback=checkpoint_callback,pretrain_steps=0,pretrain=False)
  model.save("DQFD_CARLA2")
  '''
  model = SQIL_DQN.load("./dqn_logs_xunhuan_SQIL/rl_model_80000_steps.zip")
  dataset = ExpertDataset(expert_path='expert_carla_dqn_1.npz')
  model.initializeExpertBuffer(dataset.observations,dataset.observations.shape[0]+1,dataset.actions,dataset.dones)
  model.buffer_size = dataset.observations.shape[0]+1
  generate_expert_traj(model, 'expert_carla_sqil_test',env,n_timesteps=0,n_episodes=100,limit_return=200,mlp_obs=True)
  dts = ExpertDataset(expert_path='expert_carla_sqil_test.npz')
  dts.plot()
  '''

if __name__ == '__main__':
  main()