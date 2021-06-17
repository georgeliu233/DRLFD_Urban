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
if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

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
import pygame
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
    'number_of_vehicles': 100,
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
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
    'obs_single':True, #return a single key of the obs_space dict for obs space
    'obs_name':'birdeye',#basic options have 'camera' ; 'lidar' ; 'birdeye' ; 'state'
    'add_state':False,
    'use_control':True,
    'random_seed':200
  }

  # Set gym-carla environment
  env = gym.make('carla-v0',params=params)
  env = env.unwrapped
  
  #env = DummyVecEnv([lambda: env])
  '''
  def init(self):
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    if joystick_count > 1:
        raise ValueError("Please Connect Just One Joystick")

    self._joystick = pygame.joystick.Joystick(0)
    self._joystick.init()

    self._parser = ConfigParser()
    self._parser.read('C:/WindowsNoEditor/PythonAPI/examples/wheel_config.ini')
    self._steer_idx = int(
        self._parser.get('G29 Racing Wheel', 'steering_wheel'))
    self._throttle_idx = int(
        self._parser.get('G29 Racing Wheel', 'throttle'))
    self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
    self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
    self._handbrake_idx = int(
        self._parser.get('G29 Racing Wheel', 'handbrake'))
    '''
  def dummy_expert(_obs):
    try:
      from pygame.locals import KMOD_CTRL
      from pygame.locals import KMOD_SHIFT
      from pygame.locals import K_0
      from pygame.locals import K_9
      from pygame.locals import K_BACKQUOTE
      from pygame.locals import K_BACKSPACE
      from pygame.locals import K_COMMA
      from pygame.locals import K_DOWN
      from pygame.locals import K_ESCAPE
      from pygame.locals import K_F1
      from pygame.locals import K_LEFT
      from pygame.locals import K_PERIOD
      from pygame.locals import K_RIGHT
      from pygame.locals import K_SLASH
      from pygame.locals import K_SPACE
      from pygame.locals import K_TAB
      from pygame.locals import K_UP
      from pygame.locals import K_a
      from pygame.locals import K_c
      from pygame.locals import K_d
      from pygame.locals import K_h
      from pygame.locals import K_m
      from pygame.locals import K_p
      from pygame.locals import K_q
      from pygame.locals import K_r
      from pygame.locals import K_s
      from pygame.locals import K_w
      from pygame.locals import K_MINUS
      from pygame.locals import K_EQUALS
    except ImportError:
      raise RuntimeError('cannot import pygame, make sure pygame package is installed')
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    _joystick = pygame.joystick.Joystick(0)
    _joystick.init()
    if joystick_count > 1:
      raise ValueError("Please Connect Just One Joystick")
    numAxes = _joystick.get_numaxes()
    jsInputs = [float(_joystick.get_axis(i)) for i in range(numAxes)]
    # print (jsInputs)
    #jsButtons = [float(_joystick.get_button(i)) for i in
    #                range(_joystick.get_numbuttons())]

    # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
    # For the steering, it seems fine as it is
    K1 = 0.55  # 0.55
    steerCmd = K1 * math.tan(1.1 * jsInputs[0])
    steerCmd = np.clip(steerCmd,-1.0,1.0,dtype="float32")
    K2 = 1.6  # 1.6
    throttleCmd = K2 + (2.05 * math.log10(
        -0.7 * jsInputs[2] + 1.4) - 1.2) / 0.92
    if throttleCmd <= 0:
        throttleCmd = 0
    elif throttleCmd > 1:
        throttleCmd = 1

    brakeCmd = 1.6 + (2.05 * math.log10(
        -0.7 * jsInputs[3] + 1.4) - 1.2) / 0.92
    if brakeCmd <= 0:
        brakeCmd = 0
    elif brakeCmd > 1:
        brakeCmd = 1
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """
    
    #c = env.ego.get_control()
    #print("throttle:",c.throttle,"steer:",c.steer,"brake:",c.brake)
    #acc = c.throttle if c.throttle >0 else -c.brake
    #acc = (acc+1)/2
    #steer = np.clip(c.steer,-0.2,0.2)
    #action = np.array([acc,steer])
    #print(steerCmd,brakeCmd,throttleCmd)
    if brakeCmd >0:
        return np.array([-brakeCmd/2 + 0.5,steerCmd])
    else :
        return np.array([throttleCmd/2 +0.5,steerCmd])
  def dummy_expert_key(_obs):
      try:
        from pygame.locals import K_DOWN
        from pygame.locals import K_UP
      except ImportError:
        raise RuntimeError('cannot import pygame, make sure pygame package is installed')
      keys = pygame.key.get_pressed()
      if keys[K_UP]:
          acc = 1.0
      elif keys[K_DOWN]:
          acc = 0.0
      else:
          acc = 0.5
      return acc


  #model = SQIL_DQN.load("./dqn_logs_xunhuan_SQIL/rl_model_80000_steps.zip") 
  generate_expert_traj(dummy_expert_key, 'expert_carla_new_human_continuous_key_mlp',env,n_timesteps=0,n_episodes=5,limit_return=200,mlp_obs=True)
  #model = SAC.load("./dqn_logs_xunhuan_sac2/rl_model_30000_steps.zip",env)
  #model = DQN.load("rl_model_90000_steps.zip")
  #generate_expert_traj(model, 'expert_carla_sac_test',env,n_timesteps=0,n_episodes=100,limit_return=200,mlp_obs=True)
  print("expert saved!")
  '''
  n_actions = env.action_space.shape[-1]
  model = TD3(CnnPolicy,env,gamma=0.995,learning_starts=10000,batch_size=64,gradient_steps=1,train_freq=1,action_noise= NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)))
  model.tensorboard_log = "./tensorboard_dqn_log/TD3/"
  checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_logs_xunhuan_TD3/')
  model.learn(total_timesteps=100000,callback=checkpoint_callback,reset_num_timesteps=False)
  '''
  #dataset = ExpertDataset(expert_path='C:/Users/24829/Desktop/gym-carla-master/expert_carla_new_human_test_key_3.npz',traj_limitation=-1)
  #dataset.plot()
  #print("mean_exp_return is:",np.mean(dataset.returns))
  #model = GAIL.load("./dqn_logs_xunhuan_GAIL/rl_model_146944_steps.zip",env)
  #model = TD3.load("td3_carla")
  #model = PPO2.load("./dqn_logs_xunhuan_ppo/rl_model_90000_steps.zip")

  #model = TD3.load("./dqn_logs_xunhuan_td3/rl_model_90000_steps.zip")
  #generate_expert_traj(model, 'expert_carla_TD3_test_2',env,n_timesteps=0,n_episodes=100,limit_return=200,mlp_obs=True,recurrent=True)
  
  #dataset = ExpertDataset(expert_path='C:/Users/24829/Desktop/gym-carla-master/expert_carla_TD3_test_2.npz',traj_limitation=100)
  
  
  

if __name__ == '__main__':
  main()