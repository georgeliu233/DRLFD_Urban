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

import gym
import numpy as np

from stable_baselines.sac.policies import LnCnnPolicy
from stable_baselines import SAC
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.gail import generate_expert_traj,ExpertDataset

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
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
    'obs_single':True, #return a single key of the obs_space dict for obs space
    'obs_name':'birdeye',#basic options have 'camera' ; 'lidar' ; 'birdeye' ; 'state'
    'add_state':False,
    'use_control':True,
    'random_seed':200
  }
  env = gym.make('carla-v0',params=params).unwrapped
  
  #n_actions = env.action_space.shape[-1]
  #param_noise = None
  #action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
  # Set gym-carla environment
  path = 'C:/Users/24829/Desktop/gym-carla-master/'
  dataset = ExpertDataset(expert_path=path+'expert_carla_new_human_continuous_test_key.npz')
  
  model = SAC(LnCnnPolicy,env,gamma=0.995,n_step=True,verbose=1,ratio=0.5,update_buffer_interval=5)
  model.pretrain(dataset,n_epochs=50)
  print("pretraining ok!")
  model.initializeExpertBuffer(dataset.observations,dataset.observations.shape[0]+1,dataset.actions,dataset.rewards,dataset.dones)
  #model = SAC.load("sac_carla_3",env)
  #model = DQN.load("C:/Users/24829/Desktop/rl_model_50000_steps.zip",env=env)
  callback = CheckpointCallback(save_freq=10000, save_path=path+'/sac_ratio/')
  #callback = CallbackList([checkpoint_callback, eval_callback])
  
  model.tensorboard_log=path+"/SACfD_log_ratio/"
  model.learn(total_timesteps=100000,pretrain_steps=0,mean_expert_reward=np.mean(dataset.rewards),callback=callback) 
  model.save(path+"sacfd_carla_r3")
  
  #model = SAC.load("sacfd_carla",env=env)
  #model.learn(total_timesteps=10000,pretrain_steps=0) 
  #print("SAC MODEL SAVED!!!---Hope to be work on the Car!")
  '''
  car_num = [0,50,100] #generate vehicles for each curriculum
  for i in range(3):
    vehicle = car_num[i]
    print("=======Iteration {},car number:{}========".format(i,vehicle))
    model.env.number_of_vehicles=vehicle 
    model.tensorboard_log="./tensorboard_dqn_log/DQN5/"
    model.learn(total_timesteps=50000,callback=checkpoint_callback,reset_num_timesteps=False) 

  
  #model.save("dqn_carla_6")
  print('Model Saved!!!')

 # del model # remove to demonstrate saving and loading
  
  model = SAC.load("sac_carla_2")
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