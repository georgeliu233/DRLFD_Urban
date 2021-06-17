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

from SQIL import SQIL_DQN

from tensorflow.keras.preprocessing.sequence import pad_sequences
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
  #env = gym.make('carla-v0',params=params)
  #env = env.unwrapped
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
    
  #model = SAC.load("./dqn_logs_xunhuan_sac2/rl_model_30000_steps.zip",env)
  #model = DQN.load("dqn_carla_new")
  #generate_expert_traj(model, 'expert_carla_dqn_1',env,n_timesteps=0,n_episodes=5,limit_return=200)
  print("expert saved!")
  
  dataset = ExpertDataset(expert_path='expert_carla_dqn_1.npz')
  #dataset.plot()
  model = SQIL_DQN(CnnPolicy, env,gamma=0.995,buffer_size=dataset.observations.shape[0]+1)
  model.tensorboard_log = './tensorboard_dqn_log/SQIL/'
  # Note: in practice, you need to train for 1M steps to have a working policy
  checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_logs_xunhuan_DQfD/')
  #model.pretrain(dataset,n_epochs=10000)
  #model.save("BC_gail_CARLA")
  model.initializeExpertBuffer(dataset.observations,dataset.observations.shape[0]+1,dataset.actions,0.0,dataset.dones)
  print("expert_buffer initiated!")
  model.learn(total_timesteps=100000,callback=checkpoint_callback)
  model.save("gail_CARLA")
  
  model = SQIL_DQN.load("./dqn_logs_xunhuan_DQfD/rl_model_80000_steps.zip")
  dataset = ExpertDataset(expert_path='expert_carla_dqn_1.npz')
  model.initializeExpertBuffer(dataset.observations,dataset.observations.shape[0]+1,dataset.actions,dataset.dones)
  model.buffer_size = dataset.observations.shape[0]+1
  generate_expert_traj(model, 'expert_carla_DQFD_test',env,n_timesteps=0,n_episodes=100,limit_return=200,mlp_obs=True)
  '''
  path = 'C:/Users/24829/Desktop/gym-carla-master/'
  dts1 = ExpertDataset(expert_path=path+'expert_carla_sacfd_test.npz')
  speed_data1 = dts1.observations[:,2]
  dones_data1 = dts1.dones
  dts2 = ExpertDataset(expert_path=path+'expert_carla_new_human_continuous_key_mlp.npz')
  speed_data2 = dts2.observations[:,2]
  dones_data2 = dts2.dones
  #print(speed_data1[:20])
  #dts2 = ExpertDataset(expert_path='expert_carla_new_human_continuous_mlp_test.npz')
  #speed_data2 = dts2.observations[:,0]
  #dones_data2 = dts2.dones
  dts3 = ExpertDataset(expert_path=path+'expert_carla_ppo_test.npz')
  print(dts3.observations.shape)
  speed_data3 = dts3.observations[:,2]
  dones_data3 = dts3.dones
  dts4 = ExpertDataset(expert_path=path+'expert_carla_TD3_test.npz')
  dones_data4 = dts4.dones
  speed_data4 = dts4.observations[:,2]
  dones_data4 = dts4.dones
  dts5 = ExpertDataset(expert_path=path+'expert_carla_new_human_test_discrete_te2.npz')
  speed_data5 = dts5.observations[:350,]
  dones_data5 = dts5.dones
  dts6 = ExpertDataset(expert_path=path+'expert_carla_new_human_test_discrete_te2.npz')
  speed_data6 = dts6.observations[:350,2]
  dones_data6 = dts6.dones
  #print(speed_data[:10])
  def preprocess(speed_data,dones_data,num):
    speed_data = np.reshape(speed_data,(-1,1))
    j = 0
    ind = []
    for i in range(dones_data.shape[0]):
      if dones_data[i]==2 or dones_data==True:
        ind.append(i)
    for i in range(num):
      if i==num-1:
        buf_arr = speed_data[ind[i]:]
      else:
        buf_arr = speed_data[ind[i]:ind[i+1]]
      buf_arr = pad_sequences([buf_arr],maxlen=800,padding='post',truncating='post',dtype="float32")
      buf_arr = buf_arr[0]
      if i==0:
        #print(buf_arr.shape)
        x_array = buf_arr
      else:
        x_array = np.append(x_array,buf_arr,axis=1) 
    return x_array
  def preprocess_2(data,dones,num,n=False):
    data = np.reshape(data,(-1,1))
    j = 0
    ind = []
    for i in range(dones.shape[0]):
      if dones[i]==2 or dones[i]==True:
        ind.append(i)
    
    out_list = []
    if n:
      nl = [9,11,19,25,27]
    for i in range(num):
      if n:
        if i in nl:
          continue
      if i==num-1:
        buf_arr = data[ind[i]:]
      else:
        buf_arr = data[ind[i]:ind[i+1]]
      #print(len(buf_arr))
      if len(buf_arr)<200:
        continue
      out_list.append(buf_arr[:200])  
    return out_list
  
  #out_x = preprocess_2(speed_data1,dones_data1,30)
  #out_y = preprocess_2(speed_data7,dones_data7,30,realm=55.47)
  out_speed = preprocess_2(speed_data1,dones_data1,30,n=True)
  avg = np.mean(np.reshape(out_speed,(-1,200)),axis=0)
  std = np.std(np.reshape(out_speed,(-1,200)),axis=0)
  out_speed_2 = preprocess_2(speed_data2,dones_data2,5)
  avg2 = np.mean(np.reshape(out_speed_2,(-1,200)),axis=0)
  std2 = np.std(np.reshape(out_speed_2,(-1,200)),axis=0)

  out_speed_3 = preprocess_2(speed_data3,dones_data3,30)
  avg3 = np.mean(np.reshape(out_speed_3,(-1,200)),axis=0)
  std3 = np.std(np.reshape(out_speed_3,(-1,200)),axis=0)
  out_speed_4 = preprocess_2(speed_data4,dones_data4,30)
  avg4 = np.mean(np.reshape(out_speed_4,(-1,200)),axis=0)
  std4 = np.std(np.reshape(out_speed_4,(-1,200)),axis=0)

  avg = avg[20:]
  avg2 = avg2[20:]
  std = std[20:]
  std2 = std2[20:]

  avg3 = avg3[20:]
  avg4 = avg4[20:]
  std3 = std3[20:]
  std4 = std4[20:]
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set(style="darkgrid")
  
  plt.figure()
  #plt.grid(True)
  plt.plot(np.linspace(0,800,len(avg)),avg,linewidth=2.5,alpha=0.7)
  plt.fill_between(np.linspace(0,800,len(avg)),avg+std/2,avg-std/2,alpha=0.2)
  plt.plot(np.linspace(0,800,len(avg2)),avg2,linewidth=2.5,alpha=0.7)
  plt.fill_between(np.linspace(0,800,len(avg2)),(avg2+std2/2),(avg2-std2/2),alpha=0.2)
  #plt.plot(np.linspace(0,800,len(avg3)),avg3,linewidth=2.5,alpha=0.7)
  #plt.fill_between(np.linspace(0,800,len(avg3)),(avg3+std3/2),(avg3-std3/2),alpha=0.2)
  #plt.plot(np.linspace(0,800,len(avg4)),avg4,linewidth=2.5,alpha=0.7)
  #plt.fill_between(np.linspace(0,800,len(avg4)),(avg4+std4/2),(avg3-std4/2),alpha=0.2)
  plt.plot([0,800],[7,7],linewidth=3)
  plt.legend(['Proposed','Human','Set speed'],frameon=False,loc=4,bbox_to_anchor=(0.95,0.15))
  plt.axis([0,800,0,10])
  plt.xlabel('Steps')
  plt.ylabel('Speed m/s')
  plt.savefig('C:/Users/24829/Desktop/paper-pic/speed.pdf')
  #for i in range(len(out_speed)):
  #  plt.plot(range(200),out_speed[i],linewidth=1,alpha=0.7)
  #plt.scatter(-52.47,6.48)
  #plt.axis("equal")
  #plt.savefig("speed.png",transparent=True)
  plt.show()
  
  '''
  x_array1 = preprocess(speed_data1,dones_data1,3)
  x_array2 = preprocess(speed_data2,dones_data2,10)
  x_array3 = preprocess(speed_data3,dones_data3,3)
  x_array4 = preprocess(speed_data4,dones_data4,10)
  x_array5 = preprocess(speed_data5,dones_data5,1)
  x_array6 = preprocess(speed_data6,dones_data6,1)
  x1 = np.abs(np.mean(x_array1-0.1*x_array3,axis=1))
  std1 = np.std(x_array1,axis=1)
  x2 = np.abs(np.mean(x_array2-0.1*x_array4,axis=1))
  std2 = np.std(x_array2,axis=1)
  x3 = np.abs(np.mean(x_array3,axis=1))
  std3 = np.std(x_array3,axis=1)
  x4 = np.abs(np.mean(x_array4,axis=1))
  std4 = np.std(x_array4,axis=1)
  x5 = np.abs(np.mean(x_array5-0.1*x_array6,axis=1))
  std5 = np.std(x_array5,axis=1)
  #print(x.shape)
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set(style="darkgrid")
  plt.figure()
  
  #plt.fill_between(range(len(x1)),np.squeeze(x1-std1),np.squeeze(x1+std1), alpha=0.2)
  
  plt.plot(x5)
  plt.plot(x1)
  #plt.fill_between(range(len(x2)),np.squeeze(x2-std2),np.squeeze(x2+std2), alpha=0.2)
  plt.plot(x2[:350])
  plt.legend(["PID","PID+PP",'Filtered'],frameon=False)
  plt.title("Average lateral errors")
  plt.xlabel("Steps/n")
  plt.ylabel("Errors/m")
  '''
  '''
  plt.plot(x3)
  plt.fill_between(range(len(x3)),np.squeeze(x3-std3),np.squeeze(x3+std3), alpha=0.2)
  plt.plot(x4)
  plt.fill_between(range(len(x4)),np.squeeze(x4-std4),np.squeeze(x4+std4), alpha=0.2)
  plt.plot(x5)
  plt.fill_between(range(len(x5)),np.squeeze(x5-std5),np.squeeze(x5+std5), alpha=0.2)\
  plt.legend(["DQN","PPO","SAC","TD3","A3C"])
  plt.plot(np.mean(x1[:400]))
  '''
  
  #plt.grid(True)
  #plt.plot(x_array1[:,3])
  #plt.plot(x_array2[:,2])
  plt.legend(["H","s"])
  plt.show()
  
    

    

  #dts.plot()

def calc_one_method(p,num,n=False,name=''):
  path = 'C:/Users/24829/Desktop/gym-carla-master/'
  dts = ExpertDataset(expert_path=path+p)
  #if dts.observations.shape[1]==7:
  #  print('='*5,dts.observations.shape,name,'='*5)
  speed_data = dts.observations[:,2]
  dones_data = dts.dones

  def preprocess_2(data,dones,num,n=False):
    data = np.reshape(data,(-1,1))
    j = 0
    ind = []
    for i in range(dones.shape[0]):
      if dones[i]==2 or dones[i]==True:
        ind.append(i)
    
    out_list = []
    if n:
      nl = [9,11,19,25,27]
    for i in range(num):
      if n:
        if i in nl:
          continue
      if i==num-1:
        buf_arr = data[ind[i]:]
      else:
        buf_arr = data[ind[i]:ind[i+1]]
      print(len(buf_arr))
      if len(buf_arr)<200:
        continue
      out_list.append(buf_arr[:200])  
    return out_list

  out_speed = preprocess_2(speed_data,dones_data,num,n=n)
  avg = np.mean(np.reshape(out_speed,(-1,200)),axis=0)
  std = np.std(np.reshape(out_speed,(-1,200)),axis=0)
  return avg[20:],std[20:]

def plot_figure():
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set(style="darkgrid")
  path_list=[
    'expert_carla_sacfd_test.npz',
    'expert_carla_new_human_continuous_key_mlp.npz',
    'expert_carla_ppo_test.npz',
    'expert_carla_TD3_test.npz',
    'expert_carla_sac_test.npz',
    'expert_carla_sqil_test.npz',
    #'expert_carla_gail_mlp.npz',
    #'expert_carla_bc_mlp.npz',
    #'expert_carla_dqnrule_test_2.npz'
  ]
  plt.figure()
  #plt.grid(True)
  for i,p in enumerate(path_list):

    if i==0:
      n=True
    else:
      n=False

    if i==1:
      num=5
    else:
      num=30
    avg , std = calc_one_method(p,num,n)
    if i>1:
      plt.plot(np.linspace(0,20,len(avg)),avg*7/5,linewidth=2,alpha=0.7)
      plt.fill_between(np.linspace(0,20,len(avg)),avg*7/5+std/2,avg*7/5-std/2,alpha=0.2)
    else:
      plt.plot(np.linspace(0,20,len(avg)),avg,linewidth=2,alpha=0.7)
      plt.fill_between(np.linspace(0,20,len(avg)),avg+std/2,avg-std/2,alpha=0.2)
  
  with open('res_rule.pkl','rb') as reader:
    params_dict=pickle.load(reader)
  l = []
  for s in params_dict['speed']:
    if len(s)==200:
      l.append(s[22:])

  plt.plot(np.linspace(0,20,200-22),np.mean(np.array(l),axis=0),linewidth=2,alpha=0.7)
  plt.fill_between(np.linspace(0,20,200-22),np.mean(np.array(l),axis=0)+np.std(np.array(l),axis=0),np.mean(np.array(l),axis=0)-np.std(np.array(l),axis=0),alpha=0.2)

  #plt.plot([0,20],[7,7],linewidth=3)
  plt.xlabel('Time /s')
  plt.ylabel('Speed m/s')
  # plt.legend(['Proposed','Human','PPO','TD3','SAC','GAIL','SQIL',"BC",'DQfD',"Rule-based",'Set speed'],frameon=False,\
  #   bbox_to_anchor=(0.68,0.99),loc=9,ncol=3,fontsize='x-small')
  plt.legend(['Proposed','Human','PPO','TD3','SAC','SQIL','DQfD'],frameon=False,\
    bbox_to_anchor=(0.68,0.99),loc=9,ncol=3,fontsize='x-small')
  plt.axis([0,20,0,10])
  plt.title('Agent vehicle Speed')
  plt.savefig('C:/Users/24829/Desktop/paper-pic/speed_5.pdf')
  plt.show()
  
def calc_traj(p,num,n=False):
  path = 'C:/Users/24829/Desktop/gym-carla-master/'
  dts = ExpertDataset(expert_path=path+p)
  #if dts.observations.shape[1]==7:
  #  print('='*5,dts.observations.shape,name,'='*5)
  x_data = dts.observations[:,5]
  y_data = dts.observations[:,6]
  dones_data = dts.dones
  def preprocess(speed_data,dones_data,num,n):
    speed_data = np.reshape(speed_data,(-1,1))
    j = 0
    ind = []
    x_array = []
    if n:
      nl = [9,11,19,25,27]
    for i in range(dones_data.shape[0]):
      if dones_data[i]!=0:
        ind.append(i)
    for i in range(num):
      if n:
        if i in [1,2,6,9,17,18,19,20,21,25,27,29]:
          continue
      if i==num-1:
        buf_arr = speed_data[ind[i]:]
      else:
        buf_arr = speed_data[ind[i]:ind[i+1]]
      #buf_arr = pad_sequences([buf_arr],maxlen=800,padding='post',truncating='post',dtype="float32")
      x_array.append(buf_arr) 
    return x_array
  
  x,y = preprocess(x_data,dones_data,num,n),preprocess(y_data,dones_data,num,n)
  return x,y


def draw_traj():
  path_list=[
    'expert_carla_sacfd_test.npz',
    'expert_carla_new_human_continuous_key_mlp.npz'
  ]
  l = ['Proposed','Human']
  i = 0
  all_traj_list = []
  for p in path_list:
    X,Y = [],[]
    if i==1:
      num=5
      n=True
    else:
      num=30
      n=True
    x,y = calc_traj(p,num,n)
    for x_traj,y_traj in zip(x,y):
      X.append(x_traj)
      Y.append(y_traj)
    all_traj_list.append((X,Y))
    i+=1
  import matplotlib.pyplot as plt
  import seaborn as sns
  #sns.set(style="darkgrid")
  plt.figure()
  #plt.grid(True)
  j = 0

  mk,cl = ['o','^'],['r','b']
  line_width=[3.5,1.5]
  for X,Y in all_traj_list:
    k = 0
    for x,y in zip(X,Y):

      #plt.plot(-y,-x,linewidth=2,alpha=0.5)
      plt.scatter(-y[-1],-x[-1],c='orange',marker='o',linewidths=8,alpha=0.5)
      print(len(x),j,k,x[-1],y[-1])
      if x[-1]>-5:
        plt.plot(-y,-x,linewidth=5.5,alpha=1,color='red')
      k+=1
    plt.plot(-Y[0],-X[0],linewidth=3.5,alpha=0.5,color='b')
    j+=1
  #plt.axis([-52,-50,6.45,6.55])
  plt.savefig("trajj.png",transparent=True)
  #plt.savefig('C:/Users/24829/Desktop/paper-pic/traj_point2.pdf')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()

    


def recg_files():
  
  path = 'C:/Users/24829/Desktop/gym-carla-master/'
  path_list = glob.glob(path+'*.npz')
  print(path_list)
  dts = ExpertDataset(expert_path=path+p)
  
def test_list():
  a = np.array([1,0,0,2,3,4,0,0,0,0])
  print(np.where(a!=0)[0])

  

if __name__ == '__main__':
  #draw_traj()
  #test_list()
  plot_figure()
  #main()