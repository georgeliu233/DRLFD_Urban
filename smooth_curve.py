import pandas as pd
import numpy as np
import os
import json
def smooth(csv_path,weight=0.85,fill=False):
    if isinstance (csv_path,str):
        data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
        scalar = data['Value'].values
    else:
        scalar=csv_path
    last = scalar[0]
    last_low = scalar[0]
    last_high = scalar[0]
    smoothed = []
    fill_low = []
    fill_high = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        if point <=smoothed_val:
            fill_low.append(point)
            last_low = point
        else:
            fill_low.append(last_low)
        if point >=smoothed_val:
            fill_high.append(point)
            last_high=point
        else:
            fill_high.append(last_high)
        last = smoothed_val

    if not isinstance (csv_path,str):
        return smoothed
    return data["Step"].values,smoothed,last_low,last_high,np.clip(data['Value'].values,0,1500)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
def plot_ablation():
    path = "C:/Users/24829/Desktop/duibi/"
    #csv_list =["Proposed-3","pretrain",'pretrain4','SAC'] 
    csv_list =["pretrain3",'pretrain','SAC'] 
    weight = 0.98
    x = []
    y = []
    x_origin = []
    low = []
    high = []
    for i in range(len(csv_list)):
        x1,y1,l,h,_ = smooth(path+csv_list[i]+".csv",weight=weight)
        _,y2,_,_,_ = smooth(path+csv_list[i]+".csv",weight=0.8)
        x1 = [0]+x1.tolist()
        y1 = [0]+y1
        x.append(x1)
        y.append(y1)
        
        low.append(l)
        high.append(h)

        x_origin.append(y2)
    color = ['b','g','r','c'] 

    plt.figure()
    for i in range(len(csv_list)):
        plt.plot(x[i],y[i],color=color[i],linewidth=2)
        
        #plt.fill_between(x[i],low[i],high[i],alpha=0.5)

    #for i in range(len(csv_list)):
    #    plt.plot(x[i],x_origin[i],alpha=0.2,linewidth=4.5,color=color[i])
    plt.xlim([0,100000])
    #plt.plot([0,100000],[987.37,987.37])
    #plt.legend(["Proposed","without PER","without Pretraining",'Without Q Filter'],frameon=False,loc=4,bbox_to_anchor=(1,0.05))
    plt.legend(["Proposed","without PER",'Without Q Filter'],frameon=False,loc=4,bbox_to_anchor=(1,0.05))
    #plt.plot([0,100000],[1060.43,1060.43],linewidth=2, linestyle=":",color="k")
    ##plt.fill_between([0,100000],[1060.43-454/2,1060.43-454/2],[1060.43+454/2,1060.43+454/2],alpha=0.2,color="k")
    plt.xlabel("Training steps")
    plt.ylabel("Episode reward")
    plt.title("Training performances-Ablation study")
    plt.savefig('C:/Users/24829/Desktop/paper-pic/ablation2.pdf')
    plt.show()

def plot_rl():
    path = "C:/Users/24829/Desktop/duibi/"
    csv_list =["pretrain3","ddqn",'ppo','td3','a3c','SAC','dqfd'] 
    weight = 0.98
    x = []
    y = []
    x_origin = []
    low = []
    high = []
    for i in range(len(csv_list)):
        x1,y1,l,h,_ = smooth(path+csv_list[i]+".csv",weight=weight)
        _,y2,_,_,_ = smooth(path+csv_list[i]+".csv",weight=0.85)

        # x1 = [0]+x1.tolist()
        # y1 = [0]+y1

        x.append(x1)
        y.append(y1)
        
        low.append(l)
        high.append(h)

        x_origin.append(y2)
    color = ['b','g','r','c','m','y','salmon'] 
    #with open('C:/Users/24829/Desktop/gym-carla-master/tensorboard_dqn_log/json_log.json','r') as reader:
    #    dy,dx = json.loads(reader.read())
    #    print(len(dx),len(dy))
    #x.append(dx)
    #x_origin.append(smooth(dy,0.85))
    #y.append(smooth(dy,0)) 
    plt.figure()
    for i in range(len(csv_list)):
        plt.plot(x[i],y[i],color=color[i],linewidth=2)
        
        #plt.fill_between(x[i],low[i],high[i],alpha=0.5)

    #for i in range(len(csv_list)):
    #    plt.plot(x[i],x_origin[i],alpha=0.2,linewidth=4.5,color=color[i])
    
    plt.xlim([0,100000])
    #plt.plot([0,100000],[987.37,987.37])
    plt.legend(["Proposed","DQN","PPO",'TD3','A3C','SAC','DQfD'],frameon=False,loc=4,bbox_to_anchor=(0.98,0.1),ncol=2)
    plt.plot([0,100000],[940.81,940.81],linewidth=2.5, linestyle=":",color="k")
    #plt.plot([0,100000],[1060.43,1060.43],linewidth=2, linestyle=":",color="k")
    ##plt.fill_between([0,100000],[1060.43-454/2,1060.43-454/2],[1060.43+454/2,1060.43+454/2],alpha=0.2,color="k")
    plt.xlabel("Training steps")
    plt.ylabel("Episode reward")
    plt.title("Training performances-RL")
    plt.savefig('C:/Users/24829/Desktop/paper-pic/rl3.pdf')
    plt.show()

def plot_il():
    path = "C:/Users/24829/Desktop/duibi/"
    csv_list =["pretrain3",'sqill','gailll'] 
    weight = 0.98
    x = []
    y = []
    x_origin = []
    low = []
    high = []
    for i in range(len(csv_list)):
        x1,y1,l,h,_ = smooth(path+csv_list[i]+".csv",weight=weight)
        _,y2,_,_,_ = smooth(path+csv_list[i]+".csv",weight=0.8)

        x1 = [0]+x1.tolist()
        y1 = [0]+y1

        x.append(x1)
        y.append(y1)
        
        low.append(l)
        high.append(h)

        x_origin.append(y2)
    color = ['b','r','g','m','y'] 
    plt.figure()
    for i in range(len(csv_list)):
        plt.plot(x[i],y[i],color=color[i],linewidth=2)
        
        #plt.fill_between(x[i],low[i],high[i],alpha=0.5)

    
    plt.xlim([0,100000])
    #plt.plot([0,100000],[987.37,987.37],color=color[3],linewidth=3)
    plt.plot([0,100000],[1060.43,1060.43],linewidth=2.5, linestyle="-.",color="k")
    plt.legend(["Proposed","SQIL",'GAIL','Expert'],frameon=False,loc=4,bbox_to_anchor=(1,0.18))
    #plt.legend(["Proposed","SQIL",'GAIL','BC','Expert'],frameon=False,loc=4,bbox_to_anchor=(1,0.18))
    #for i in range(len(csv_list)):
    #    plt.plot(x[i],x_origin[i],alpha=0.2,linewidth=4.5,color=color[i])
    plt.plot([0,100000],[940.81,940.81],linewidth=2.5, linestyle=":",color="k")
    plt.fill_between([0,100000],[1060.43-454/2,1060.43-454/2],[1060.43+454/2,1060.43+454/2],alpha=0.1,color="k")
    plt.xlabel("Training steps")
    plt.ylabel("Episode reward")
    plt.title("Training performances-IL")
    plt.savefig('C:/Users/24829/Desktop/paper-pic/il2.pdf')
    plt.show()

plot_il()