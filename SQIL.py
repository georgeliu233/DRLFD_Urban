from functools import partial

import tensorflow as tf
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.deepq.build_graph import build_train
from stable_baselines.deepq.policies import DQNPolicy

from stable_baselines import DQN

from skimage.measure import block_reduce

class SQIL_DQN(DQN):
    
    def _initializeExpertBuffer(self, np_arr, obs_len):
        """
        not for public use!!! use initializeExpertBuffer instead!!!
        """
        done = np.array([[False] for i in range(0, len(np_arr)-1)])
        done[-1] = True

        self.expert_buffer.extend(np_arr[:-1, :obs_len], np_arr[:-1, obs_len:], np.ones(len(np_arr)-1), np_arr[1:, :obs_len], done)
    
    def initializeExpertBuffer(self, np_arr_list, obs_len,action_list,reward_list,done_list):
        """
        expects to be given a list of np_arrays (trajectories), sets all rewards to 1
        """
        print(self.prioritized_replay)
        if self.prioritized_replay:
                self.expert_buffer = PrioritizedReplayBuffer(obs_len, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = 100000
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.exp_beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
        else:
            self.expert_buffer = ReplayBuffer(obs_len)
            self.exp_beta_schedule = None

        for i in range(obs_len-2):
            obs,obs_ = np_arr_list[i],np_arr_list[i+1]
            obs = np.reshape(obs,(64,64,3))
            obs_ = np.reshape(obs_,(64,64,3))
            self.expert_buffer.add(obs,action_list[i],reward_list[i],obs_,done_list[i])

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):
        
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
                
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=self.exploration_initial_eps,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            episode_successes = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            reset = True
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()

            for _ in range(total_timesteps):
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)

                self.num_timesteps += 1

                # Stop training if return value is False
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, rew
                # Store transition in the replay buffer, but change reward to 0 (use it for plot later though)
                self.replay_buffer.add(obs_, action, reward_, new_obs_, float(done))
                #self.replay_buffer.add(obs_, action, 0, new_obs_, float(done))
                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                if writer is not None:
                    ep_rew = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                        self.num_timesteps)

                episode_rewards[-1] += reward_
                if done:
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                can_sample = self.replay_buffer.can_sample(self.batch_size)
                if can_sample and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:

                    callback.on_rollout_end()
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    # pytype:disable=bad-unpacking
                    if self.prioritized_replay:
                        assert self.beta_schedule is not None, \
                               "BUG: should be LinearSchedule when self.prioritized_replay True"
                        experience = self.replay_buffer.sample(int(self.batch_size*0.75),
                                                               beta=self.beta_schedule.value(self.num_timesteps),
                                                               env=self._vec_normalize_env)
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        expert_experience = self.expert_buffer.sample(int(self.batch_size*0.25),
                                                               beta=self.exp_beta_schedule.value(self.num_timesteps),
                                                               env=self._vec_normalize_env)
                        (obses_t_exp, actions_exp, rewards_exp, obses_tp1_exp, dones_exp, weights_exp, batch_idxes_exp) = expert_experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size,
                                                                                                env=self._vec_normalize_env)
                        #also sample from expert buffer
                        obses_t_exp, actions_exp, rewards_exp, obses_tp1_exp, dones_exp = self.expert_buffer.sample(self.batch_size,
                                                                                                env=self._vec_normalize_env)
                        weights, batch_idxes = np.ones_like(rewards), None
                        weights_exp, batch_idxes_exp = np.ones_like(rewards_exp), None
                    # pytype:enable=bad-unpacking

                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        if (1 + self.num_timesteps) % 100 == 0:
                            run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, td_errors = self._train_step(np.append(obses_t, obses_t_exp, axis = 0), np.append(actions, actions_exp.flatten(), axis = 0),
                                                                    np.append(rewards, rewards_exp.flatten(), axis = 0), np.append(obses_tp1, obses_tp1_exp, axis = 0),
                                                                    np.append(obses_tp1, obses_tp1_exp, axis = 0),
                                                                    np.append(dones.flatten(), dones_exp.flatten(), axis = 0), np.append(weights, weights_exp), sess=self.sess, options=run_option,
                                                                    run_metadata=run_metadata)
                            writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        else:
                            summary, td_errors = self._train_step(np.append(obses_t, obses_t_exp, axis = 0), np.append(actions, actions_exp.flatten(), axis = 0),
                                                                    np.append(rewards, rewards_exp.flatten(), axis = 0), np.append(obses_tp1, obses_tp1_exp, axis = 0),
                                                                    np.append(obses_tp1, obses_tp1_exp, axis = 0),
                                                                    np.append(dones.flatten(), dones_exp.flatten(), axis = 0), np.append(weights, weights_exp), sess=self.sess)
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, td_errors = self._train_step(np.append(obses_t, obses_t_exp, axis = 0), np.append(actions, actions_exp.flatten(), axis = 0),
                                                                    np.append(rewards, rewards_exp.flatten(), axis = 0), np.append(obses_tp1, obses_tp1_exp, axis = 0),
                                                                    np.append(obses_tp1, obses_tp1_exp, axis = 0),
                                                                    np.append(dones.flatten(), dones_exp.flatten(), axis = 0), np.append(weights, weights_exp), sess=self.sess)

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                        #print(new_priorities.shape)
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities[:int(self.batch_size*0.75)])
                        self.expert_buffer.update_priorities(batch_idxes_exp, new_priorities[int(self.batch_size*0.75):])


                    callback.on_rollout_start()

                if can_sample and self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring",
                                          int(100 * self.exploration.value(self.num_timesteps)))
                    logger.dump_tabular()

        episode_rewards = (list(avg_n(episode_rewards, 10)))
        ts = pd.Series(episode_rewards, index = range(0, len(episode_rewards)))
        ax = ts.plot()
        ax.set_ylabel("average reward")
        ax.set_xlabel("batch of 10 episodes")
        plt.show()
        
        callback.on_training_end()
        return self

def avg_n(data, LEN):
    """
    taken from https://stackoverflow.com/questions/39814034/how-do-i-get-the-average-of-every-10-numbers-in-list-in-python
    """
    datasum = cnt = 0 
    for num in data:
        datasum += num
        cnt += 1
        if cnt == LEN: 
            yield datasum / LEN
            datasum = cnt = 0 
    if cnt: 
        yield datasum / cnt