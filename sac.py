import sys
import time
import warnings

import numpy as np
import tensorflow as tf
from collections import deque
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action
from stable_baselines.common.schedules import get_schedule_fn
#from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.DQFD_buffers import ReplayBuffer, PrioritizedReplayBuffer,NStepTransitionBuffer
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger


class SAC(OffPolicyRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf DDPG for the different action noise type.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for SAC normally but can help exploring when using HER + SAC.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on SAC logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=2,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None,prioritized_replay=True,prioritized_replay_alpha=0.3, 
                 prioritized_replay_beta0=1.0, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6,ratio=0.75,n_step=False,update_buffer_interval=100,max_ratio=0.9):

        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)
        self.prioritized_replay  = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.update_buffer_interval = update_buffer_interval
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau

        self.ratio = ratio
        self.init_ratio = ratio
        self.max_ratio = max_ratio

        self.n_step = n_step
        self.n_step_length = 10
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = unscale_action(self.action_space, self.deterministic_action)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def initializeExpertBuffer(self, np_arr_list, obs_len,action_list,reward_list,done_list):
        """
        expects to be given a list of np_arrays (trajectories), sets all rewards to 1
        """
        #print(self.prioritized_replay)
        if self.prioritized_replay:
                self.expert_buffer = PrioritizedReplayBuffer(obs_len, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = 100000
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
        else:
            self.expert_buffer = ReplayBuffer(obs_len)
            self.exp_beta_schedule = None
        if self.n_step:
            n_step_buffer=deque(maxlen=self.n_step_length)
            self.expert_N_buffer = NStepTransitionBuffer(obs_len,n_step=self.n_step_length,gamma=self.gamma)
            
        for i in range(obs_len-2):
            obs,obs_ = np_arr_list[i],np_arr_list[i+1]
            obs = np.reshape(obs,(64,64,3))
            obs_ = np.reshape(obs_,(64,64,3))
            if done_list[i]==2 or done_list[i]==True:
                done = True
            else:
                done = False
            if not self.n_step:
                self.expert_buffer.add(obs,action_list[i],reward_list[i],obs_,done,1)
            else:
                trans = (obs,action_list[i],reward_list[i],obs_,done)
                n_step_buffer.append(trans)
                self.expert_N_buffer.add((obs,action_list[i],reward_list[i],obs_,done))
                if len(n_step_buffer)== self.n_step_length:    
                    #self.expert_buffer.add(obs,action_list[i],reward_list[i],obs_,done_list[i],1)
                    one_step = n_step_buffer[0]
                    self.expert_buffer.add(one_step[0],one_step[1],one_step[2],one_step[3],one_step[4],1)

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)
                if self.prioritized_replay:
                    self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                    if self.prioritized_replay_beta_iters is None:
                        prioritized_replay_beta_iters = 100000
                    else:
                        prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                    self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                        initial_p=self.prioritized_replay_beta0,
                                                        final_p=1.0)
                else:
                    self.replay_buffer = ReplayBuffer(self.buffer_size)
                    self.beta_schedule = None
                
                if self.n_step:
                    self.replay_N_buffer=NStepTransitionBuffer(self.buffer_size,self.n_step_length,self.gamma)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.is_demo_ph = tf.placeholder(tf.float32, shape=(None, 1), name='is_demonstrations')
                    self.weight_ph = tf.placeholder(tf.float32, shape=(None, 1), name='importance_weight')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    if self.n_step:
                        self.next_observations_ph_n = self.target_policy.obs_ph
                        self.processed_next_obs_ph_n = self.target_policy.processed_obs
                        self.rewards_ph_n = tf.placeholder(tf.float32, shape=(None, 1), name='n_step_rewards')
                        self.terminals_ph_n = tf.placeholder(tf.float32, shape=(None, 1), name='n_step_terminals')
                        
                    

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probability of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    self.obs_ph, self.actions_ph, self.deterministic_actions_ph = self._get_pretrain_placeholders()
                    
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False,
                                                                    reuse=True)
                    dtm_qf1,dtm_qf2,_ = self.policy_tf.make_critics(self.processed_obs_ph, self.deterministic_actions_ph,
                                                                     create_qf=True,create_vf=False,
                                                                    reuse=True)
                    

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target
                    if self.n_step:
                        _,_,value_target_n = self.policy_tf.make_critics(self.processed_next_obs_ph_n,
                                                                     create_qf=False, create_vf=True,reuse=True)
                        self.value_target_n = value_target_n

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)
                    
                    # Target for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    qf1_loss = 0.5 * tf.reduce_mean(((q_backup - qf1) ** 2)*self.weight_ph)
                    qf1_loss_col = tf.reduce_mean(((q_backup - qf1) ** 2),1)
                    qf2_loss = 0.5 * tf.reduce_mean(((q_backup - qf2) ** 2)*self.weight_ph)
                    if self.n_step:
                        q_backup_n = tf.stop_gradient(
                        self.rewards_ph_n +
                        (1 - self.terminals_ph_n) *( self.gamma**self.n_step_length ) * self.value_target_n)
                        qf1_loss_n = 0.5 * tf.reduce_mean(((q_backup_n - qf1) ** 2)*self.weight_ph)
                        qf1_loss_n_col = tf.reduce_mean(((q_backup_n - qf1) ** 2),1)
                        qf2_loss_n = 0.5 * tf.reduce_mean(((q_backup_n - qf2) ** 2)*self.weight_ph)
                    if self.n_step:
                        value_for_priority = qf1_loss_col + qf1_loss_n_col
                    else:
                        value_for_priority = qf1_loss_col
                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy)*self.weight_ph)
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean((self.ent_coef * logp_pi - min_qf_pi)*self.weight_ph)
                    actor_for_priority = tf.reduce_mean(self.ent_coef * logp_pi - min_qf_pi,1)
                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the Gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    min_q = tf.minimum(dtm_qf1,dtm_qf2)

                    Q_filter = tf.cast((qf1 > min_q)|(qf2 > min_q),tf.float32)
                    #Q_filter_1 = tf.cast(qf1 > min_q,tf.float32)
                    #Q_filter_2 = tf.cast(qf2 > min_q,tf.float32)
                    im_loss1 = tf.square(self.actions_ph - self.deterministic_actions_ph)*Q_filter*self.is_demo_ph
                    #im_loss2 = tf.square(self.actions_ph - self.deterministic_actions_ph)*Q_filter_2*self.is_demo_ph
                    #actor_loss_di1 = tf.reduce_mean(im_loss1)
                    #actor_loss_di2 = tf.reduce_mean(im_loss2)
                    self.actor_loss_di = tf.reduce_mean(im_loss1)
                    imitation_for_priority = tf.reduce_mean(im_loss1,axis=1)
                    regularizerpi = tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0, scale_l2=1e-5, scope="model/pi")
                    all_trainable_weights_pi = tf.trainable_variables('model/pi')
                    regularization_penalty_pi = tf.contrib.layers.apply_regularization(regularizerpi, all_trainable_weights_pi)

                    policy_loss = policy_kl_loss + regularization_penalty_pi + self.actor_loss_di


                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_loss = 0.5 * tf.reduce_mean(((value_fn - v_backup) ** 2)*self.weight_ph)
                    
                    #value_for_priority = tf.reduce_mean((value_fn - v_backup) ** 2,1)
                    regularizervf = tf.contrib.layers.l1_l2_regularizer(scale_l1=0.0, scale_l2=1e-5, scope='model/values_fn')
                    all_trainable_weights_vf = tf_util.get_trainable_vars('model/values_fn')
                    regularization_penalty_vf = tf.contrib.layers.apply_regularization(regularizervf, all_trainable_weights_vf)
                    if self.n_step:
                        values_losses = qf1_loss + qf2_loss + value_loss + regularization_penalty_vf + qf1_loss_n + qf2_loss_n
                    else:
                        values_losses = qf1_loss + qf2_loss + value_loss + regularization_penalty_vf


                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=tf_util.get_trainable_vars('model/pi'))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = tf_util.get_trainable_vars('model/values_fn')

                    source_params = tf_util.get_trainable_vars("model/values_fn/vf")
                    target_params = tf_util.get_trainable_vars("target/values_fn/vf")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy,actor_for_priority,value_for_priority,imitation_for_priority,self.actor_loss_di, policy_train_op, train_values_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('value_loss', value_loss)
                    tf.summary.scalar("Imitation_loss",self.actor_loss_di)
                    tf.summary.scalar('entropy', self.entropy)
                    tf.summary.scalar('importance weight',tf.reduce_mean(self.weight_ph))
                    if ent_coef_loss is not None:
                        tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        tf.summary.scalar('ent_coef', self.ent_coef)

                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.target_params = tf_util.get_trainable_vars("target/values_fn/vf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def pretrain_sac(self,pretrain_steps):
        print("=====SAC Pretraining=====")
        for step in range(pretrain_steps):
            # Compute current learning_rate
            frac = 1.0 - step / pretrain_steps
            current_lr = self.learning_rate(frac)
            # Update policy and critics (q functions)
            policy_loss, qf1_loss, qf2_loss, value_loss,*entropy =self._train_step(step, writer=None,learning_rate=current_lr,pretrain=True)
            if step % 50==0:
                 print("** Pretraining step: |",step/pretrain_steps," Actor loss: |",policy_loss, "Critic loss|",value_loss," Actor expert loss|",entropy[-1] )
            # Update target network
            if step % self.target_update_interval == 0:
                # Update target network
                self.sess.run(self.target_update_op)
            self.step += 1
        print("Pretrin complete!!!")
    def _train_step(self, step, writer, learning_rate,pretrain=False):
        # Sample a batch from the replay buffer
        if not pretrain:
            a = self.ratio
            if not self.prioritized_replay:
                batch = self.replay_buffer.sample(int(self.batch_size*a))
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones,batch_demos,batch_idx = batch
                weight= np.ones_like(batch_rewards)
            else:
                batch = self.replay_buffer.sample(int(self.batch_size*a),beta=self.beta_schedule.value(self.num_timesteps))
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones,batch_demos,weight,batch_idx = batch
            batch_rewards = batch_rewards.reshape(-1, 1)
            one_batch_r = batch_rewards
            batch_dones = batch_dones.reshape(-1, 1)
            batch_demos = batch_demos.reshape(-1, 1)
            weight = weight
            weight = weight.reshape(-1,1)

            if not self.prioritized_replay:
                expert_batch = self.expert_buffer.sample(int(self.batch_size*(1-a)))
                exp_batch_obs, exp_batch_actions, exp_batch_rewards, exp_batch_next_obs, exp_batch_dones,exp_demos,exp_batch_idx = expert_batch
                exp_weight= np.ones_like(exp_batch_rewards)
            else:
                expert_batch = self.expert_buffer.sample(int(self.batch_size*(1-a)),beta=self.beta_schedule.value(self.num_timesteps))
                exp_batch_obs, exp_batch_actions, exp_batch_rewards, exp_batch_next_obs, exp_batch_dones,exp_demos,exp_weight,exp_batch_idx = expert_batch
            #print(exp_batch_idx.shape)
            exp_batch_rewards = exp_batch_rewards.reshape(-1, 1)

            #self.new_ratio = self.ratio
            ##summ_r = np.mean(batch_rewards)>np.mean(exp_batch_rewards)
            #if summ_r:
            #    self.new_ratio = min(self.new_ratio + 2/self.batch_size,0.9)
            #else:
            #    self.new_ratio = max(self.new_ratio - 1/self.batch_size,0.1)
            exp_batch_dones = exp_batch_dones.reshape(-1, 1)
            exp_demos = exp_demos.reshape(-1,1)
            exp_weight = exp_weight
            exp_weight =exp_weight.reshape(-1,1)

            batch_obs = np.vstack((batch_obs,exp_batch_obs))
            batch_actions = np.vstack((batch_actions,exp_batch_actions))
            batch_rewards = np.vstack((batch_rewards,exp_batch_rewards))
            batch_next_obs = np.vstack((batch_next_obs,exp_batch_next_obs))
            batch_dones = np.vstack((batch_dones,exp_batch_dones))
            batch_demos = np.vstack((batch_demos,exp_demos))
            weight = np.vstack((weight,exp_weight))
            if self.n_step:
                nbatch = self.replay_N_buffer.sample(batch_idx)
                ex_nbatch = self.expert_N_buffer.sample(exp_batch_idx)

                _,_, nbatch_rewards, nbatch_next_obs, nbatch_dones = nbatch
                nbatch_rewards = nbatch_rewards.reshape(-1, 1)
                nbatch_dones = nbatch_dones.reshape(-1, 1)
                _,_, ex_nbatch_rewards, ex_nbatch_next_obs, ex_nbatch_dones = ex_nbatch
                ex_nbatch_rewards = ex_nbatch_rewards.reshape(-1, 1)
                ex_nbatch_dones = ex_nbatch_dones.reshape(-1, 1)

                nbatch_rewards = np.vstack((nbatch_rewards,ex_nbatch_rewards))
                nbatch_next_obs = np.vstack((nbatch_next_obs,ex_nbatch_next_obs))
                nbatch_dones = np.vstack((nbatch_dones,ex_nbatch_dones))
                #print(nbatch_dones.shape,ex_nbatch_dones.shape)
    
        else:
            
            if not self.prioritized_replay:
                batch = self.expert_buffer.sample(self.batch_size)
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones,batch_demos,batch_idx = batch
                weight= np.ones_like(batch_rewards)
            else:
                batch = self.expert_buffer.sample(self.batch_size,beta=self.beta_schedule.value(self.step))
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones,batch_demos,weight,batch_idx = batch
            batch_rewards = batch_rewards.reshape(-1, 1)
            batch_dones = batch_dones.reshape(-1, 1)
            batch_demos = batch_demos.reshape(-1, 1) 
            weight = weight.reshape(-1,1)
            if self.n_step:
                nbatch = self.expert_N_buffer.sample(batch_idx)
                _,_, nbatch_rewards, nbatch_next_obs, nbatch_dones = nbatch
                nbatch_rewards = nbatch_rewards.reshape(-1, 1)
                nbatch_dones = nbatch_dones.reshape(-1, 1)
        
        if self.n_step:
            feed_dict = {
                self.observations_ph: batch_obs,
                self.actions_ph: batch_actions,
                self.next_observations_ph: batch_next_obs,
                self.weight_ph:weight,
                self.rewards_ph: batch_rewards,
                self.is_demo_ph:batch_demos,
                self.terminals_ph: batch_dones,
                self.learning_rate_ph: learning_rate,
                self.next_observations_ph_n: nbatch_next_obs,
                self.rewards_ph_n: nbatch_rewards,
                self.terminals_ph_n: nbatch_dones,
                self.is_demo_ph:batch_demos

            }
        else:
            feed_dict = {
                self.observations_ph: batch_obs,
                self.actions_ph: batch_actions,
                self.next_observations_ph: batch_next_obs,
                self.weight_ph:weight,
                self.rewards_ph: batch_rewards,
                self.is_demo_ph:batch_demos,
                self.terminals_ph: batch_dones,
                self.learning_rate_ph: learning_rate
            }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None and not pretrain:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]
        actor_for_priority = values[5]
        value_for_priority = values[6]
        imitation_for_priority = values[7]
        actor_loss_di = values[8]
        #print(values[0].shape,values[2].shape)
        #print(actor_for_priority.shape,value_for_priority.shape)
        if self.prioritized_replay:
            if not pretrain:
                td = self.prioritized_replay_eps + 1*(actor_for_priority**2)[:int(self.batch_size*a),] + value_for_priority[:int(self.batch_size*a),]
                td_expert = self.prioritized_replay_eps + 1*(imitation_for_priority)[int(self.batch_size*a):,] + value_for_priority[int(self.batch_size*a):,]
                self.replay_buffer.update_priorities(batch_idx, td)
                self.expert_buffer.update_priorities(exp_batch_idx, td_expert)
            else:
                td = self.prioritized_replay_eps + 1*actor_for_priority**2 + value_for_priority
                self.expert_buffer.update_priorities(batch_idx, td)


        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef,actor_loss_di,one_batch_r,exp_batch_rewards

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy,actor_loss_di,one_batch_r,exp_batch_rewards

    def learn(self, total_timesteps,pretrain_steps,mean_expert_reward, callback=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            self.step = 0
            if pretrain_steps is not 0:
                self.pretrain_sac(pretrain_steps)
            # Initial learning rate
            current_lr = self.learning_rate(1)


            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()

            n_updates = 0
            buffer_n = deque(maxlen=self.n_step_length)
            infos_values = []
            
            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()
            print("=====SAC Exploring=====")
            all_r = []
            all_exp_r=[]
            all_r_step = []
            all_exp_r_step=[]
            for step in range(total_timesteps):
                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if self.num_timesteps < self.learning_starts or np.random.rand() < self.random_exploration:
                    # actions sampled from action space are from range specific to the environment
                    # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                    unscaled_action = self.env.action_space.sample()
                    action = scale_action(self.action_space, unscaled_action)
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # inferred actions need to be transformed to environment action_space before stepping
                    unscaled_action = unscale_action(self.action_space, action)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(unscaled_action)

                self.num_timesteps += 1

                # Only stop training if return value is False, not when it is None. This is for backwards
                # compatibility with callbacks that have no return statement.
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, reward

                if self.n_step:
                    trans = (obs_, action, reward_, new_obs_, float(done))
                    buffer_n.append(trans)
                    self.replay_N_buffer.add((obs_, action, reward_, new_obs_, float(done)))
                    if len(buffer_n)==self.n_step_length:
                        #self.replay_buffer.add(obs_, action, reward_, new_obs_, float(done),0)
                        one_step = buffer_n[0]
                        self.replay_buffer.add(one_step[0], one_step[1], one_step[2], one_step[3], float(one_step[4]),0)

                        
                else:
                    # Store transition in the replay buffer.
                    self.replay_buffer.add(obs_, action, reward_, new_obs_, float(done),0)
                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    self.ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_reward,
                                                        ep_done, writer, self.num_timesteps)

                if step % self.train_freq == 0:
                    callback.on_rollout_end()

                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        infoss = self._train_step(step, writer, current_lr)
                        all_r.append(np.sum(infoss[-2]))
                        all_exp_r.append(np.sum(infoss[-1]))
                        all_r_step.append(infoss[-2].shape[0])
                        all_exp_r_step.append(infoss[-1].shape[0])
                        mb_infos_vals.append(infoss[:-2])
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)
                    

                    callback.on_rollout_start()
                if step % self.update_buffer_interval ==0 and step>self.learning_starts:
                    mean_agent = sum(all_r)/sum(all_r_step) 
                    mean_exp = sum(all_exp_r)/sum(all_exp_r_step) 
                    add_r = mean_agent>mean_exp-0.5
                    all_r = []
                    all_exp_r = []
                    all_r_step = []
                    all_exp_r_step = []
                    
                    if add_r:
                        self.ratio = min(self.ratio+2/self.batch_size,self.max_ratio)
                    else:
                        self.ratio = max(self.ratio-1/self.batch_size,self.init_ratio)
                    print('|new-ratio:',self.ratio,'|mean-agent:',mean_agent,'|mean-exp:',mean_exp-0.5,'|')
                    smry = tf.Summary(value=[tf.Summary.Value(tag="ratio", simple_value=self.ratio)])
                    writer.add_summary(smry,step)
                episode_rewards[-1] += reward_
                if done:
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    #if episode_rewards[-1] >= mean_expert_reward:
                    #    self.ratio  = np.clip((self.ratio+1/self.batch_size),0,60/self.batch_s
                    

                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            callback.on_training_end()
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and outputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions) # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
