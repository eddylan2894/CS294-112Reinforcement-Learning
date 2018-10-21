# ref : https://github.com/SpinachR/ubuntuTest/blob/master/beautifulCodes/tensorflow_advanced_index_slicing.ipynb
# ref : https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf
import uuid
import time
import pickle
import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import my_utils
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):

  def __init__(
    self,
    env,
    q_func,
    optimizer_spec,
    session,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10,
    rew_file=None,
    double_q=True,
    lander=False):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.session = session
    self.exploration = exploration
    self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file

    ###############
    # BUILD MODEL #
    ###############
    print("Double Q = "+str(double_q) )
    if len(self.env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = self.env.observation_space.shape
    else:
        img_h, img_w, img_c = self.env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    self.num_actions = self.env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    self.obs_t_ph              = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    self.act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    self.rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    self.obs_tp1_ph            = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    self.done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    if lander:
      obs_t_float = self.obs_t_ph
      obs_tp1_float = self.obs_tp1_ph
    else:
      obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
      obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
    ######

    # MY CODE HERE

    ######
    # for logging
    #self.time_log, self.mean_rew_log, self.episode_log, self.max_rew_log = [], [], [], []
    self.logg = []
    self.tau = tf.placeholder(shape = None, dtype=tf.float32)

    q_f = q_func(obs_t_float, self.num_actions, scope="q_func", reuse=False)
    self.action_q = tf.argmax( q_f, 1, output_type = tf.int32)

    # Boltzmann softmax
    self.bolt_exp = tf.nn.softmax(q_f/self.tau)
    # add q(s,a)
    if double_q:
      q_f_n = q_func(obs_tp1_float, self.num_actions, scope = "q_func", reuse = True)

    self.q_target = q_func(obs_tp1_float, self.num_actions, scope="q_target", reuse=False)

    #if self.initialized:
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')
    #else:
    #  q_target = self.q_func(obs_tp1_float, self.num_actions, scope="#", reuse=True)
    #  target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    
    # modify here
    #q_t = my_utils.idx_qfunc(q_f, self.act_t_ph)
    tmp = tf.multiply (tf.one_hot(self.act_t_ph, depth=q_f.shape[1] ), q_f)
    q_t = tf.reduce_sum(tmp, axis = 1)

    #if self.done_mask_ph[0] == 1.0:
      # done
      #print("done")
      #y = self.rew_t_ph
    #else:
      #print("done")
    if double_q:
      a_fn = tf.argmax(q_f_n, 1, output_type = tf.int32)
      #print(a_fn.shape)
      #print(self.q_target.shape)
      #exit()
      q_target_fn = tf.multiply(tf.one_hot(a_fn,depth=self.q_target.shape[1]), self.q_target)
      y = self.rew_t_ph + gamma * (1 - self.done_mask_ph) * tf.reduce_sum(q_target_fn, axis=1)
      y = tf.stop_gradient(y)
    else:
      y = self.rew_t_ph + gamma * (1 - self.done_mask_ph) * tf.reduce_max(self.q_target, axis=1)
         
    self.total_error = tf.reduce_mean( huber_loss( q_t - y ) )
    #self.total_error = huber_loss( q_t - y )

    
    # MY CODE ENDS HERE

    # construct optimization op (with gradient clipping)
    self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
    self.train_fn = minimize_and_clip(optimizer, self.total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    self.update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############
    self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.last_obs = self.env.reset()
    self.log_every_n_steps = 10000

    self.start_time = None
    self.t = 0

  def stopping_criterion_met(self):
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self):
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Here, your code needs to store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.
    # At the end of this block of code, the simulator should have been
    # advanced one step, and the replay buffer should contain one more
    # transition.
    # Specifically, self.last_obs must point to the new latest observation.
    # Useful functions you'll need to call:
    # obs, reward, done, info = env.step(action)
    # this steps the environment forward one step
    # obs = env.reset()
    # this resets the environment if you reached an episode boundary.
    # Don't forget to call env.reset() to get a new observation if done
    # is true!!
    # Note that you cannot use "self.last_obs" directly as input
    # into your network, since it needs to be processed to include context
    # from previous frames. You should check out the replay buffer
    # implementation in dqn_utils.py to see what functionality the replay
    # buffer exposes. The replay buffer has a function called
    # encode_recent_observation that will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    # Don't forget to include epsilon greedy exploration!
    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)

    #####

    # YOUR CODE HERE
    # store the last observation
    #if(self.model_initialized == False):
    #  self.session.run(tf.global_variables_initializer())
    #  self.model_initialized = True

    idx = self.replay_buffer.store_frame(self.last_obs)
    #print(self.replay_buffer.size)
    # get the context of the last observation
    # make a prediction
    #action_q = my_utils.oh_action(self.q_f)


    #schedule for probability of chosing random action
    if (self.model_initialized == False):
      action = np.random.randint(0, self.num_actions)

    else:
      lo_c = self.replay_buffer.encode_recent_observation()
      #action = self.session.run(self.action_q, feed_dict={self.obs_t_ph : lo_c[None,:]})
      action = self.session.run(self.bolt_exp, feed_dict={self.obs_t_ph : lo_c[None,:], self.tau: self.exploration.value(self.t)})
      action = action[0]
      #print(action.shape)
      #exit()
      action = np.random.choice(self.num_actions, size = 1 , p = action)
      action = action[0]
      # Boltzmann softmax 
      #if ( np.random.binomial(1, self.exploration.value(self.t)) == 1):
      # Choose randomly among the suboptimal actions
      #  while(1):
      #    action_r = np.random.randint(0, self.num_actions)
      #    if(action_r != action):
      #      action = action_r
      #      break


    obs, reward, done, info = self.env.step(action)#[action.shape[0]-1])
    # store the whole step (o,a,r,s')
    self.replay_buffer.store_effect( idx, action, reward, done)    
    
    if done:
      obs = self.env.reset()
    self.last_obs = obs
    # last_obs point to the new latest observation
  def update_model(self):
    ### 3. Perform experience replay and train the network.
    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
      # Here, you should perform training. Training consists of four steps:
      # 3.a: use the replay buffer to sample a batch of transitions (see the
      # replay buffer code for function definition, each batch that you sample
      # should consist of current observations, current actions, rewards,
      # next observations, and done indicator).
      # 3.b: initialize the model if it has not been initialized yet; to do
      # that, call
      #    initialize_interdependent_variables(self.session, tf.global_variables(), {
      #        self.obs_t_ph: obs_t_batch,
      #        self.obs_tp1_ph: obs_tp1_batch,
      #    })
      # where obs_t_batch and obs_tp1_batch are the batches of observations at
      # the current and next time step. The boolean variable model_initialized
      # indicates whether or not the model has been initialized.
      # Remember that you have to update the target network too (see 3.d)!
      # 3.c: train the model. To do this, you'll need to use the self.train_fn and
      # self.total_error ops that were created earlier: self.total_error is what you
      # created to compute the total Bellman error in a batch, and self.train_fn
      # will actually perform a gradient step and update the network parameters
      # to reduce total_error. When calling self.session.run on these you'll need to
      # populate the following placeholders:
      # self.obs_t_ph
      # self.act_t_ph
      # self.rew_t_ph
      # self.obs_tp1_ph
      # self.done_mask_ph
      # (this is needed for computing self.total_error)
      # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
      # (this is needed by the optimizer to choose the learning rate)
      # 3.d: periodically update the target network by calling
      # self.session.run(self.update_target_fn)
      # you should update every target_update_freq steps, and you may find the
      # variable self.num_param_updates useful for this (it was initialized to 0)
      #####

      # YOUR CODE HERE
      # sample a batch
      obs_c, act_c, rew_c, obs_n, don = self.replay_buffer.sample(batch_size = self.batch_size)
      # initialize the model if haven't done yet
      if self.model_initialized == False:
        initialize_interdependent_variables(self.session, tf.global_variables(), {
             self.obs_t_ph: obs_c,
             self.obs_tp1_ph: obs_n,
        })
        self.model_initialized = True
      # for loop here?
      _ , error= self.session.run([self.train_fn, self.total_error], feed_dict={self.obs_t_ph : obs_c,
                                                                       self.act_t_ph : act_c,
                                                                       self.rew_t_ph : rew_c,
                                                                       self.obs_tp1_ph : obs_n,
                                                                       self.done_mask_ph : don,
                                                                       self.learning_rate : self.optimizer_spec.lr_schedule.value(self.t)
                                                                       })
      # print(error)
      self.num_param_updates += 1
      # update variables
      if(self.num_param_updates % self.target_update_freq == 0):
        self.session.run( self.update_target_fn)
    self.t += 1

  def log_progress(self):
    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

    if len(episode_rewards) > 0:
      self.mean_episode_reward = np.mean(episode_rewards[-100:])

    if len(episode_rewards) > 100:
      self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    if self.t % self.log_every_n_steps == 0 and self.model_initialized:
      print("Timestep %d" % (self.t,))
      print("mean reward (100 episodes) %f" % self.mean_episode_reward)
      print("best mean reward %f" % self.best_mean_episode_reward)
      print("episodes %d" % len(episode_rewards))
      print("exploration %f" % self.exploration.value(self.t))
      print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
      print("\n")
      if self.start_time is not None:
        print("running time %f" % ((time.time() - self.start_time) / 60.))


      #self.start_time = time.time()
      #self.time_log.append(self.t)
      #self.mean_rew_log.append(self.mean_episode_reward)
      #self.episode_log.append(len(episode_rewards))
      #self.max_rew_log.append(self.best_mean_episode_reward)
      sys.stdout.flush()
      self.logg.append([self.t, self.mean_episode_reward, self.best_mean_episode_reward, len(episode_rewards)])
      with open(self.rew_file, 'wb') as f:
        pickle.dump(self.logg, f, pickle.HIGHEST_PROTOCOL)
      #with open(self.rew_file, 'wb') as f:
      #  pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)
  """
  def save_log_file(self):
    log_dic = {"time":self.time_log,
    "mean_episode_reward": self.mean_rew_log,
    "episode_log": self.episode_log,
    "best_mean_episode_reward":self.max_rew_log}
    with open(self.rew_file, 'wb') as f:
      pickle.dump(log_dic, f, pickle.HIGHEST_PROTOCOL)
  """
def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  while not alg.stopping_criterion_met():
    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    alg.update_model()
    alg.log_progress()
  #alg.save_log_file()


  

