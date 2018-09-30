#python bonus.py Hopper-v2 --num_rollouts=20


import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tf_util
import gym
import load_policy
import my_utils

def main():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('ev', type=str)
    #parser.add_argument('envname2', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    with tf.Session():
        exp_m =[]
        exp_s = []
        st_m = []
        st_s = []

        ev = args.ev
        tf_util.initialize()
        model_32 = keras.models.load_model( ev +'32_model.h5')
        model_64 = keras.models.load_model( ev +'_model.h5')
        model_128 = keras.models.load_model( ev +'128_model.h5')
        model_32dag = keras.models.load_model( ev +'32_DaggerModel.h5')
        model_64dag = keras.models.load_model( ev +'_DaggerModel.h5')
        model_128dag = keras.models.load_model( ev +'128_DaggerModel.h5')
        policy_fn = load_policy.load_policy('experts/'+ev+'.pkl')

        import gym
        env = gym.make(ev)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        print(ev)
        exp_mean, exp_std, __, __ = my_utils.run_simulation(env, args.num_rollouts, policy_fn, args.render, max_steps)
        mean_32, std_32, __, __ = my_utils.run_simulation(env, args.num_rollouts, model_32.predict, args.render, max_steps)
        mean_64, std_64, __, __ = my_utils.run_simulation(env, args.num_rollouts, model_64.predict, args.render, max_steps)
        mean_128, std_128, __, __ = my_utils.run_simulation(env, args.num_rollouts, model_128.predict, args.render, max_steps)
        mean_32dag, std_32dag, __, __ = my_utils.run_simulation(env, args.num_rollouts, model_32dag.predict, args.render, max_steps)
        mean_64dag, std_64dag, __, __ = my_utils.run_simulation(env, args.num_rollouts, model_64dag.predict, args.render, max_steps)
        mean_128dag, std_128dag, __, __ = my_utils.run_simulation(env, args.num_rollouts, model_128dag.predict, args.render, max_steps)
        
        print('Expert\'s Mean: {} Std: {}'.format(exp_mean, exp_std))
        print('Student(32)\'s Mean: {} Std: {}'.format(mean_32, std_32))
        print('Student(64)\'s Mean: {} Std: {}'.format(mean_64, std_64))
        print('Student(128)\'s Mean: {} Std: {}'.format(mean_128, std_128))
        print('Student(32 DAgger)\'s Mean: {} Std: {}'.format(mean_32dag, std_32dag))
        print('Student(64 DAgger)\'s Mean: {} Std: {}'.format(mean_64dag, std_64dag))
        print('Student(128 DAgger)\'s Mean: {} Std: {}'.format(mean_128dag, std_128dag))
        

if __name__ == '__main__':
    main()
