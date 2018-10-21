#python q2_1.py Reacher-v2 Hopper-v2 --num_rollouts=20


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
    parser.add_argument('envname1', type=str)
    parser.add_argument('envname2', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()


    env_name = [ args.envname1, args.envname2]
    with tf.Session():
        exp_m =[]
        exp_s = []
        st_m = []
        st_s = []

        for ev in env_name :
            tf_util.initialize()
            model = keras.models.load_model('Trained_model'+ ev +'_model.h5')
            policy_fn = load_policy.load_policy('experts/'+ev+'.pkl')

            import gym
            env = gym.make(ev)
            max_steps = args.max_timesteps or env.spec.timestep_limit
            print(ev)
            exp_mean, exp_std, __, __ = my_utils.run_simulation(env, args.num_rollouts, policy_fn, args.render, max_steps)
            st_mean, st_std, __, __ = my_utils.run_simulation(env, args.num_rollouts, model.predict, args.render, max_steps)
            exp_m.append(exp_mean)
            exp_s.append(exp_std)
            st_m.append(st_mean)
            st_s.append(st_std)
        i = 0
        for ev in env_name :
            print(ev)
            print('Expert\'s Mean: {} Std: {}'.format(exp_m[i],exp_s[i]))
            print('Student\'s Mean: {} Std: {}'.format(st_m[i],st_s[i]))
            i += 1;


if __name__ == '__main__':
    main()
