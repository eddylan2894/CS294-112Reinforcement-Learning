#python dag_bonus.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts=10


import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tf_util
import gym
import load_policy
import my_utils

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


num_l = 128

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    file_name = args.envname

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    ### Load Expert Data
    with open('expert_data/'+file_name + '.pkl', 'rb') as f:
        data = pickle.loads(f.read())

    data_ob = data['observations']
    data_ac = data['actions']
    #data_ob = data_ob[:,None,:]
    data_ac = data_ac.reshape(data_ac.shape[0],data_ac.shape[2])
    
    EPOCHS = 50
    d_epochs = 5


    with tf.Session():
        tf_util.initialize()
        #model = keras.models.load_model(args.envname+ '_model.h5')
        model_dag = keras.models.load_model(args.envname+str(num_l)+ '_model.h5')
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        
        exp_mean_records = []
        exp_std_records = []
        bc_mean_records = []
        bc_std_records = []
        dag_mean_records = []
        dag_std_records = []

        for i in range(d_epochs):
            
            print("Running Dagger iteration: {}".format(i))
            ### Train model based on expert data
            model_dag.fit(data_ob, data_ac, epochs=EPOCHS,
                        validation_split=0.2, verbose=0,
                        callbacks=[PrintDot()])

            # Generate new data
            mean, std, observations, actions = my_utils.run_simulation(env, args.num_rollouts, model_dag.predict, args.render, max_steps)
            observations_st = np.array(observations);
            actions_st = np.array(actions)

            ### Expert labels the data
            
            actions_ex = policy_fn(observations_st)
            
            ### Update the model_dag via the new data
            
            data_ob = np.concatenate((data_ob, observations_st))
            data_ac = np.concatenate((data_ac, actions_ex))

            ### Test after one Dagger iteration and save the results
            # Dagger Result
            #print("Testing Dagger")
            #dag_mean, dag_std, __, __ = my_utils.run_simulation(env, args.num_rollouts, model_dag.predict, args.render, max_steps)
            # Expert Result
            #print('Testing Expert')
            #exp_mean, exp_std, __, __ = my_utils.run_simulation(env, args.num_rollouts, policy_fn, args.render, max_steps)
            # Behavior Cloning Result
            #print('Testing BC')
            #bc_mean, bc_std, __, __ = my_utils.run_simulation(env, args.num_rollouts, model.predict, args.render, max_steps)

            #dag_mean_records.append(dag_mean)
            #dag_std_records.append(dag_std)

            #exp_mean_records.append(exp_mean)
            #exp_std_records.append(exp_std)

            #bc_mean_records.append(bc_mean)
            #bc_std_records.append(bc_std)

        model_dag.save(file_name+str(num_l)+'_DaggerModel.h5')
        
        #dic = {'Expert': (exp_mean_records, exp_std_records), 'Dagger': (dag_mean_records,dag_std_records), 'BC':(bc_mean_records,bc_std_records)}
        #my_utils.pltbars( args.envname , dic)


if __name__ == '__main__':
    main()
