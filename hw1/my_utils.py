import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tf_util

def pltbars( env_name, dic ):
    fig = plt.figure()
    #x_pos = np.arange(1, len(dic[dict.keys()[0]][0]) +1
    i = 0
    p = []
    label = []
    for key in dic:
        x_pos = np.arange(1, len(dic[key][0]) +1 )      
        p.append( plt.errorbar(x_pos, dic[key][0], yerr = dic[key][1], fmt='o-') )
        label.append(key)
        #plt.legend([p[i]], [key])
    #handles, labels = fig.get_legend_handles_labels()
    plt.legend(p, label)

    plt.title('Learning Curve ('+ env_name + ')')
    plt.ylabel('Reward')
    plt.xlabel('Iteration(s)')
    #fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Save the figure and show
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()

def run_simulation(env, num_rollouts,predict_function, render, max_steps):
    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            #action = policy_fn(obs[None,:])
            #print(obs[None,:].shape)
            #print(type(obs))
            #print(obs[:].shape)
            action = predict_function(obs[None,:])
            action = action[None,:]
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    # returns mean, std, observations, actions
    return np.mean(returns) , np.std(returns) , observations , actions

def plt_bars_23( env_name, dic ):
    fig = plt.figure()
    #x_pos = np.arange(1, len(dic[dict.keys()[0]][0]) +1
    i = 0
    p = []
    label = []
    for key in dic:
        x_pos = np.arange(1, len(dic[key][0]) +1 )      
        p.append( plt.errorbar(x_pos, dic[key][0], yerr = dic[key][1], fmt='o-') )
        label.append(key)
        #plt.legend([p[i]], [key])
    #handles, labels = fig.get_legend_handles_labels()
    plt.legend(p, label)

    plt.title('Performance of Different Training Epochs ('+ env_name + ')')
    plt.ylabel('Reward')
    plt.xlabel('Training Epochs(50x)')
    #fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Save the figure and show
    plt.savefig(env_name+'2_3.png')
    plt.show()

def run_simulation_LSTM(env, num_rollouts,model, render, max_steps):
    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        model.reset_states()

        obs_ac = []
        while not done:
            #action = policy_fn(obs[None,:])
            #print(obs[None,:].shape)
            #print(type(obs))
            #print(obs[:].shape)
            #obs_ac.append(obs)
            #obs = np.array(obs_ac)
            #print(obs[None,:].shape)
            action = model.predict(obs[None,None,:])
            #action = action[:,-1,:]
            #print(action.shape)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    # returns mean, std, observations, actions
    return np.mean(returns) , np.std(returns) , observations , actions