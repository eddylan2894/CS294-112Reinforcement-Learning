# CS294-112 HW 3: Q-Learning and Arctic Critic
In this homework, we implement the Q learning and actor critic algorithms.
First replace the `gym/envs/box2d/lunar_lander.py` file with the given file.
### Q-Learning:
The main Q-Learning algorithm is implemented in the `dqn.py` file and it can be run under different environments by running `run_dqn_<>` files. For example, for vanilla Q Learning on the Pong game, run `python run_dqn_atari.py`. For double Q-Learning, change the *double_q* parameter of `dqn.learn()` in the `run_dqn_atari.py` file to *True*. The exploration policy for this part is epsilon-greedy.

I implemented two exploration policies- Boltzmann exploration and Baysian Neural Networks (BNN). For Boltzmann exploration method, run `python run_dqn_atari_exp.py` and for Baysian method, run `python run_dqn_atari_bay.py`. Please look into the codes and tune the temperature of the Boltzmann method and the keep_prob variable in the BNN as you want.

### Actor Critic:

First, we sweep four sets of parameters on a easy task- CartPole- to find the best set:

`python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_1 -ntu 1 -ngsptu 1`

`python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 100_1 -ntu 100 -ngsptu 1`

`python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_100 -ntu 1 -ngsptu 100`

`python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 10_10 -ntu 10 -ngsptu 10`

After finding that 10, 10 is the best setting for target and gradient updates, we run the more complicated tasks on this setting.

InvertedPendulum:
`python train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name i10_10 -ntu 10 -ngsptu 10`

HalfCheetah:
`python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name hc10_10 -ntu 10 -ngsptu 10`

### Visualization:
 The data are stored in a `.pkl` file with a random number filename. Modify `my_plot.py` to plot the results of Q-Learning data by include the files in the *filename* list and modify the plotting settings accordingly (legends, colors and title).

 The actor-critc part's results are store in the `data` folder, run `python plot.py data/*` to multiple results or direct to the selected folder to plot one training result(`python plot.py data/<>`).

 The Q-Learning training data are stored in the `new_data` folder.
