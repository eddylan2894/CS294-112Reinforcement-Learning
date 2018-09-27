# CS294-112 Deep Reinforcement Learning Homerwork 2 - Policy Gradient

This project implements a policy gradient algorithm, the functionality includes reward to go, advantage centering and state-dependent baseline to make accelerate the learning process. The `train_pg_f18.py` is our main file and the training information and results will be recorded in the `data` folder. After training, run `plot.py` to plot out the learning rate curves. The commands are listed as below regarding different environments.

Note: Replace the `lunarlander.py` file in your gym directory with the provided file.
### CartPole
In the directory, type `./cartpole.sh` to run the experiment. The bash file runs the algorithm under the following different settings:
1. small batch (1000), no reward-to-go, no advantage centering
2. small batch (1000), with reward to go, no advantage centering
3. small batch (1000), reward to go, advantage centering
4. large batch (5000), no reward-to-go, no advantage centering
5. large batch (5000), with reward to go, no advantage centering
6. large batch (5000), reward to go, advantage centering

The pre-run data is stored in the `./data/sb` and `./data/lb`. Use `python plot.py data/sb/*` and `python plot.py data/lb/*` to plot the  learning curves.
### InvertedPendulum

Use `python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 450 -lr 0.025 -rtg --exp_name hc_b450_r0.025test` to run this experiment.
The pre-run data is stored in the `./data/ip`. Use `python plot.py data/ip/` to plot the learning curves.

### LunarLander
Use `python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005` to run the experiment.
The pre-run data is stored in the `./data/ll`. Use `python plot.py data/ll/` to plot the learning curves.

### HalfCheetah
Use `hc_test.sh` to find the optimal combination of batch size [10000, 20000, 50000] and learning rate [0.005, 0.01, 0.02], and use `hc_run.sh` to run the with batch size 50000 and learning rate 0.025 under the following different settings.
1. advantage centering
2. advantage centering and reward to go
3. advantage centering and baseline
4. advantage centering, reward to go, and baseline

The pre-run data is stored in the `./data/hc_test` and `./data/hc_run`. Use `python plot.py data/hc_test/*` and `python plot.py data/hc_run/*` to plot the learning curves.
