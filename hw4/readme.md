## HW4 Model-Based Reinforcement Learning

In this homework we implemented a model-based reinforcement algorithm based on this [paper](https://arxiv.org/pdf/1708.02596.pdf). We train a dynamics model to predict the next state and choose the best action (lowest cost) among the random actions.

### Demo
Use `python main.py q1` to train the model  with the data collected with the random policy. The prediction of your trained model will be store in the `data` folder.

Use `python main.py q2` to train the model and decide action according to it. The return of the model-based reinforcement learning should be better than the random policy.

Use `python main.py q3 --exp_name default` to train the model with new collected rollouts. The return should achieve 300 after 10 iterations. Use `python plot.py --exps HalfCheetah_q3_default --save HalfCheetah_q3_default` to plot the result. Use `./run_q3.bash` to test the algorithm with different parameter settings.

Use `./run_all.sh` to run all the demos.
