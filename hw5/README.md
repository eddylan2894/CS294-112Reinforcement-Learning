# CS294-112 HW 5c: Meta-Learning

In this homework, we implement the Meta Reinforcement Learning with RNN (recurrent neural network).

Problem1:
Use the following command to run the MLP on a task where the task's ID is given to the model explicitly.
`python train_policy.py pm-obs --exp_name problem1 --history 1 -lr 5e-5 -n 200 --num_tasks 4`

Problem2:
Unlike problem 1, the task ID is not given to the model in this problem.
Use `python train_policy.py pm --exp_name <problem2> --history 60 --discount 0.90 -lr 5e-4 -n 60` to run the experiment on the model without RNN architecture and use  
`python train_policy.py pm --exp_name <problem2_r> --history 60 --discount 0.90 -lr 5e-4 -n 60 --recurrent` to run the experiment on the model with RNN architecture.
(You could change the `exp_name` and `history` lengths for this part.)

Problem3:
In this problem, we further test on the training and validation sets' distribution mismatch. The train/test goal, instead of being both set on the whole plane, would be set on alternating checkerboard pattern so that they have different distribution. Use `python train_policy.py pm --exp_name problem3_r_h60_s1 --history 60 --discount 0.90 -lr 5e-4 -n 60 --recurrent --side 1` to test on this problem. You could change the number after `--side` to change the square size (n*n) in the checkerboard pattern.

See the [HW5c PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5c.pdf) for further instructions.
