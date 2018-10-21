## Homework 1 - Imitation Learning

The bahavior cloning and DAgger algorithms are implemented in this homework. Behavior cloning is simply performing supervise learning on experts' actions. However, the training data and test data may come from different distribution, thus we utilize the DAgger algorithm to solve this distribution mismatch problem.

First, to generate expert data, run the `demo.bash` file. The file runs the `run_expert.py` with different parameter settings. The experts' data are then stored in the `./experts_data` folder. The pre-run data are already stored in this folder, one can skip this step.

For training behavior cloning model, run the `behavior_cloning file` by typing `python bahivior_cloning.py`. The model consists of two hidden layer and one output layer. We can modify `file_name` and `num_l` variables in the file to change the environments and the size of the hidden layers. The model after training will be store in a `<env_name>_<layer_size>.h5` file. The trained models are sotre in the `./Trained_model` folder.

Use `python q2_1.py Reacher-v2 Hopper-v2 --num_rollouts=20` to run the simulation on Reacher and Hopper tasks. This file loads the policy from pre-train models located in the `./Trained_model` and `experts` folder. Please change the arguments to fit your needs.

Use `python q2_3.py` to train the policy with different number of epochs and see the performance of different models.

Use `python q3_1.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts=10` to compare the performance of DAgger behavior cloning, and the expert in the Hopper environment.

Implemented DAgger algorithm: The algorithm repeats four steps- train the model (50 epochs), generate observation data by running simulations on the model, let the expert label the generated data, and finally concatenate the new labeled data to the original expert data set. The DAgger algorithm steps were repeated five times and each time the model generates the same amount of the original data (10 rollouts), and thus the data would be 5 times bigger than the original one.

Use `python bonus.py Hopper-v2 --num_rollouts=20` to see the performance under different model size. (2 hidden layers size of (32, 64 or 128) and one output layers). The preformance are tested with models trained 300 epochs with 10 rollouts of expert's data.
