import tensorflow as tf
import numpy as np

import utils


class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1):
        self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._init_dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3

        self._sess, self._state_ph, self._action_ph, self._next_state_ph,\
            self._next_state_pred, self._loss, self._optimizer, self._best_action = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        state_ph = tf.placeholder(tf.float32, [None, self._state_dim])
        action_ph = tf.placeholder(tf.float32, [None, self._action_dim])
        next_state_ph = tf.placeholder(tf.float32, [None, self._state_dim])

        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state

            implementation details (in order):
                (a) Normalize both the state and action by using the statistics of self._init_dataset and
                    the utils.normalize function
                (b) Concatenate the normalized state and action
                (c) Pass the concatenated, normalized state-action tensor through a neural network with
                    self._nn_layers number of layers using the function utils.build_mlp. The resulting output
                    is the normalized predicted difference between the next state and the current state
                (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
                    the predicted next state

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        #print(self._init_dataset.shape)
        # Normalize state and action
        #d_mean, d_std = np.mean(self._init_dataset), np.std(self._init_dataset)
        n_state = utils.normalize(state, self._init_dataset.state_mean, self._init_dataset.state_std )
        n_action = utils.normalize(action, self._init_dataset.action_mean, self._init_dataset.action_std)
        # Predict next state based on the current state and action
        n_sa = tf.concat((n_state, n_action), axis=1)
        n_nsp = utils.build_mlp(n_sa, output_dim=self._state_dim, scope="f_transition", n_layers=self._nn_layers, reuse=reuse)
        next_state_pred = state + utils.unnormalize(n_nsp, self._init_dataset.delta_state_mean, self._init_dataset.delta_state_std)

        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent

            implementation details (in order):
                (a) Compute both the actual state difference and the predicted state difference
                (b) Normalize both of these state differences by using the statistics of self._init_dataset and
                    the utils.normalize function
                (c) The loss function is the mean-squared-error between the normalized state difference and
                    normalized predicted state difference
                (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # Compute the differences
        as_dif = next_state_ph - state_ph
        ps_dif = next_state_pred - state_ph
        # Normalize the differences
        n_as_dif = utils.normalize(as_dif, self._init_dataset.delta_state_mean, self._init_dataset.delta_state_std)
        n_ps_dif = utils.normalize(ps_dif, self._init_dataset.delta_state_mean, self._init_dataset.delta_state_std)
        loss = tf.nn.l2_loss(n_as_dif - n_ps_dif)
        # Create the optimizer and train
        # ref: https://stackoverflow.com/questions/33788989/tensorflow-using-adam-optimizer
        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])

            implementation details (in order):
                (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
                (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
                    each of length self._horizon
                (c) Starting from the input state, unroll each action sequence using your neural network
                    dynamics model
                (d) While unrolling the action sequences, keep track of the cost of each action sequence
                    using self._cost_fn
                (e) Find the action sequence with the lowest cost, and return the first action in that sequence

            Hints:
                (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
                    2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
                    state or action size
                (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
                (iii) Use tf.random_uniform(...) to generate the random action sequences

        """
        ### PROBLEM 2
        ### YOUR CODE HERE
        """
        action_dic = {}
        for _ in range(self._num_random_action_selection):
            # Test a random action sequence
            # Record the first action
            first_a = tf.random_uniform([1,self._action_dim], minval= self._init_dataset.action_mean-self._init_dataset.action_std, 
                                           maxval= self._init_dataset.action_mean+self._init_dataset.action_std, 
                                           dtype=tf.float32)
            s = self._dynamics_func(state_ph, first_a, reuse=True)
            t_cost = self._cost_fn(state_ph, first_a, s)
            # The rest of the sequence
            for __ in range(self._horizon - 1):
                ra = tf.random_uniform([1,self._action_dim], minval= self._init_dataset.action_mean-self._init_dataset.action_std, 
                                           maxval= self._init_dataset.action_mean+self._init_dataset.action_std, 
                                           dtype=tf.float32)
                pred_s = self._dynamics_func(s, ra, reuse=True)
                t_cost = t_cost + self._cost_fn(s, ra, pred_s)
            # if there is two identical random actions, choose the one with smaller cost; else, store the value
            if first_a in action_dic:
                action_dic[first_a] = min(action_dic[first_a], t_cost)
            else:
                action_dic[first_a] = t_cost
        # Choose the action with minimum cost       
        best_action = min(action_dic, key=action_dic.get)
        """
        ### First 
        first_s = tf.broadcast_to(state_ph, [self._num_random_action_selection ,self._state_dim])
        first_a = tf.random_uniform([self._num_random_action_selection , self._action_dim], 
                                            minval= self._action_space_low, 
                                            maxval= self._action_space_high, 
                                            dtype=tf.float32)
        pred_s = self._dynamics_func(first_s, first_a, reuse=True)
        t_cost = self._cost_fn(first_s, first_a, pred_s)

        for _ in range(self._horizon-1):
            s = pred_s
            ra = tf.random_uniform([self._num_random_action_selection , self._action_dim], 
                                            minval= self._action_space_low, 
                                            maxval= self._action_space_high, 
                                            dtype=tf.float32)
            pred_s = self._dynamics_func(s, ra, reuse=True)
            t_cost = t_cost + self._cost_fn(s, ra, pred_s)

        best_action = first_a[tf.argmin(t_cost),:]

        return best_action

    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        sess = tf.Session()

        ### PROBLEM 1
        ### YOUR CODE HERE
        state_ph, action_ph, next_state_ph = self._setup_placeholders()
        next_state_pred = self._dynamics_func(state_ph, action_ph, reuse=False)
        loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)

        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action = self._setup_action_selection(state_ph)
        #best_action = None
        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, \
                next_state_pred, loss, optimizer, best_action

    def train_step(self, states, actions, next_states):
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        ### PROBLEM 1
        ### YOUR CODE HERE

        _ , loss = self._sess.run([self._optimizer, self._loss], feed_dict={self._state_ph: states,
                                                                            self._action_ph: actions,
                                                                            self._next_state_ph: next_states})
        return loss

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)

        ### PROBLEM 1
        ### YOUR CODE HERE
        next_state_pred = self._sess.run(self._next_state_pred, feed_dict={self._state_ph: state[None,:], self._action_ph: action[None,:]})
        next_state_pred = next_state_pred[0]
        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action = self._sess.run(self._best_action, feed_dict={self._state_ph:state[None,:]})
        #print(np.shape(best_action))
        #best_action = best_action[0]
        assert np.shape(best_action) == (self._action_dim,)
        return best_action
