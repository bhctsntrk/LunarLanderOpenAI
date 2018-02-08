from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
from itertools import count
from replay_memory import ReplayMemory, Transition
import env_wrappers
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=False, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# README PLEASE
# The algorithm is based on DeepMind's "Playing Atari with Deep Reinforcement Learning" paper, using two networks, training and target
# to train the deep Q learner
# I am not using the given ReplayMemory data structure. Instead I am using a numpy array for storing the replay memory, it tends to be faster on my GPU
# I have used 2 layers with 50 nodes in the first and 40 in the second with relu activation
# I have used huber loss function and AdamOptimizer, I tried RMSProp and GradientDescent it performs better than those.
# I have read and referred sources such as - DeepMind's paper,
# Deep Reinforcement Learning Tutorials by Morvan (YouTube Channel) and Siraj Raval (YouTube Channel)
# I have trained over 455 episodes, as it converges to a 10 game average of 200+ by then
# Learning rate is 0.0005 and epsilon is set to .95 and reduced by .00225 after each episode, batch size is 32
# The Best Model is dqn-model-1000, PLEASE RUN THIS MODEL FOR BEST TRAINED RESULTS, it is included in checkpoint


class DQN(object):
    """
    A starter class to implement the Deep Q Network algorithm

    TODOs specify the main areas where logic needs to be added.

    If you get an error a Box2D error using the pip version try installing from source:
    > git clone https://github.com/pybox2d/pybox2d
    > pip install -e .

    """

    def __init__(self, env):

        self.env = env
        self.sess = tf.Session()

        # A few starter hyperparameters
        self.batch_size = 32
        self.gamma = 0.99
        self.learning_rate = 0.0005
        self.memory_cap = 10000
        # If using e-greedy exploration
        self.eps_start = 0.95
        self.epsilon = self.eps_start
        self.eps_end = 0.05
        self.eps_decay = 40000 # in episodes
        # If using a target network
        self.clone_steps = 500
        self.input_size = env.observation_space.shape[0]
        # memory
        self.replay_memory = ReplayMemory(100000)
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 10000
        self.memory = np.zeros((self.memory_cap, self.input_size * 2 + 2))
        self.actions = env.action_space.n

        # define yours training operations here...
        self.observation_input = tf.placeholder(tf.float32, [None, self.input_size])
        self.observation_input_target = tf.placeholder(tf.float32, [None, self.input_size])
        self.q_target = tf.placeholder(tf.float32, [None, self.actions], name='Q_target')  # for calculating loss
        self.train_network = self.build_model(self.observation_input)
        self.target_network = self.build_model(self.observation_input_target,'target')

        t_params = tf.get_collection('target_params')
        e_params = tf.get_collection('train_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # define your update operations here...
        self.loss = tf.reduce_mean(tf.losses.huber_loss(self.q_target, self.train_network))
        self.reducer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


        self.num_episodes = 0
        self.num_steps = 0
        self.cost_his = []

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, observation_input, scope='train'):
        """
        TODO: Define the tensorflow model

        Hint: You will need to define and input placeholder and output Q-values

        Currently returns an op that gives all zeros.
        """

        with tf.variable_scope(scope):
            namespace, layer1_nodes, layer2_nodes, weights, biases = [scope+'_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, 40, \
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first hidden layer
            w1 = tf.get_variable('w1', [self.input_size, layer1_nodes], initializer=weights, collections=namespace)
            b1 = tf.get_variable('b1', [1, layer1_nodes], initializer=biases, collections=namespace)
            l1 = tf.nn.relu(tf.matmul(observation_input, w1) + b1)

            # second hidden layer
            w2 = tf.get_variable('w2', [layer1_nodes, layer1_nodes], initializer=weights, collections=namespace)
            b2 = tf.get_variable('b2', [1, layer1_nodes], initializer=biases, collections=namespace)
            l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # output layer
            w3 = tf.get_variable('w3', [layer1_nodes, self.actions], initializer=weights, collections=namespace)
            b3 = tf.get_variable('b3', [1, self.actions], initializer=biases, collections=namespace)
        return tf.matmul(l2, w3) + b3


    def select_action(self, obs, evaluation_mode=False):
        """
        TODO: Select an action given an observation using your model. This
        should include any exploration strategy you wish to implement

        If evaluation_mode=True, then this function should behave as if training is
        finished. This may be reducing exploration, etc.

        Currently returns a random action.
        """
        if np.random.uniform() > self.eps_start or evaluation_mode:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.train_network, feed_dict={self.observation_input: obs[np.newaxis, :]})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.actions)


        return action

    def update(self):
        """
        TODO: Implement the functionality to update the network according to the
        Q-learning rule
        """

        if self.num_steps % self.clone_steps == 0:
            self.sess.run(self.replace_target_op)

        if self.num_steps > self.memory_cap:
            sample_index = np.random.choice(self.memory_cap, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.num_steps, size=self.batch_size)
        batch_data = self.memory[sample_index, :]

        target_q, train_q = self.sess.run(
            [self.target_network, self.train_network],
            feed_dict={
                self.observation_input_target: batch_data[:, -self.input_size:],  # fixed params
                self.observation_input: batch_data[:, :self.input_size],  # newest params
            })

        fixed_target = train_q.copy()

        i_train = batch_data[:, self.input_size].astype(int)
        reward = batch_data[:, self.input_size + 1]

        fixed_target[np.arange(self.batch_size, dtype=np.int32), i_train] = reward + self.gamma * np.max(target_q, axis=1)

        _,cost = self.sess.run([self.reducer, self.loss],
                      feed_dict={self.observation_input: batch_data[:, :self.input_size],
                                 self.q_target: fixed_target})
        self.cost_his.append(cost)

    def train(self):
        """
        The training loop. This runs a single episode.

        TODO: Implement the following as desired:
            1. Storing transitions to the ReplayMemory
            2. Updating the network at some frequency
            3. Backing up the current parameters to a reference, target network
        """
        done = False
        obs = env.reset()
        total_reward = 0.0
        while not done:
            action = self.select_action(obs, evaluation_mode=False)
            next_obs, reward, done, info = env.step(action)

            self.memory[self.num_steps % self.memory_cap, :] = np.hstack((obs, [action, reward], next_obs))

            if(self.num_steps>5000):
                self.update()

            total_reward += reward

            obs = next_obs
            self.num_steps += 1
        print("Training Episode #", self.num_episodes, " with reward: ", total_reward, " and eps - ", self.eps_start)
        self.eps_start -= .00225
        self.num_episodes += 1
        return total_reward

    def eval(self, save_snapshot=False):
        """
        Run an evaluation episode, this will call
        """
        total_reward = 0.0
        ep_steps = 0
        done = False
        obs = env.reset()
        while not done:
            env.render()
            action = self.select_action(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print ("Evaluation episode: ", total_reward)
        if save_snapshot:
            print ("Saving state with Saver")
            self.saver.save(self.sess, 'models/dqn-model', dqn.num_episodes)

def plot_cost(episode_list):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(episode_list)), episode_list)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()

def train(dqn):
    episode_history = []
    for i in range(1,455):
        curr = dqn.train()
        env.render()
        episode_history.append(curr)
        # every 10 episodes run an evaluation episode
        if i % 10==0 and i>99:
            dqn.eval()
    plot_cost(episode_history)
    print("Saving state with Saver")
    dqn.saver.save(dqn.sess, 'models/dqn-model', 1000)
    # plot_cost(dqn.cost_his)

def eval(dqn):
    """
    Load the latest model and run a test episode
    """
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/'+model_name))
    dqn.eval(save_snapshot=False)


if __name__ == '__main__':
    # On the LunarLander-v2 env a near-optimal score is some where around 250.
    # Your agent should be able to get to a score >0 fairly quickly at which point
    # it may simply be hitting the ground too hard or a bit jerky. Getting to ~250
    # may require some fine tuning.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)
