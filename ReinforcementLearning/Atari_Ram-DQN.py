import gym
import numpy as np
import random
import os
import tensorflow as tf

from datetime import datetime
from os.path import isfile
from keras.models import Sequential
from keras.optimizers import sgd, adam, rmsprop
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution1D


class DQN():
    def __init__(self, model_lr=0.001, epsilon=0.1, epsilon_min=0.1, epsilon_decay = 5000000, total_episodes = 1000000,
                 memory_max_replay = 10000, mini_batch_size = 128, target_network_update_freq = 10000,
                 gym_game = 'Pong-ram-v0', gym_action_num = 2, gym_action_shift = 2,
                 render = False, loadWeights = True):
        '''
        :param model_lr:  adam optimizer learning rate
        :param epsilon:   inital epsilon for epsilon greedy exploration algorithm
        :param epsilon_min:  final epsilon for epsilon greedy exploration algorithm
        :param epsilon_decay:  steps requried before epislon reaches epsilon_min.
                               using the formula:  epsilon = epsilon - (1 / (epsilon_decay))
        :param total_episodes:  the total number of episodes that the program will execute
        :param memory_max_replay:  number of State, Action, Rewards, nextState moves that are stored in the reply memory
        :param mini_batch_size:  size of mini batch that is randomly sampled from replay memory
        :param swap_nn_weights_step_count:  number of steps taken before the target-action-value neural network model
                                            loads the weights from the action-value neural network model
        :param gym_game:  select a non ram game from https://gym.openai.com/envs#atari (e.g. Breakout-v0)
        :param gym_action_num:  number of actions that a game has. (e.g. Breakout has 3 actions (https://gym.openai.com/envs/Breakout-v0))
        :param gym_action_shift:  game actions starting index.  (e.g. Breakout starting index is 2, because the actions are {2,3,4})
        :param render:  True to display the game.  False to not.
        :param loadWeights:  Load weights from previous run of game.
        '''
        self.model_lr = model_lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.render = render
        self.loadWeights = loadWeights
        self.total_episodes = total_episodes
        self.memory_max_replay = memory_max_replay
        self.mini_batch_size = mini_batch_size
        self.target_network_update_freq = target_network_update_freq
        self.gym_action_num = gym_action_num
        self.gym_action_shift = gym_action_shift
        self.gym_game = gym_game
        self.weights_file = "DQN_Weights_{}.h5".format(self.gym_game)

        np.set_printoptions(precision=4) #Display for Numpy arrays

    def main(self):
        env = gym.make(self.gym_game)

        print('Action space:', env.action_space, env.action_space.sample())
        print('Observation space:', env.observation_space)

        self.target_action_value = self.__create_nn_model()
        self.action_value = self.__create_nn_model()
        self.__load_weights()
        expReplay = ExperienceReplay(max_memory=self.memory_max_replay, batch_size=self.mini_batch_size)
        totalSteps = 0

        for episode in range(self.total_episodes):
            stepCount = 0
            score = 0
            loss = 0
            done = False

            state = self.__process_gym_state(env.reset())

            #Play Game and store SARS in Replay Memory
            while done == False:
                if self.render:
                    env.render()
                action = self.__get_epsilon_action(state)
                nextState, reward, done, info = env.step(action + self.gym_action_shift)
                nextState = self.__process_gym_state(nextState)
                expReplay.store_transition(state.reshape(128), action, reward, nextState.reshape(128), done)
                score += reward
                stepCount += 1
                totalSteps += 1
                state = nextState
                self.__update_epsilon_decay()

                #Get mini batch to train network.  This method calcuates the target values
                inputs, targets = expReplay.random_mini_batch(action_value = self.action_value,
                                                              target_action_value=self.target_action_value)
                loss += self.action_value.train_on_batch(inputs, targets)

            self.__save_weights(self.action_value)

            print(str(datetime.now()), expReplay.memory_size(), " Episode: ", episode, " epsilon ",  '%0.4f' % self.epsilon,
                  " loss ", (loss/stepCount), " step: ", stepCount, " total Steps ", totalSteps, " score ", score,
                  "\tSample Q-Values: ",targets[0:1], ",\t", targets[1:2], ",\t", targets[2:3])

            totalSteps = self.__update_target_action_value(totalSteps)

    def __create_nn_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=128, bias=True))
        model.add(Activation('relu'))
        model.add(Dense(256, bias=True))
        model.add(Activation('relu'))
        model.add(Dense(64, bias=True))
        model.add(Activation('relu'))
        model.add(Dense(self.gym_action_num, bias=True))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer=rmsprop(lr=self.model_lr))
        return model

    def __get_epsilon_action(self, currentState):
        if random.random() < self.epsilon:
            action = random.choice([i for i in range(0, self.gym_action_num)])
            return action
        else:
            actionQvalues = self.action_value.predict(currentState)
            action = np.argmax(actionQvalues[0])
            return action

    def __update_epsilon_decay(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon -= (1 / (self.epsilon_decay))

    def __update_target_action_value(self, totalSteps):
        if self.target_network_update_freq < totalSteps:
            self.target_action_value.load_weights(self.weights_file)
            print("   Set target_action_value weights to action_value at ", totalSteps, " steps")
            totalSteps = 0
        return totalSteps

    def __process_gym_state(self, gym_state):
        state = np.reshape(gym_state, (1, 128))
        return state

    def __save_weights(self, model):
        if isfile(self.weights_file):
            os.remove(self.weights_file)
        model.save_weights(self.weights_file)

    def __load_weights(self):
        if self.loadWeights and isfile(self.weights_file):
            self.target_action_value.load_weights(self.weights_file)
            self.action_value.load_weights(self.weights_file)


class ExperienceReplay():
    def __init__(self, max_memory=1000, batch_size = 32):
        self.memory_state = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_nextState = []
        self.memory_done = []

        self.max_memory = max_memory
        self.batch_size = batch_size

    def store_transition(self, state, action, reward, nextState, done):
        self.memory_state.append(state)
        self.memory_action.append(action)
        self.memory_reward.append(reward)
        self.memory_nextState.append(nextState)
        self.memory_done.append(done)
        if self.max_memory < len(self.memory_state):
            self.memory_state.pop(0)
            self.memory_action.pop(0)
            self.memory_reward.pop(0)
            self.memory_nextState.pop(0)
            self.memory_done.pop(0)

    def memory_size(self):
        return len(self.memory_state)

    def random_mini_batch(self, action_value, target_action_value, gamma = 0.99):
        memory_length = len(self.memory_state)
        batch_size = min(memory_length, self.batch_size)
        randidx = np.random.choice(memory_length, size=batch_size, replace=False)

        mini_batch_states = np.take(self.memory_state, randidx, axis=0)
        mini_batch_actions = np.take(self.memory_action, randidx, axis=0)
        mini_batch_rewards = np.take(self.memory_reward, randidx, axis=0)
        mini_batch_nextStates = np.take(self.memory_nextState, randidx, axis=0)
        mini_batch_done = np.take(self.memory_done, randidx, axis=0)

        targets = action_value.predict(mini_batch_states)
        Qvalue = target_action_value.predict(mini_batch_nextStates)
        for i, target in enumerate(targets):
            action = mini_batch_actions[i]
            reward = mini_batch_rewards[i]
            done = mini_batch_done[i]
            target[action] = reward
            if done == False:
                q = Qvalue[i]
                target[action] += gamma * np.max(q)

        return mini_batch_states, targets


if __name__ == "__main__":
    #dqn = DQN(gym_game = 'Pong-ram-v0', gym_action_num = 2, gym_action_shift = 2)
    dqn = DQN()
    dqn.main()