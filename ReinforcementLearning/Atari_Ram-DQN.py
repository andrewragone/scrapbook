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
    def __init__(self, model_lr=0.0001, epsilon=1.0, epsilon_min=0.1, epsilon_decay = 5000, total_episodes = 1000000,
                 memory_max_replay = 20000, mini_batch_size = 2000, target_network_update_freq = 10000,
                 gym_game = 'SpaceInvaders-ram-v0', gym_action_num = 3, gym_action_shift = 2,
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
            done = False

            processed_state = np.zeros((128, 4))
            state = self.__process_gym_state(env.reset(), processed_state)

            #Play Game and store SARS in Replay Memory
            while done == False:
                if self.render:
                    env.render()
                action = self.__get_epsilon_action(state)
                nextState, reward, done, info = env.step(action + self.gym_action_shift)
                nextState = self.__process_gym_state(nextState, state)
                expReplay.store_transition(state, action, reward, nextState, done)
                score += reward
                stepCount += 1
                totalSteps += 1
                state = nextState

                # #Get mini batch to train network.  This method calcuates the target values
                # inputs, targets = expReplay.random_mini_batch(action_value = self.action_value,
                #                                               target_action_value=self.target_action_value,
                #                                               batch_size=self.mini_batch_size)
                # loss = self.action_value.train_on_batch(inputs, targets)
                # if stepCount%20 == 0:
                #     print("\t",str(datetime.now()), "memory ", expReplay.memory_size(), " epsilon ",  '%0.6f' % self.epsilon,
                #           " step: ", stepCount, " loss ", '%0.5f' % loss,
                #           " score ", score, targets[0:1], targets[1:2], targets[2:3])


            #Get mini batch to train network.  This method calcuates the target values
            print(str(datetime.now()),"Get mini batch to train network.  This method calcuates the target values")
            inputs, targets = expReplay.random_mini_batch(action_value = self.action_value,
                                                          target_action_value=self.target_action_value)
            print(str(datetime.now()),"Calculate Loss")
            loss = self.action_value.train_on_batch(inputs, targets)
            self.__save_weights(self.action_value)
            self.__update_epsilon_decay()

            print(str(datetime.now()), expReplay.memory_size(), " Episode: ", episode, " epsilon ",  '%0.4f' % self.epsilon,
                  " loss ", '%0.5f' % loss, " step: ", stepCount, " total Steps ", totalSteps, " score ", score,
                  "\t",targets[0:1], "\t", targets[1:2], "\t", targets[2:3])


            totalSteps = self.__update_target_action_value(totalSteps)

    def __create_nn_model(self):
        model = Sequential()
        # model.add(Convolution1D(64, 8, subsample_length=4, input_shape=(128, 4), bias=False))
        # model.add(Activation('relu'))
        # model.add(Convolution1D(64, 4, subsample_length=2, bias=False))
        # model.add(Activation('relu'))
        # model.add(Convolution1D(64, 3, subsample_length=2, bias=False))
        # model.add(Activation('relu'))
        model.add(Dense(128, input_shape=(128,4), bias=False))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(64, bias=False))
        model.add(Activation('relu'))
        model.add(Dense(self.gym_action_num, bias=False))
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

    def __process_gym_state(self, gym_state, processed_state):
        #processed_state = np.reshape(processed_state, (512))
        gym_state = np.reshape(gym_state, (128, 1))
        processed_state = np.reshape(processed_state, (128, 4))
        state = np.append(processed_state, gym_state, axis=1)
        state = np.reshape(state, (128, 5))
        state = np.delete(state, 0, 1)
        state = np.reshape(state, (1, 128, 4))
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
        self.memory = np.empty((max_memory, 5), dtype=object)
        self.memory_length  = 0
        self.max_memory = max_memory
        self.batch_size = batch_size

    def store_transition(self, state, action, reward, nextState, done):
        self.memory = np.roll(self.memory,5)
        self.memory[0][0] = state
        self.memory[0][1] = action
        self.memory[0][2] = reward
        self.memory[0][3] = nextState
        self.memory[0][4] = done

        if self.max_memory > self.memory_length:
            self.memory_length += 1

    def memory_size(self):
        return self.memory_length

    def random_mini_batch(self, action_value, target_action_value, gamma = 0.99):

        batch_size = min(self.memory_length -1, self.batch_size)
        randidx = np.random.choice(self.memory_length, size=batch_size, replace=False)
        mini_batch = self.memory[randidx]
        #states = mini_batch[:,0]
        #targets = action_value.predict(states)

        states = []
        targets = []
        for m in mini_batch:
            #m = self.memory[i]
            state, action, reward, nextState, isDone = m[0], m[1], m[2], m[3], m[4]
            target = action_value.predict(state)
            if isDone == True:
                target[0, action] = -1
            else:
                Qvalue = np.max(target_action_value.predict(nextState))
                target[0, action] = reward + gamma * Qvalue

            states.append(state[0])
            targets.append(target[0])

        states = np.array(states)
        targets = np.array(targets)
        return states, targets

if __name__ == "__main__":
    dqn = DQN()
    dqn.main()