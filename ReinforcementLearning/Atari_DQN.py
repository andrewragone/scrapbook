import gym
import numpy as np
import random
import os

from datetime import datetime
from os.path import isfile
from keras.models import Sequential
from keras.optimizers import sgd, adam
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D


class DQN():
    def __init__(self, model_lr=0.00001, epsilon=0.99, epsilon_min=0.01, epsilon_decay = 5000, total_episodes = 1000000,
                 memory_max_replay = 3000, memory_batch_size = 300, swap_nn_weights_step_count = 5000,
                 gym_game = 'SpaceInvaders-v0', gym_action_num = 3, gym_action_shift = 2,
                 render = True, loadWeights = True):
        '''
        :param model_lr:  adam optimizer learning rate
        :param epsilon:   inital epsilon for epsilon greedy exploration algorithm
        :param epsilon_min:  final epsilon for epsilon greedy exploration algorithm
        :param epsilon_decay:  decay epislon at using the formula:  epsilon = epsilon - (1 / (epsilon_decay))
        :param total_episodes:  the total number of episodes that the program will execute
        :param memory_max_replay:  number of State, Action, Rewards, nextState moves that are stored in the reply memory
        :param memory_batch_size:  size of mini batch that is randomly sampled from replay memory
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
        self.memory_batch_size = memory_batch_size
        self.swap_nn_weights_step_count = swap_nn_weights_step_count
        self.gym_action_num = gym_action_num
        self.gym_action_shift = gym_action_shift
        self.gym_game = gym_game
        self.weights_file = "{}_NN_QLearning_Weights.h5".format(self.gym_game)


        self.rgb_vector = np.array([.333333, .333333, .333333]).reshape(3,1)
        np.set_printoptions(precision=4)

    def main(self):
        env = gym.make(self.gym_game)

        print('Action space:', env.action_space, env.action_space.sample())
        print('Observation space:', env.observation_space)

        self.target_action_value = self.__create_nn_model()
        self.action_value = self.__create_nn_model()
        self.__load_weights()
        expReplay = ExperienceReplay(max_memory=self.memory_max_replay)
        totalSteps = 0

        for episode in range(self.total_episodes):
            stepCount = 0
            score = 0
            done = False

            processed_state = np.zeros((210, 160, 4))
            state = self.__process_gym_state(env.reset(), processed_state)

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

            inputs, targets = expReplay.random_mini_batch(action_value = self.action_value,
                                                          target_action_value=self.target_action_value,
                                                          batch_size=self.memory_batch_size)
            loss = self.action_value.train_on_batch(inputs, targets)

            self.__save_weights(self.action_value)
            self.__update_epsilon_decay()

            print(str(datetime.now()), " Episode: ", episode, " epsilon ",  '%0.6f' % self.epsilon, " loss ", '%0.7f' % loss, " score ", score, " Done in ", stepCount, " total Steps ", totalSteps)
            print("Top 10 action-value from mini batch sample")
            print(targets[0:10])

            totalSteps = self.__update_target_action_value(totalSteps)

    def __create_nn_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4),   input_shape=(210, 160, 4), bias=False))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), bias=False))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, bias=False))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, bias=False))
        model.add(Activation('relu'))
        model.add(Dense(self.gym_action_num, bias=False))
        model.compile(loss='mean_squared_error', optimizer=adam(lr=self.model_lr))
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
        if self.swap_nn_weights_step_count < totalSteps:
            self.target_action_value.load_weights(self.weights_file)
            print("   Set target_action_value weights to action_value at ", totalSteps, " steps")
            totalSteps = 0
        return totalSteps

    def __process_gym_state(self, gym_state, processed_state):
        gym_state = np.matmul(gym_state, self.rgb_vector).reshape((210, 160))
        state = np.append(processed_state, gym_state)
        state = np.reshape(state, (210, 160, 5))
        state = np.delete(state, 0, 2)
        state = np.reshape(state, (1, 210, 160, 4))
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
    def __init__(self, max_memory=1000):
        self.memory = []
        self.max_memory = max_memory

    def store_transition(self, state, action, reward, nextState, done):
        self.memory.append({"state": state, "action": action, "reward": reward, "nextState": nextState, "done": done})
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def clear_memory(self):
        self.memory.clear()

    def random_mini_batch(self, action_value, target_action_value, batch_size = 50, gamma = 0.99):
        len_memory = len(self.memory)
        batch_size = min(len_memory, batch_size)

        states = []
        targets = []
        for i, idx in enumerate(np.random.randint(0, len_memory, batch_size)):
            m = self.memory[idx]
            state, action, reward, nextState, isDone = m["state"], m["action"], m["reward"], m["nextState"], m["done"]
            target = action_value.predict(state)
            if isDone == True:
                target[0, action] = reward
            else:
                Qvalue = target_action_value.predict(nextState)
                Qvalue = np.max(Qvalue)
                target[0, action] = reward + gamma * Qvalue

            states.append(state[0])
            targets.append(target[0])

        states = np.array(states)
        targets = np.array(targets)
        return states, targets

if __name__ == "__main__":
    dqn = DQN()
    dqn.main()