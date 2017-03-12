import gym
import numpy as np
import random
import os

from datetime import datetime
from os.path import isfile
from keras.models import Sequential
from keras.optimizers import sgd, adam
from keras.layers.core import Dense, Activation, Flatten


class DQN():
    def __init__(self, epsilon=0.9, epsilon_min=0.01, epsilon_decay = 10000, total_episodes = 1000000,
                 memory_max_replay = 10000, memory_batch_size = 500, swap_nn_weights_step_count = 5000,
                 weights_file = "Breakout_NN_QLearning_Weights.h5", render = False, loadWeights = False):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.render = render
        self.loadWeights = loadWeights
        self.total_episodes = total_episodes
        self.memory_max_replay = memory_max_replay
        self.memory_batch_size = memory_batch_size
        self.swap_nn_weights_step_count = swap_nn_weights_step_count
        self.weights_file = weights_file

    def run(self):
        env = gym.make('Breakout-ram-v0')

        print('Action space:', env.action_space, env.action_space.sample())
        print('Observation space:', env.observation_space)
        print("cart_position, pole_angle, cart_velocity, angle_rate_of_change")

        action_model = self.__create_nn_model()
        learning_model = self.__create_nn_model()
        self.__load_weights(action_model, learning_model)
        expReplay = ExperienceReplay(max_memory=self.memory_max_replay)
        totalSteps = 0

        for episode in range(self.total_episodes):
            stepCount = 0
            processed_state = np.zeros((4,128))
            state = self.__process_gym_state(env.reset(), processed_state)
            done = False
            score = 0
            while done == False:
                if self.render:
                    env.render()
                action = self.__get_epsilon_action(action_model, state, env)
                nextState, reward, done, info = env.step(action)
                nextState = self.__process_gym_state(nextState, state)
                expReplay.store_transition(state, action, reward*100, nextState, done)
                score += reward
                stepCount += 1
                totalSteps += 1
                state = nextState

            inputs, targets = expReplay.random_mini_batch(model=action_model, batch_size=self.memory_batch_size)
            print(targets)
            loss = learning_model.train_on_batch(inputs, targets)
            self.__update_epsilon_decay()
            self.__save_weights(learning_model)
            print(str(datetime.now()), " Episode: ", episode, " epsilon ",  '%0.6f' % self.epsilon, " loss ", '%0.7f' % loss, " score ", score, " Done in ", stepCount)
            if self.swap_nn_weights_step_count < totalSteps:
                action_model.load_weights(self.weights_file)
                print("   Set action_model weights to learning_model at ", totalSteps, " steps")
                totalSteps = 0

    def __create_nn_model(self):
        model = Sequential()
        model.add(Dense(6, bias=False, input_shape=(4,128)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(6, bias=False))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer=adam(lr=0.1))
        return model

    def __get_epsilon_action(self, model, currentState, env):
        if random.random() < self.epsilon:
            action = random.choice([2,3,4])
            return action
        else:
            actionQvalues = model.predict(currentState)
            action = np.argmax(actionQvalues[0])
            return action

    def __update_epsilon_decay(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon -= (1 / (self.epsilon_decay))

    def __process_gym_state(self, gym_state, processed_state):
        state = np.append(processed_state, gym_state)
        state = np.reshape(state, (5,128))
        state = np.delete(state, 0, 0)
        state = np.reshape(state, (1, 4, 128))
        return state

    def __save_weights(self, learning_model):
        if isfile(self.weights_file):
            os.remove(self.weights_file)
        learning_model.save_weights(self.weights_file)

    def __load_weights(self, action_model, learning_model):
        if self.loadWeights:
            action_model.load_weights(self.weights_file)
            learning_model.load_weights(self.weights_file)


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

    def random_mini_batch(self, model, batch_size = 50, gamma = 0.99):
        len_memory = len(self.memory)
        batch_size = min(len_memory, batch_size)

        states = []
        targets = []
        for i, idx in enumerate(np.random.randint(0, len_memory, batch_size)):
            m = self.memory[idx]
            state = m["state"]
            action = m["action"]
            reward = m["reward"]
            nextState = m["nextState"]
            isDone = m["done"]

            target = model.predict(state)
            if isDone == True:
                target[0, action] = reward
            else:
                Qvalue = model.predict(nextState)
                Qvalue = np.max(Qvalue )
                target[0, action] = reward + gamma * Qvalue

            states.append(state[0])
            targets.append(target[0])

        states = np.array(states)
        targets = np.array(targets)
        return states, targets

if __name__ == "__main__":
    dqn = DQN()
    dqn.run()