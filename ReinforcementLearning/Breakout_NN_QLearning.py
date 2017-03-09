import gym
import numpy as np
import random
import os

from datetime import datetime
from os.path import isfile
from keras.models import Sequential
from keras.optimizers import sgd, adam
from keras.layers.core import Dense, Activation


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
                Qvalue = np.max(model.predict(nextState))
                target[0, action] = reward + gamma * Qvalue

            states.append(state[0])
            targets.append(target[0])

        states = np.array(states)
        targets = np.array(targets)
        return states, targets

class DQN():
    def __init__(self, epsilon=0.1, epsilon_min=0.001, epsilon_decay = 5000, total_episodes = 1000000,
                 memory_max_replay = 1000, memory_batch_size = 50,
                 swap_nn_weights_step_count = 5000, render = False, loadWeights=True):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.render = render
        self.loadWeights = loadWeights
        self.total_episodes = total_episodes
        self.memory_max_replay = memory_max_replay
        self.memory_batch_size = memory_batch_size
        self.swap_nn_weights_step_count = swap_nn_weights_step_count
        self.WEIGHT_FILE = "Breakout_NN_QLearning_Weights.h5"

    def create_nn_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=128))
        model.add(Activation('softmax'))
        model.add(Dense(6))
        model.compile(loss='mean_squared_error', optimizer=adam(lr=0.1))
        return model

    def get_epsilon_action(self, model, currentState, env):
        if  random.random() < self.epsilon:
            action = env.action_space.sample()
            return action
        else:
            actionQvalues = model.predict(currentState)
            action = np.argmax(actionQvalues[0])
            return action

    def state_filter(self, state):
        return state.reshape(1, 128)

    def save_weights(self, learning_model):
        if isfile(self.WEIGHT_FILE):
            os.remove(self.WEIGHT_FILE)
        learning_model.save_weights(self.WEIGHT_FILE)

    def load_weights(self, action_model, learning_model):
        if self.loadWeights:
            action_model.load_weights(self.WEIGHT_FILE)
            learning_model.load_weights(self.WEIGHT_FILE)

    def update_epsilon_decay(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon -= (1 / (self.epsilon_decay))

    def run(self):
        env = gym.make('Breakout-ram-v0')
        print('Action space:', env.action_space)
        print('Observation space:', env.observation_space)
        print("cart_position, pole_angle, cart_velocity, angle_rate_of_change")

        action_model = self.create_nn_model()
        learning_model = self.create_nn_model()

        self.load_weights(action_model, learning_model)

        expReplay = ExperienceReplay(max_memory=self.memory_max_replay)
        totalSteps = 0

        for episode in range(self.total_episodes):
            stepCount = 0
            state = self.state_filter(env.reset())
            done = False

            while done == False:
                if self.render:
                    env.render()
                action = self.get_epsilon_action(action_model, state, env)
                nextState, reward, done, info = env.step(action)
                nextState = self.state_filter(nextState)

                expReplay.store_transition(state, action, reward, nextState, done)

                stepCount += 1
                totalSteps += 1
                state = nextState

            inputs, targets = expReplay.random_mini_batch(model=action_model, batch_size=self.memory_batch_size)

            loss = learning_model.train_on_batch(inputs, targets)

            self.update_epsilon_decay()

            self.save_weights(learning_model)

            print(str(datetime.now()), " Episode: ", episode, " epsilon ",  '%0.6f' % self.epsilon, " loss ", '%0.7f' % loss, " Done in ", stepCount)

            if self.swap_nn_weights_step_count < totalSteps:
                action_model.load_weights(self.WEIGHT_FILE)
                print("   Set action_model weights to learning_model at ", totalSteps, " steps")
                totalSteps = 0


if __name__ == "__main__":
    dqn = DQN(epsilon = 0.5, render=False, loadWeights=True)
    dqn.run()