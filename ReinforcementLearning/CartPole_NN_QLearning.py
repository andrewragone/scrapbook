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
    def __init__(self, epsilon=0.1, epsilon_min=0.001, epsilon_decay = 5000, total_episodes = 5000,
                 max_step_count_per_episode = 500, memory_max_replay = 500, memory_batch_size = 50,
                 swap_nn_weights_step_count=500, render = False, loadWeights=True):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.render = render
        self.loadWeights = loadWeights
        self.total_episodes = total_episodes
        self.max_step_count_per_episode = max_step_count_per_episode
        self.memory_max_replay = memory_max_replay
        self.memory_batch_size = memory_batch_size
        self.swap_nn_weights_step_count = swap_nn_weights_step_count
        self.WEIGHT_FILE = "CartPole_NN_QLearning_Weights.h5"

    def create_nn_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=4))
        model.add(Activation('softmax'))
        model.add(Dense(2))
        model.compile(loss='mse', optimizer=adam(lr=0.01))
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
        return state.reshape(1, 4)

    def reward_filter(self, nextState, done):
        cart_position = nextState[0][0]
        if done == False:
            reward = 0
        if cart_position > 0.5 or cart_position < -0.5:
            reward = -0.5
        if done == True:
            reward = -1
        return reward

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
        env = gym.make('CartPole-v0')
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
            actionHistory = []

            while done == False:
                if self.render:
                    env.render()
                action = self.get_epsilon_action(action_model, state, env)
                nextState, reward, done, info = env.step(action)

                nextState = self.state_filter(nextState)
                nextAction = self.get_epsilon_action(action_model, nextState, env)

                reward = self.reward_filter(nextState, done)

                expReplay.store_transition(state, action, reward, nextState, done)

                stepCount += 1
                totalSteps += 1
                state = nextState
                action = nextAction
                actionHistory.append(action)

                if stepCount > self.max_step_count_per_episode:
                    self.render = True
                    break

            inputs, targets = expReplay.random_mini_batch(model=action_model, batch_size=self.memory_batch_size)

            loss = learning_model.train_on_batch(inputs, targets)

            self.update_epsilon_decay()

            self.save_weights(learning_model)

            print(str(datetime.now()), " Episode: ", episode, " loss ", loss, " Done in ", stepCount, " epsilon  ", self.epsilon,  actionHistory)

            if self.swap_nn_weights_step_count < totalSteps:
                action_model.load_weights(self.WEIGHT_FILE)
                print("   Set action_model weights to learning_model at ", totalSteps, " steps")
                totalSteps = 0


if __name__ == "__main__":
    dqn = DQN(epsilon = 0.5, render=False, loadWeights=False)
    dqn.run()