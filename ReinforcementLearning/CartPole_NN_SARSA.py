import gym
import numpy as np
import math
import random
import os

from datetime import  datetime
from os.path import isfile
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import sgd, adam
from keras.layers.core import Dense, Dropout, Activation, Flatten


class ExperienceReplay():
    def __init__(self, max_memory=1000):
        self.memory = []
        self.max_memory = max_memory

    def store_transition(self, state, action, reward, nextState, nextAction, done):
        self.memory.append({"state": state, "action": action, "reward": reward, "nextState": nextState, "nextAction": nextAction, "done": done})
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def clear_memory(self):
        self.memory.clear()

    def random_mini_batch(self, model, batch_size = 50, gamma = 0.99):
        len_memory = len(self.memory)
        batch_size = min(len_memory, batch_size)
        # print ("len_memory", len_memory, "batch_size", batch_size)
        states = []
        targets = []
        for i, idx in enumerate(np.random.randint(0, len_memory, batch_size)):
            m = self.memory[idx]
            state = m["state"]
            action = m["action"]
            reward = m["reward"]
            nextState = m["nextState"]
            nextAction = m["nextAction"]
            isDone = m["done"]

            target = model.predict(state)
            if isDone == True:
                target[0, action] = reward
            else:
                QvaluesForAllActions = model.predict(nextState)
                Qvalue = QvaluesForAllActions[0][nextAction]
                target[0, action] = reward + gamma * Qvalue

            states.append(state[0])
            targets.append(target[0])

        states = np.array(states)
        targets = np.array(targets)
        return states, targets

class DQN():
    def __init__(self, epsilon=0.1, render = False, loadWeights=True):
        self.epsilon = epsilon
        self.render = render
        self.loadWeights = loadWeights
        self.WEIGHT_FILE = "CartPole_NN_Weights.h5"


    def create_nn_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=4))
        model.add(Activation('softmax'))
        model.add(Dense(2))
        model.compile(loss='mse', optimizer=adam(lr=0.1))
        return model

    def get_epsilon_action(self, model, currentState, env):
        if  random.random() < self.epsilon:
            action = env.action_space.sample()
            #print("random action: ", action )
            return action
        else:
            actionQvalues = model.predict(currentState)
            action = np.argmax(actionQvalues[0])
            #print("network action: ", action)
            return action

    def state_filter(self, state):
        return state.reshape(1, 4)

    def run(self):
        env = gym.make('CartPole-v0')
        print('Action space:', env.action_space)
        print('Observation space:', env.observation_space)
        print("cart_position, pole_angle, cart_velocity, angle_rate_of_change")
        action_model = self.create_nn_model()
        learning_model = self.create_nn_model()
        if self.loadWeights:
            action_model.load_weights(self.WEIGHT_FILE)
            learning_model.load_weights(self.WEIGHT_FILE)

        expReplay = ExperienceReplay(max_memory=1000)
        totalSteps = 0

        for episode in range(2000):
            stepCount = 0
            loss = 0
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

                #reward = math.cos(nextState[1])
                if done == False:
                    reward = 0
                if done == True:
                    reward = -1

                expReplay.store_transition(state, action, reward, nextState, nextAction, done)

                stepCount += 1
                totalSteps += 1
                state = nextState
                action = nextAction

                actionHistory.append(action)

            inputs, targets = expReplay.random_mini_batch(action_model, batch_size=50)

            loss = learning_model.train_on_batch(inputs, targets)

            if self.epsilon >= 0.001:
                self.epsilon -= (1 / (1000))

            if isfile(self.WEIGHT_FILE):
                os.remove(self.WEIGHT_FILE)
            learning_model.save_weights(self.WEIGHT_FILE)

            print(str(datetime.now()), " Episode: ", episode," loss ", loss, " Done in ", stepCount, " epsilon  ", self.epsilon,  actionHistory)

            if totalSteps > 100:
                action_model.load_weights(self.WEIGHT_FILE)
                print("   Set action_model weights to learning_model at ", totalSteps, " steps")
                totalSteps = 0


if __name__ == "__main__":
    dqn = DQN(epsilon = 0.5, render=False, loadWeights=False)
    dqn.run()