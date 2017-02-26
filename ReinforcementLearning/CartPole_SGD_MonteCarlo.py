import gym
import numpy as np
import math
import random


def q_function(states, actions, weights):
    '''
    Gets a Q Value for the state/action/weights pair
    :param states:
    :param actions:
    :param weights:
    :return:
    '''
    featureVector = get_feature_vector(states, actions)
    qValue = np.matmul(weights.reshape((1,10)), featureVector)
    return qValue

def get_feature_vector(states, actions):
    '''
    Converts the state and action vectors to a 1 dimensional feature vector with 10 values
    :param states: 4 values in 1D vector
    :param actions: 2 values in 1D vector
    :return: a 1 dimensional feature vector with 10 values
    '''
    states = states.reshape(1,5)
    states2x = np.concatenate((states,states))
    actions = actions.reshape(2,1)
    featureVector = np.multiply(states2x,actions)
    featureVector1D = featureVector.reshape(10)
    return featureVector1D

def np_state(gym_states):
    return np.append(np.array(gym_states), 1)

def get_epsilon_action(weights, currentState, epsilon, env):
    if  random.random() < epsilon:
        return get_vactor_action(env.action_space.sample())
    else:
        return get_max_action(weights, currentState)

def get_max_action(weights, currentState):
    left_action = np.array([1,0])
    right_action = np.array([0, 1])
    q_values_left = q_function(currentState, left_action, weights)
    q_values_right = q_function(currentState, right_action, weights)
    if q_values_left > q_values_right:
        return left_action
    else:
        return right_action

def get_vactor_action(action):
    if action == 0:
        return np.array([0,1])
    else:
        return np.array([1,0])

def run():
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    epsilon = 0.1
    alpha = 0.01
    weights = np.random.rand(2,5)

    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    env = gym.make('CartPole-v0')
    print('Action space:', env.action_space)
    print('Observation space:', env.observation_space)

    memory = []

    for episode in range(2000):
        stepCount = 0
        done = False
        state = np_state(env.reset())
        while True:
            action = get_epsilon_action(weights, state, epsilon, env)
            scalarAction = np.argmax(action)
            nextState, reward, done, _ = env.step(scalarAction)
            reward = math.cos(nextState[1])
            memory.append({"state": state, "action": action, "reward": reward} )
            state = np_state(nextState)
            stepCount = stepCount + 1

            if episode > 200:
                epsilon = 0.00
                alpha = 0.001
                env.render()
                print(action)

            if done == True:
                reward = -10
                endAction = np.array([0, 0])
                memory.append({"state": state, "action": endAction, "reward": reward} )

                if episode > 200:
                    print("Episode: ", episode, " Done in ", stepCount)
                    print(np.round(weights,6))
                break

        for mem in memory:
            state = mem["state"]
            action = mem["action"]
            reward = mem["reward"]

            learner = q_function(state, action, weights)
            gradient =  get_feature_vector(state, action)
            weights_delta = alpha * (reward - learner) * gradient
            weights_delta2d = weights_delta.reshape(2, 5)
            weights = weights + weights_delta2d


if __name__ == "__main__":
    run()