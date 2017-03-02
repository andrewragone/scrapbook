import gym
import numpy as np
import math
import random


def q_function(states, actions, weights):
    '''
    Returns a 2d vector.  The argmax is the action
    '''
    featureVector = get_feature_vector(states, actions)
    qValue = np.matmul(weights.reshape((1,10)), featureVector)
    return qValue

def get_feature_vector(states, actions):
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
    gamma = 0.5
    alpha = 0.01
    tdLambda = 0.5
    weights = np.random.rand(2,5)


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    env = gym.make('CartPole-v0')
    print('Action space:', env.action_space)
    print('Observation space:', env.observation_space)

    for episode in range(2000):
        currentState = np_state(env.reset())
        currentAction = get_epsilon_action(weights, currentState, epsilon, env)
        eligibilityTrace = np.zeros(10)

        t = 0
        T = 0
        while True:
            a = np.argmax(currentAction)
            nextState, reward, done, _ = env.step(a)
            reward = math.cos(nextState[1])
            nextState = np_state(nextState)
            nextAction = get_epsilon_action(weights, nextState, epsilon, env)

            #Calculate eligibility Trace
            gradient = get_feature_vector(currentState, currentAction)
            eligibilityTrace = gamma * tdLambda * eligibilityTrace  + gradient

            # Calculate Delta
            target = reward + gamma * q_function(nextState, nextAction, weights)
            learner = q_function(currentState, currentAction, weights)
            deltaT = target - learner

            #Update Weights
            weights_delta =  alpha * deltaT * eligibilityTrace
            weights_delta2d = weights_delta.reshape(2, 5)
            weights = weights + weights_delta2d


            #Next State
            currentState = np.copy(nextState)
            currentAction = np.copy(nextAction)
            t += 1

            if episode>1800:
                epsilon = 0.00
                alpha = 0.001
                env.render()

            if done == True:
                if episode > 200:
                    print("Episode: ", episode, " Done in ", t)
                    print(np.round(weights,6))
                break




if __name__ == "__main__":
    run()