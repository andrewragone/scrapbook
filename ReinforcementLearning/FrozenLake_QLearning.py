import gym
import numpy as np
import random

def get_epsilon_action(env, Q, observation, epsilon):
    if  random.random() < epsilon:
        return env.action_space.sample()
    else:
        return get_max_action(Q, observation)

def get_max_action(Q, observation):
    actions = Q[observation,:]
    action = np.argmax(actions)
    return action


np.set_printoptions(precision=4)
env = gym.make('FrozenLake-v0')

#Initialize table with all zeros
Q = np.random.rand(env.observation_space.n,env.action_space.n)
#Q = np.zeros([env.observation_space.n,env.action_space.n])
lastQ = np.copy(Q)
# Set learning parameters
lr = 0.1
y = 0.99
epsilon = 0.1
num_episodes = 1000000
#create lists to contain total rewards and steps per episode

rList = []
for episode in np.arange(0, 100000, 1):
    #Reset environment and get first new observation
    observation = env.reset()

    rAll = 0
    isDone = False
    stepCount = 0
    #Choose an action by greedily (with noise) picking from Q table
    action = get_epsilon_action(env, Q, observation, epsilon)
    #The Q-Table learning algorithm
    while True:
        stepCount+=1

        #Get new state and reward from environment
        observationPrime, reward, isDone, info = env.step(action)
        #print ("ACTION:  ", action, "\tSTATE: ", observationPrime)
        #env.render()
        if isDone == True and reward == 0:
            reward = -1
        # Update Q-Table with new knowledge
        actionPrime = get_epsilon_action(env, Q, observation, epsilon )
        actionMax = get_max_action(Q, observation)
        Q[observation, action] = Q[observation, action] + lr * (reward + y * Q[observationPrime, actionMax] - Q[observation, action])
        observation = observationPrime
        action = actionPrime

        if isDone == True:
            if episode%1000 == 0:

                print("episode", episode, "\tstepCount ", stepCount, "\tobservation ", observation, "\treward ", reward, "\tepsilon ", epsilon)
                #print(np.round(lastQ - Q,3))
                #print(np.round(Q, 3))
                print(np.argmax(Q, axis=1))
                lastEpisode = episode
                lastQ = np.copy(Q)
            if episode == 50000:
                lr = 0.01
            break

    rList.append(rAll)

print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)