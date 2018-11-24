import numpy as np
from gridworld import GridWorld

def print_policy(policy_matrix):
    '''Print the policy using specific symbol.

    * terminal state
    ^ > v < up, right, down, left
    # obstacle
    '''
    counter = 0
    shape = policy_matrix.shape
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(policy_matrix[row,col] == -1): policy_string += " *  "
            elif(policy_matrix[row,col] == 0): policy_string += " ^  "
            elif(policy_matrix[row,col] == 1): policy_string += " >  "
            elif(policy_matrix[row,col] == 2): policy_string += " v  "
            elif(policy_matrix[row,col] == 3): policy_string += " <  "
            elif(np.isnan(policy_matrix[row,col])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)

def get_return(state_list, gamma):
    '''Get the return for a list of action-state values.

    @return get the Return
    '''
    counter = 0
    return_value = 0
    for visit in state_list:
        reward = visit[2]
        return_value += reward * np.power(gamma, counter)
        counter += 1
    return return_value

def update_policy(episode_list, policy_matrix, state_action_matrix):
    '''Update a policy making it greedy in respect of the state-action matrix.

    @return the updated policy
    '''
    for visit in episode_list:
        observation = visit[0]
        col = observation[1] + (observation[0]*4)
        if(policy_matrix[observation[0], observation[1]] != -1):
            policy_matrix[observation[0], observation[1]] = \
                np.argmax(state_action_matrix[:,col])
    return policy_matrix



def main():

    env = GridWorld(5, 5)

    #Define the state matrix
    state_matrix = np.zeros((5,5))
    state_matrix[0, 4] = 1
    print("State Matrix:")
    print(state_matrix)

    #Define the reward matrix
    reward_matrix = np.full((5,5), 0)
    reward_matrix[0, 4] = 1
    print("Reward Matrix:")
    print(reward_matrix)

    #Define the transition matrix
    transition_matrix = np.array([[0.7, 0.1, 0.1, 0.1],
                                  [0.1, 0.7, 0.1, 0.1],
                                  [0.1, 0.1, 0.7, 0.1],
                                  [0.1, 0.1, 0.1, 0.7]])

    #Random policy
    policy_matrix = np.random.randint(low=0, high=4, size=(5, 5)).astype(np.float32)
    policy_matrix[0,4] = -1

    #Set the matrices in the world
    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    state_action_matrix = np.random.random_sample((4,25)) # Q
    #init with 1.0e-10 to avoid division by zero
    running_mean_matrix = np.full((4,25), 1.0e-10)
    gamma = 0.999
    tot_epoch = 500000
    print_epoch = 3000

    for epoch in range(tot_epoch):
        #Starting a new episode
        episode_list = list()
        #Reset and return the first observation and reward
        observation = env.reset(exploring_starts=True)
        #action = np.random.choice(4, 1)
        #action = policy_matrix[observation[0], observation[1]]
        #episode_list.append((observation, action, reward))
        is_starting = True
        for _ in range(1000):
            #Take the action from the action matrix
            action = policy_matrix[observation[0], observation[1]]
            #If the episode just started then it is
                #necessary to choose a random action (exploring starts)
            if(is_starting):
                action = np.random.randint(0, 4)
                is_starting = False
            #Move one step in the environment and get obs and reward
            new_observation, reward, done = env.step(action)
            #Append the visit in the episode list
            episode_list.append((observation, action, reward))
            observation = new_observation
            if done: break
        #The episode is finished, now estimating the utilities
        counter = 0
        #Checkup to identify if it is the first visit to a state
        checkup_matrix = np.zeros((4,25))
        #This cycle is the implementation of First-Visit MC.
        #For each state stored in the episode list check if it
        #is the rist visit and then estimate the return.
        for visit in episode_list:
            observation = visit[0]
            action = visit[1]
            col = int(observation[1] + (observation[0]*4))
            row = int(action)
            if(checkup_matrix[row, col] == 0):
                return_value = get_return(episode_list[counter:], gamma)
                running_mean_matrix[row, col] += 1
                state_action_matrix[row, col] += return_value
                checkup_matrix[row, col] = 1
            counter += 1
        #Policy Update
        policy_matrix = update_policy(episode_list,
                                      policy_matrix,
                                      state_action_matrix/running_mean_matrix)
        #Printing
        if(epoch % print_epoch == 0):
            print("")
            print("State-Action matrix after " + str(epoch+1) + " iterations:")
            print(state_action_matrix / running_mean_matrix)
            print("Policy matrix after " + str(epoch+1) + " iterations:")
            print(policy_matrix)
            print_policy(policy_matrix)
    #Time to check the utility matrix obtained
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(state_action_matrix / running_mean_matrix)


if __name__ == "__main__":
    main()
