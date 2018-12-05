import numpy as np
from gridworld import GridWorld
import matplotlib.pyplot as plt
import pdb

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
    #pdb.set_trace()
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
        #pdb.set_trace()
        observation = visit[0]
        col = observation[1] + (observation[0]*4)
        if(policy_matrix[observation[0], observation[1]] != -1):
            policy_matrix[observation[0], observation[1]] = \
                np.argmax(state_action_matrix[:,col])
    return policy_matrix

from cvxopt import matrix, solvers

def irl(n_states, n_actions, transition_probability, policy, discount, Rmax,
        l1):
    """
    Find a reward function with inverse RL as described in Ng & Russell, 2000.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    policy: Vector mapping state ints to action ints. Shape (N,).
    discount: Discount factor. float.
    Rmax: Maximum reward. float.
    l1: l1 regularisation. float.
    -> Reward vector
    """

    A = set(range(n_actions))  # Set of actions to help manage reordering
                               # actions.
    # The transition policy convention is different here to the rest of the code
    # for legacy reasons; here, we reorder axes to fix this. We expect the
    # new probabilities to be of the shape (A, N, N).
    transition_probability = np.transpose(transition_probability, (1, 0, 2))

    def T(a, s):
        """
        Shorthand for a dot product used a lot in the LP formulation.
        """

        return np.dot(transition_probability[policy[s], s] -
                      transition_probability[a, s],
                      np.linalg.inv(np.eye(n_states) -
                        discount*transition_probability[policy[s]]))

    # This entire function just computes the block matrices used for the LP
    # formulation of IRL.

    # Minimise c . x.
    c = -np.hstack([np.zeros(n_states), np.ones(n_states),
                    -l1*np.ones(n_states)])
    zero_stack1 = np.zeros((n_states*(n_actions-1), n_states))
    T_stack = np.vstack([
        -T(a, s)
        for s in range(n_states)
        for a in A - {policy[s]}
    ])
    I_stack1 = np.vstack([
        np.eye(1, n_states, s)
        for s in range(n_states)
        for a in A - {policy[s]}
    ])
    I_stack2 = np.eye(n_states)
    zero_stack2 = np.zeros((n_states, n_states))

    D_left = np.vstack([T_stack, T_stack, -I_stack2, I_stack2])
    D_middle = np.vstack([I_stack1, zero_stack1, zero_stack2, zero_stack2])
    D_right = np.vstack([zero_stack1, zero_stack1, -I_stack2, -I_stack2])

    D = np.hstack([D_left, D_middle, D_right])
    b = np.zeros((n_states*(n_actions-1)*2 + 2*n_states, 1))
    bounds = np.array([(None, None)]*2*n_states + [(-Rmax, Rmax)]*n_states)

    # We still need to bound R. To do this, we just add
    # -I R <= Rmax 1
    # I R <= Rmax 1
    # So to D we need to add -I and I, and to b we need to add Rmax 1 and Rmax 1
    D_bounds = np.hstack([
        np.vstack([
            -np.eye(n_states),
            np.eye(n_states)]),
        np.vstack([
            np.zeros((n_states, n_states)),
            np.zeros((n_states, n_states))]),
        np.vstack([
            np.zeros((n_states, n_states)),
            np.zeros((n_states, n_states))])])
    b_bounds = np.vstack([Rmax*np.ones((n_states, 1))]*2)
    D = np.vstack((D, D_bounds))
    b = np.vstack((b, b_bounds))
    A_ub = matrix(D)
    b = matrix(b)
    c = matrix(c)
    results = solvers.lp(c, A_ub, b)
    r = np.asarray(results["x"][:n_states], dtype=np.double)

    return r.reshape((n_states,))


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
    gamma = 0.5
    tot_epoch = 30000
    print_epoch = 3000

    for epoch in range(tot_epoch):
        #Starting a new episode
        episode_list = list()
        #Reset and return the first observation and reward
        observation = env.reset(exploring_starts=False)
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

        #pdb.set_trace()
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
    print(policy_matrix)
    
    state_value_matrix = state_action_matrix.max(axis=0)
    print(state_value_matrix)


    policy_matrix[policy_matrix==-1] = 0
    final_policy_list = policy_matrix.reshape(-1).astype(int)
    print(final_policy_list)


    # ## Random State Transition Matrix

    # In[13]:


    random_state_transition_matrix = np.random.rand(25,4,25)
    random_state_transition_matrix = random_state_transition_matrix/random_state_transition_matrix.sum(axis=1)[:,None]
    print(random_state_transition_matrix.shape)


    # ## With a handcrafted State Transition Matrix

    fixed_state_transition_matrix = np.load("gw_transition_probability.npy")
    print(fixed_state_transition_matrix.shape)


    # In[20]:


    r_random = irl(n_states=25,n_actions=4,transition_probability=random_state_transition_matrix,    policy=final_policy_list,discount=0.2,Rmax=1,l1=5)


    # In[24]:

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    plt.subplot(3,2,1)
    plt.pcolor(np.flip(reward_matrix,0))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(3,2,2)
    plt.pcolor(r_random.reshape((5, 5)))
    plt.colorbar()
    plt.title("Recovered reward (RANDOM)")
    #plt.show()


    r_fixed = irl(n_states=25,n_actions=4,transition_probability=fixed_state_transition_matrix,    policy=final_policy_list,discount=0.5,Rmax=10,l1=5)


    plt.subplot(3,2,3)
    plt.pcolor(np.flip(reward_matrix,0))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(3,2,4)
    plt.pcolor(r_fixed.reshape((5, 5)))
    plt.colorbar()
    plt.title("Recovered reward (FIXED)")
    #plt.show()
    
    import irl.linear_irl as linear_irl
    import irl.mdp.gridworld as gridworld

    grid_size = 5
    discount = 0.2
    wind = 0.3
    trajectory_length = 3*grid_size

    gw = gridworld.Gridworld(grid_size, wind, discount)
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])

    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])

    policy = [gw.optimal_policy_deterministic(s) for s in range(gw.n_states)]
    print(policy)
    #final_policy_list = list(final_policy.reshape(-1).astype(int))
    #r = linear_irl.irl(gw.n_states, gw.n_actions, gw.transition_probability,policy, gw.discount, 1, 5)
    r = linear_irl.irl(gw.n_states, gw.n_actions, gw.transition_probability,policy, gw.discount, 1, 5)
    print(r.shape)
    print(gw.optimal_policy_deterministic)
    print(np.array(policy).reshape(5,5))

    plt.subplot(3,2,5)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(3,2,6)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()


if __name__ == "__main__":
    main()
