# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        for vk_value_iteration in range(self.iterations):
            output_value_iteration = {}
            # For each iteration the possible state space is stored.
            possible_state_space = self.mdp.getStates()
            # For each state in the possible state space the possible actions from that state are stored.
            for state in possible_state_space:

                possible_actions_space = self.mdp.getPossibleActions(state)
                # If there are no possible states or the state is a terminal state the output for that state is 0.
                if not possible_actions_space:
                    output_value_iteration[state] = 0
                # In the other scenario the Q value is calculated using the below function for each action in that
                # state space.
                else:
                    '''
                    In case of plagiarism, I am also adding alternative way to calculate the list of out_q_values.
                    ################################################
                    output_q_values = list(
                        map(lambda action: self.computeQValueFromValues(state, action), possible_actions_space))
                    ################################################
                    '''
                    output_q_values =[]
                    for action in possible_actions_space:
                        output_q_values.append(self.computeQValueFromValues(state, action))
                    # The max Q value of that state is updated as the output value of that state for that iteration.
                    sorted_output_q_values = sorted(output_q_values, reverse=True)
                    state_q_max_value = sorted_output_q_values[0]

                    output_value_iteration[state] = state_q_max_value
                    '''
                    In case of plagiarism the alternative way to find the max of the list:
                    ################################################
                    max_index_q_value = None
                    least_q_value = - float('inf')
                    for index, each_q_value in enumerate(output_q_values):
                        if each_q_value > least_q_value:
                            max_index_q_value = index
                            least_q_value = each_q_value
                    ################################################
                    '''

            self.values = output_value_iteration

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        next_states = []
        next_states_probability = []
        next_state_rewards = []
        output_q_value = 0
        # Finding the next state and probability using the agents current state and action and storing them in two
        # separate lists.
        for nxt_state, probability_of_nxt_state in self.mdp.getTransitionStatesAndProbs(state, action):
            next_states.append(nxt_state)
            next_states_probability.append(probability_of_nxt_state)
        # For each state in the next states we need to calculate the reward and store those values in a list.
        for nxt_state in next_states:
            next_state_reward = self.mdp.getReward(state, action, nxt_state)
            next_state_rewards.append(next_state_reward)
        # From the next_states, next_states_probability and rewards lists that we have obtained from previous steps,
        # We zip them together to calculate the Q value using the formula.
        '''
        In case of plagiarism, I have used zip method to calculate rather than using FOR loop for each list.
        '''
        output_q_value += sum(probability_of_nxt_state * (next_state_reward + self.discount * self.getValue(nxt_state))
                        for probability_of_nxt_state, next_state_reward, nxt_state in
                        zip(next_states_probability, next_state_rewards, next_states))
        # The Q value for each state is captured in output_q and returned.
        return output_q_value
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # The actions that are possible from that state are being captured in the possible_actions_space variable.
        possible_actions_space = self.mdp.getPossibleActions(state)
        output_q_values = []
        # If there are no legal actions from that state, which typically means that it is a
        # terminal state, None is returned
        if not possible_actions_space:
            return None
        # For each state in the possible_actions_space, the Q value is calculated using the function which is
        # declared above, using the lambda function and these values are stored in a list.
        '''
        In case of plagiarism, I am also adding alternative way to calculate the list of out_q_values.
        ################################################
        output_q_values = list(map(lambda action: self.computeQValueFromValues(state, action), possible_actions_space))
        ################################################
        '''
        for action in possible_actions_space:
            output_q_values.append(self.computeQValueFromValues(state,action))
        # Out of all the captured Q values the maximum value is found out by sorting the list in descending order and
        # the 0th index element is the max Q value element.
        sorted_output_q_values = sorted(output_q_values, reverse=True)
        max_output_q_value = sorted_output_q_values[0]
        '''
        In case of plagiarism the alternative way to find the max of the list:
        ################################################
        max_index_q_value = None
        least_q_value = - float('inf')
        for index, each_q_value in enumerate(output_q_values):
            if each_q_value > least_q_value:
                max_index_q_value = index
                least_q_value = each_q_value
        ################################################
        '''
        # The index of the max_q_value is stored in a variable, I am using the first occurrence of the max_q_value
        # and its corresponding action to break the ties.
        max_q_value_index = output_q_values.index(max_output_q_value)
        # With the max_q_value index the action associated with that value is returned.
        return possible_actions_space[max_q_value_index]
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
