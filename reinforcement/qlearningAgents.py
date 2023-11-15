# qlearningAgents.py
# ------------------
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
#

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # For the constructor a counter is initialised to store Q-values for state-action pairs.
        # util.Counter is a dict with a default value of 0 for missing keys. GIVEN IN ValueIterationAgents.
        self.output_q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # If the state and action pair exists.
        if (state, action) in self.output_q_values:
            return self.output_q_values[(state, action)]
        # If a state has never been seen 0.0 is returned.
        else:
            return 0.0
        # util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # The actions that are possible from that state are being captured in the possible_actions_space variable.
        possible_actions_space = self.getLegalActions(state)
        # If there are no legal actions from that state, which typically means that it is a
        # terminal state, 0.0 is returned
        if not possible_actions_space:
            return 0.0
        # For each state in the possible_actions_space, the Q value is calculated using the function which is
        # declared above, using the lambda function and these values are stored in a list.
        output_q_values = []
        for action in possible_actions_space:
            output_q_values.append(self.getQValue(state,action))
        '''output_q_values = list(map(lambda action: self.getQValue(state, action), possible_actions_space))'''
        # Out of all the captured Q values the maximum value is found out using inbuilt max funtion.
        max_output_q_value = max(output_q_values)
        # The max Q value for that state is returned.
        return max_output_q_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # The actions that are possible from that state are being captured in the possible_actions_space variable.
        possible_actions_space = self.getLegalActions(state)
        # If there are no legal actions from that state, which typically means that it is a
        # terminal state, None is returned
        if not possible_actions_space:
            return None
        # For each state in the possible_actions_space, the Q value is calculated using the function which is
        # declared above, using the lambda function and these values are stored in a list.
        output_q_values =[]
        '''output_q_values = list(map(lambda action: self.getQValue(state, action), possible_actions_space))'''
        for action in possible_actions_space:
            output_q_values.append(self.getQValue(state,action))
        # Out of all the captured Q values the maximum value is found out using inbuilt max funtion.
        max_output_q_value = max(output_q_values)
        # The ties in this case is handled using the random function which randomly chooses the action among all the
        # maximizing actions.
        max_q_value_index = []
        for index, each_q_value in enumerate(output_q_values):
            if each_q_value == max_output_q_value:
                max_q_value_index.append(index)
        # With the indexes of all occurrences of max q values being stored, the random function returns one index
        # randomly from the stored indexes.
        random_action_maximising = random.choice(max_q_value_index)
        # Randomly chosen maximizing action is returned.
        return possible_actions_space[random_action_maximising]
        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        # The actions that are possible from that state are being captured in the possible_actions_space variable.
        possible_actions_space = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # If there are no legal actions from that state, which typically means that it is a
        # terminal state, None is returned
        if not possible_actions_space:
            return action
        # Using util.flipCoin we calculate the probability of event success happening or epsilon.
        epsilon_success_probability = util.flipCoin(self.epsilon)
        # If there is epsilon_success_probability then randomly an action is chosen from possible actions space.
        # Else a best action policy is calculated using the function defined above for that state.
        '''
        ############################################################
        # if epsilon_success_probability:
        #     return random.choice(possible_actions_space)
        # else:
        #     return self.computeActionFromQValues(state)
        ############################################################
        '''
        action = epsilon_success_probability and random.choice(possible_actions_space) or self.computeActionFromQValues(state)
        # The best action in either of cases is returned.
        return action
        # util.raiseNotDefined()


    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # The Q value for the state is calculated using the formula.
        # The max Q value for the next state is calculated using the function defined above.
        next_state_max_q_value = self.computeValueFromQValues(nextState)
        # The max Q value of the next state is multiplied with discount factor.
        discount_factor_reward = next_state_max_q_value * self.discount
        # The discount factor reward is then multiplied with the learning rate factor.
        learning_factor_reward = self.alpha * (reward+discount_factor_reward)
        # The current state's Q value is calculated using the getQValue function defined above.
        current_state_q_value = self.getQValue(state, action)
        # The updated Q value is calculated using the current state's Q value multiplied by learning factor reward
        # multiplied by 1 minus alpha.
        new_output_q_value = (1 - self.alpha) * current_state_q_value + learning_factor_reward
        # The new updated value is stored with the states Q value dictionary.
        self.output_q_values[(state, action)] = new_output_q_value

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # The feature vectors associated with the state corresponding to the action has been extracted.
        features_vector_states = self.featExtractor.getFeatures(state, action)
        # The weights for the features vectors keys, basically the coordinates and the action have been stored.
        weights_features_keys = self.getWeights()
        output_q_values = []
        # For each feature vector in the state the keys of the feature vector dictionary are multiplied with the
        # weights associated with the keys to calculate the respective Q values.
        for vector in features_vector_states.keys():
            answer_q_value = weights_features_keys[vector]*features_vector_states[vector]
            output_q_values.append(answer_q_value)
        # The summation of the Q values has been returned as the new Q value.
        return sum(output_q_values)
        # util.raiseNotDefined()


    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # The feature vectors associated with the state corresponding to the action has been extracted.
        features_vector_states = self.featExtractor.getFeatures(state, action)
        # The weights for the features vectors keys, basically the coordinates and the action have been stored.
        weights_features_keys = self.getWeights()

        # The Q value for the state is calculated using the formula.
        # The max Q value for the next state is calculated using the function defined above.
        next_state_max_q_value = self.computeValueFromQValues(nextState)
        # The max Q value of the next state is multiplied with discount factor.
        discount_factor_reward = next_state_max_q_value * self.discount
        # The discount factor reward is then multiplied with the learning rate factor.
        learning_factor_reward = reward + discount_factor_reward
        # The current state's Q value is calculated using the getQValue function defined above.
        q_value_current_state = self.getQValue(state, action)
        # According to the formula the new Q value is calculated as below
        new_output_q_value = self.alpha * (learning_factor_reward-q_value_current_state)
        # For each feature vector key, the old vector key attribute is multiplied with new Q value and added to the
        # old weight of the attribute to provide the updated weight based on the transition action.
        for vector_attribute in features_vector_states.keys():
            weights_features_keys.update(
                {vector_attribute: weights_features_keys[vector_attribute] + new_output_q_value * features_vector_states[vector_attribute]})



        # util.raiseNotDefined()


    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            # features = self.featExtractor.getFeatures(state, action)
            "*** YOUR CODE HERE ***"
            pass
