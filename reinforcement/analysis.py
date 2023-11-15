# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
    """
    answerDiscount = 0.5  # A moderate value here makes the agent prefer the close exit for immediate rewards.
    answerNoise = 0.1  # A lower value for noise ensures deterministic behavior of the agent when moving towards cliff.
    answerLivingReward = -3.0  # A negative value lets the agent to end the game because of negative rewards.
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question2b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
    """
    answerDiscount = 0.5  # A moderate value here makes the agent prefer the close exit for immediate rewards.
    answerNoise = 0.1  # A lower value for noise ensures deterministic behavior of the agent when moving towards cliff.
    answerLivingReward = -1.0  # A lesser negative value here compared to previous case is used as the agent will
    # find an exit to end that game because of negative rewards and also plans for safer path.
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question2c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
    """
    answerDiscount = 0.8  # A high value here makes the agent prefer the distant exit focusing on the future rewards.
    answerNoise = 0.2  # A lower value here minimizes the risk of moves which can make the agent fall of the cliff.
    answerLivingReward = -1.0  # A lesser negative value here is used as the agent will find an exit to end that game
    # because of negative rewards and also plans for safer path without falling of the cliff.
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question2d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    answerDiscount = 0.8  # A high value here makes the agent prefer the distant exit focusing on the future rewards.
    answerNoise = 0.2  # A lower value here minimizes the risk of moves which can make the agent fall of the cliff.
    answerLivingReward = 1  # A positive value here compared to the previous case is used to let the agent get to the
    # exit allowing to take the longest path.
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question2e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
    """
    answerDiscount = 0  # A value of 0 here makes the exits and cliffs less appealing for the agent.
    answerNoise = 0.3  # A positive value for noise in the scenario create randomness to any direct path.
    answerLivingReward = 0.1  # The low reward value helps the agent to move to another states avoiding terminal state.
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis

    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
