# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()

def getLegalActionsNoStop(index, gameState):
        possibleActions = gameState.getLegalActions(index)
        if Directions.STOP in possibleActions:
            possibleActions.remove(Directions.STOP)
        return possibleActions


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)
        self.temperature = 1.0


class AIAgent(MultiAgentSearchAgent):
    def alphabeta(self, agent, depth, gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState), None
        possibleActions = getLegalActionsNoStop(agent, gameState)
        if agent == 0:  # maximize for pacman
            value = -999999
            action_scores = []
            for action in getLegalActionsNoStop(agent, gameState):
                action_score, new_action = self.alphabeta(1, depth, gameState.generateSuccessor(agent, action), alpha,
                                                          beta)
                action_scores.append(action_score)
                value = max(value, action_score)
                alpha = max(alpha, value)
                if beta <= alpha:  # alpha-beta pruning
                    break
            max_action = max(action_scores)
            max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
            chosenIndex = random.choice(max_indices)
            return value, possibleActions[chosenIndex]
        else:  # minimize for ghosts
            nextAgent = agent + 1  # get the next agent
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:  # increase depth every time all agents have moved
                depth += 1
            possibleActions = getLegalActionsNoStop(agent, gameState)
            value = 999999
            action_scores = []
            for action in possibleActions:
                action_score, new_action = self.alphabeta(nextAgent, depth, gameState.generateSuccessor(agent, action), alpha, beta)
                action_scores.append(action_score)
                value = min(value, action_score)
                beta = min(beta, value)
                if beta <= alpha:  # alpha-beta pruning
                    break
            min_action = min(action_scores)
            min_indices = [index for index in range(len(action_scores)) if action_scores[index] == min_action]
            chosenIndex = random.choice(min_indices)
            return value, possibleActions[chosenIndex]

    def getAction(self, gameState: GameState):
        """
        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # TODO: Your code goes here
        # util.raiseNotDefined()

        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction using alpha-beta pruning.
        """
        possibleActions = getLegalActionsNoStop(0, gameState)
        alpha = -999999
        beta = 999999
        #action_scores = [self.alphabeta(0, 0, gameState.generateSuccessor(0, action), alpha, beta) for action
        #                 in possibleActions]
        action_score, action = self.alphabeta(0, 0, gameState, alpha, beta)
        return action