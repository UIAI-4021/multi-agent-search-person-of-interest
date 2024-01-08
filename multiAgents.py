import random
import util

from game import Agent
from game import Directions
from pacman import GameState
from util import manhattanDistance


def scoreEvaluationFunction(currentGameState: GameState):
    return currentGameState.getScore()


def getLegalActionsNoStop(index, gameState):
    possibleActions = gameState.getLegalActions(index)
    if Directions.STOP in possibleActions:
        possibleActions.remove(Directions.STOP)
    return possibleActions


def evaluationFunction(currentGameState):
    score = currentGameState.getScore()
    pacman = currentGameState.getPacmanPosition()
    ghost_state = currentGameState.getGhostStates()
    food = currentGameState.getFood()

    foods = food.asList()
    food_distances = [manhattanDistance(pacman, food) for food in foods]
    if food_distances:
        min_food_distances = min(food_distances)
        score += 1.0 / min_food_distances

    for ghostState in ghost_state:
        ghost = ghostState.getPosition()
        ghost_distance = manhattanDistance(pacman, ghost)
        if ghost_distance < 1:
            score -= 100

    return score


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    def alpha_beta(self, agent, depth, gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return evaluationFunction(gameState), None
        if agent == 0:
            return self.maxValue(gameState, depth, agent, alpha, beta)
        else:
            return self.minValue(gameState, depth, agent, alpha, beta)

    def maxValue(self, gameState, depth, agent, alpha, beta):
        possibleActions = getLegalActionsNoStop(agent, gameState)
        value = -float("inf")
        action_scores = []
        for action in possibleActions:
            action_score, new_action = self.alpha_beta(1, depth, gameState.generateSuccessor(agent, action), alpha,
                                                       beta)
            action_scores.append(action_score)
            value = max(value, action_score)
            alpha = max(alpha, value)
            if beta < alpha:
                break
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return value, possibleActions[chosenIndex]

    def minValue(self, gameState, depth, agent, alpha, beta):
        nextAgent = agent + 1
        if gameState.getNumAgents() == nextAgent:
            nextAgent = 0
        if nextAgent == 0:
            depth += 1
        possibleActions = getLegalActionsNoStop(agent, gameState)
        value = float("inf")
        action_scores = []
        for action in possibleActions:
            action_score, new_action = self.alpha_beta(nextAgent, depth, gameState.generateSuccessor(agent, action),
                                                       alpha, beta)
            action_scores.append(action_score)

            value = min(value, action_score)
            beta = min(beta, value)
            if beta < alpha:
                break
        return value, None

    def getAction(self, gameState: GameState):
        alpha = -float("inf")
        beta = float("inf")
        action_score, action = self.alpha_beta(0, 0, gameState, alpha, beta)
        return action
