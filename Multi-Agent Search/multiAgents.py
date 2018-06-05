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
import sys

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        max = sys.maxint
        score = 0
        distance = 0     
        def foodDist(position):
            queue = util.Queue()
            queue.push((position, 0))
            closed =set()
            closed.add((position))
            while not queue.isEmpty():
                pos, distance = queue.pop()
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    newx = pos[0] + dx
                    newy = pos[1] + dy
                    if(((newx, newy) not in closed) and(not successorGameState.hasWall(newx, newy))):
                        if successorGameState.hasFood(newx, newy):
                            return distance+1
                        closed.add((newx, newy))
                        queue.push(((newx, newy), distance+1))

        if successorGameState.isWin():
            return max
        """foodList = newFood.asList();
        foodDist=[]
        for food in foodList:
            foodDist =foodDist+[manhattanDistance(food,newPos)]
        if successorGameState.isWin():
            foodDist= [max]
        if currentGameState.getPacmanPosition() == newPos:
            return -max"""
        ghostPositions = successorGameState.getGhostPositions()
        ghostDistance=[]
        for ghost in ghostPositions:
            ghostDistance = ghostDistance+[manhattanDistance(newPos, ghost)]
        
        for ghost in ghostDistance:
            if ghost < 2:
               return -max  
             
        #minfoodDist =min(foodDist)
        score += successorGameState.getScore()-foodDist(newPos)*0.01;
        #return 1000/sum(foodDist) + 10000/len(foodDist)
        return score
        #return successorGameState.getScore()
        


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        #successor_state=gameState.generateSuccessor(agentIndex, action)
        """maxint = sys.maxint
        total_Agents = gameState.getNumAgents()

        def min_value(state, agentIndex, depth):
            
            bestscore = maxint
            actions = state.getLegalActions(agentIndex)
            
            next_states=[]
            for action in actions:
                next_states =next_states+[(state.generateSuccessor(agentIndex, action))]
            for next_state in next_states:
                score=value(next_state, (agentIndex+1)%total_Agents, depth+1)
                if(bestscore>score):
                    bestscore=score                  
            return bestscore
        
        def max_value(state, agentIndex, depth):
            
            bestscore = -maxint
            actions = state.getLegalActions(agentIndex)
            
            next_states=[]
            for action in actions:
                next_states =next_states+[(state.generateSuccessor(agentIndex, action))]
            for next_state in next_states:
                score=value(next_state, (agentIndex+1)%total_Agents, depth+1)
                if(bestscore<score):
                    bestscore=score 
                
            return bestscore
        
        def value(state, agentIndex, depth):
            if state.isWin():
                return self.evaluationFunction(state)
            if state.isLose():
                return self.evaluationFunction(state)
            if  depth > self.depth*total_Agents:        
                return self.evaluationFunction(state)
            if agentIndex>0:
                return min_value(state, agentIndex, depth)
            else:
                return max_value(state, agentIndex, depth)

        bestVal = -maxint
        bestAction = None
        actions = gameState.getLegalActions()
        
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            
            val = value(next_state,1,2)
            if val>bestVal:
                bestVal = val
                bestAction = action
        
        
        return bestAction"""
        def max_value(gameState, depth, agentIndex):
            agentIndex=0
            legal_actionList = gameState.getLegalActions(0)
            bestScore = -sys.maxint 
            bestAction = None
            if depth == self.depth:
                return (self.evaluationFunction(gameState),bestAction)

            

            if len(legal_actionList) == 0:
                return (self.evaluationFunction(gameState),bestAction)

            for action in legal_actionList:
                next_states= gameState.generateSuccessor(0, action)
                score = min_value( next_states, 1, depth)[0]
                if (score > bestScore):
                    bestScore= score
                    bestAction=action 
            return (bestScore, bestAction)



        def min_value(gameState, agentIndex, depth):
            legal_actionList= gameState.getLegalActions(agentIndex)
            bestScore = sys.maxint
            bestAction = None

            if len(legal_actionList) == 0:
                return (self.evaluationFunction(gameState),bestAction)

            for action in legal_actionList:
                next_states = gameState.generateSuccessor(agentIndex, action)
                if (agentIndex == gameState.getNumAgents() - 1):
                    score = max_value(next_states, depth + 1,0)[0]
                else:
                    score = min_value(next_states, agentIndex + 1, depth)[0]

                if (score < bestScore):
                    #bestScore=min(newScore,bestScore)
                    bestScore=score
                    bestAction=action 
            return (bestScore, bestAction)

        return max_value(gameState,0,0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        def max_value(gameState, depth, agentIndex, alpha, beta):
            agentIndex=0
            legal_actionList = gameState.getLegalActions(0)
            bestScore = -sys.maxint 
            bestAction = None
            if depth == self.depth:
                return (self.evaluationFunction(gameState),bestAction)       
                      

            if len(legal_actionList) == 0:
                return (self.evaluationFunction(gameState),bestAction)

            for action in legal_actionList:
               
                if (alpha > beta):
                    return (bestScore, bestAction)
                next_states = gameState.generateSuccessor(0, action)
                score = min_value(next_states, 1, depth, alpha, beta)[0]
                 #bestScore=min(newScore,bestScore)
                if (score > bestScore):
                    bestScore= score
                    bestAction=action 
                if (score > alpha):
                    alpha = score
            return (bestScore, bestAction)



        def min_value(gameState, agentIndex, depth, alpha, beta):
            legal_actionList= gameState.getLegalActions(agentIndex)
            bestScore = sys.maxint
            bestAction = None

            if len(legal_actionList) == 0:
                return (self.evaluationFunction(gameState),bestAction)

            for action in legal_actionList:
                
                if (alpha > beta):
                    return (bestScore, bestAction)
                next_states = gameState.generateSuccessor(agentIndex, action)
                
                if (agentIndex == gameState.getNumAgents() - 1):
                    score = max_value(next_states, depth + 1, 0,alpha, beta)[0]
                else:
                    score = min_value(next_states, agentIndex + 1, depth, alpha, beta)[0]

                if (score < bestScore):
                    bestScore= score
                    bestAction=action 
                if (score < beta):
                    beta = score
            return (bestScore, bestAction)

        return max_value(gameState, 0,0, -sys.maxint, sys.maxint)[1]



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def max_value(gameState,depth,agentIndex):
            agentIndex=0
            legal_actionList = gameState.getLegalActions(0)
            bestScore = -sys.maxint 
            bestAction = None
            if depth == self.depth:
                return (self.evaluationFunction(gameState),bestAction)

            

            if len(legal_actionList) == 0:
                return (self.evaluationFunction(gameState),bestAction)

            for action in legal_actionList:
                next_states= gameState.generateSuccessor(0, action)
                score = exp_value( next_states, 1, depth)[0]
                if (score > bestScore):
                    bestScore= score
                    bestAction=action 
            return (bestScore, bestAction)



        def exp_value(gameState,agentIndex, depth):
            legal_actionList = gameState.getLegalActions(agentIndex)
            finalScore = 0
            bestAction = None

            if len(legal_actionList) == 0:
                return (self.evaluationFunction(gameState),bestAction)

            for action in  legal_actionList :
                next_states = gameState.generateSuccessor(agentIndex, action)
                if (agentIndex == gameState.getNumAgents() - 1):
                    score = max_value(next_states,depth + 1,0)[0]
                else:
                    score = exp_value(next_states, agentIndex + 1, depth)[0]
                finalScore += score/len(legal_actionList)
            return (finalScore, bestAction)

        return max_value(gameState,0,0)[1]
        
        

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    Position = currentGameState.getPacmanPosition()
    
    GhostPosition = currentGameState.getGhostPositions()
    for ghost in GhostPosition:
        pos=(int(ghost[0]), int(ghost[1]))
        ghostDistance = [manhattanDistance(Position,pos)]
        safe = 0

    
    for dist in ghostDistance:
        if dist < 2:
            safe += 2
        elif dist < 3:
            safe += 1
        elif dist < 4:
            safe += 0.5
        elif dist < 5:
            safe += 0.1
        elif dist < 10:
            safe -= 0.03
        
    Food = currentGameState.getFood()
    foodList = Food.asList()
    foodDist=[]
    for food in foodList:
        foodDist = foodDist+[manhattanDistance(Position, food)]
    foodDist = sorted(foodDist)

    wallList = currentGameState.getWalls().asList()
    totalEmpty = 0
    for food in foodList:
        x,y = food
        emp = 0
        for pos in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            i , j = pos
            if Food[x + i][y + j] == False and (x + i, y + j) not in wallList:
               emp += 1
        totalEmpty +=emp 
    
    score = currentGameState.getScore()

    
    if len(foodDist) > 0:
        score += (min(ghostDistance) * (1.0/min(foodDist)**2) - totalEmpty * 6.5)

    return score+ safe
# Abbreviation
better = betterEvaluationFunction

