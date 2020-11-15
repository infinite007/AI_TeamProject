#####################
# CS571 Fall 2020   #
# Group 10 Project  #
#####################

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
        newGhostPositions = [g.getPosition() for g in newGhostStates]

        score = 0 #Term to add to succ state score
        #Stay away from ghosts
        ghost_dists = [util.manhattanDistance(newPos,gPos) for gPos in newGhostPositions]
        if min(ghost_dists) < 2:
            score -= 1000
        #Get the food
        score -= 500*newFood.count()
        #Get closer to food
        if newFood.count() > 0:
            newFoodPos = newFood.asList()
            min_food_dist = min([util.manhattanDistance(newPos,fPos) for fPos in newFoodPos])
            score -= min_food_dist
        return successorGameState.getScore() + score

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
        #Implementation is based on pseudocode from lecture slides

        #Returns a value, action tuple given state, agent to act, and depth in search tree
        def value(state, agent_index, depth):
            if depth == self.depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None
            #Ply ends after all min layers
            if agent_index == state.getNumAgents()-1:
                depth +=1
            if agent_index == 0:
                return max_value(state, agent_index, depth)
            else:
                return min_value(state, agent_index, depth)

        #For max layers
        def max_value(state, agent_index, depth):
            v = -1e9
            actions = state.getLegalActions(agent_index)
            next_states = [state.generateSuccessor(agent_index, a) for a in actions]
            next_agent_index = (agent_index + 1)%state.getNumAgents()
            values = [value(s, next_agent_index, depth)[0] for s in next_states]
            action = None
            for i in range(len(values)):
                if values[i] > v:
                    v = values[i]
                    action = actions[i]
            return v, action

        #For min layers
        def min_value(state, agent_index, depth):
            v = 1e9
            actions = state.getLegalActions(agent_index)
            next_states = [state.generateSuccessor(agent_index, a) for a in actions]
            next_agent_index = (agent_index + 1)%state.getNumAgents()
            values = [value(s, next_agent_index, depth)[0] for s in next_states]
            action = None
            for i in range(len(values)):
                if values[i] < v:
                    v = min(v, values[i])
                    action = actions[i]
            return v, action

        depth = 0
        v, action = value(gameState, 0, depth)
        return action
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #Implementation based on pseudocode provided in Project 2 Question 3
        #Returns a value, action tuple given a state, agent to act, depth in search tree, alpha and beta
        def value(state, agent_index, depth, alpha, beta):
            if depth == self.depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None
            #Ply ends with last agent taking a turn
            if agent_index == state.getNumAgents()-1:
                depth +=1
            if agent_index == 0:
                return max_value(state, agent_index, depth, alpha, beta)
            else:
                return min_value(state, agent_index, depth, alpha, beta)

        #For max layers
        def max_value(state, agent_index, depth, alpha, beta):
            v = -1e9
            actions = state.getLegalActions(agent_index)
            next_agent_index = (agent_index + 1)%state.getNumAgents()
            action = None
            for i in range(len(actions)):
                next_state = state.generateSuccessor(agent_index, actions[i])
                value_res = value(next_state, next_agent_index, depth, alpha, beta)[0]
                if value_res > v:
                    v = value_res
                    action = actions[i]
                if v > beta:
                    return v, action
                alpha = max(alpha, v)
            return v, action

        #For min layers
        def min_value(state, agent_index, depth, alpha, beta):
            v = 1e9
            actions = state.getLegalActions(agent_index)
            next_agent_index = (agent_index + 1)%state.getNumAgents()
            action = None
            for i in range(len(actions)):
                next_state = state.generateSuccessor(agent_index, actions[i])
                value_res = value(next_state, next_agent_index, depth, alpha, beta)[0]
                if value_res < v:
                    v = value_res
                    action = actions[i]
                if v < alpha:
                    return v, action
                beta = min(beta, v)

            return v, action

        depth = 0
        alpha = -1e9
        beta = 1e9
        v, action = value(gameState, 0, depth, alpha, beta)
        return action

        util.raiseNotDefined()

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
        #Same as minimax agent, except the min_value is replaced with the expected value of actions given uniform random sampling
        def value(state, agent_index, depth):
            if depth == self.depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None
            if agent_index == state.getNumAgents()-1:
                depth +=1
            if agent_index == 0:
                return max_value(state, agent_index, depth)
            else:
                return min_value(state, agent_index, depth)

        def max_value(state, agent_index, depth):
            v = -1e9
            actions = state.getLegalActions(agent_index)
            next_states = [state.generateSuccessor(agent_index, a) for a in actions]
            next_agent_index = (agent_index + 1)%state.getNumAgents()
            values = [value(s, next_agent_index, depth)[0] for s in next_states]
            action = None
            for i in range(len(values)):
                if values[i] > v:
                    v = values[i]
                    action = actions[i]
            return v, action

        
        def min_value(state, agent_index, depth):
            v = 1e9
            actions = state.getLegalActions(agent_index)
            next_states = [state.generateSuccessor(agent_index, a) for a in actions]
            next_agent_index = (agent_index + 1)%state.getNumAgents()
            values = [value(s, next_agent_index, depth)[0] for s in next_states]
            action = None
            #Value is average over all legal actions
            v = 1.0/len(values)*sum(values)
            return v, action

        depth = 0
        v, action = value(gameState, 0, depth)
        return action

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This evaluation function has 5 terms in addition to the nominal score of the state. 
    1) The negative of the number of remaining pellets (to encourage getting the food)
    2) The negative of the remaining power pellets (to encourage getting the power pellets)
    3) +1 if the Manhattan distance between pacman and a ghost is less than that ghost's scare timer (to encourage eating the ghosts)
    4) -100 if Pacman is within Manhattan distance 2 of a ghost that is not scared (to stay alive)
    5) -1 * the Manhattan distance to the closest pellet (to encourage moving efficiently)
    """
    
    "*** YOUR CODE HERE ***"
    caps = currentGameState.getCapsules()
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    GhostPositions = [g.getPosition() for g in GhostStates]
    #Get the food
    score = -1*Food.count()
    score += -1*len(caps)
    #Stay away from ghosts or try to eat them if they're scared
    ghost_dists = [util.manhattanDistance(Pos,gPos) for gPos in GhostPositions]
    for i in range(len(ghost_dists)):
        if ScaredTimes[i] > ghost_dists[i]:
            score += 1
        else:
            if ghost_dists[i] < 2:
                score += -100

    #Get closer to food
    if Food.count() > 0:
        FoodPos = Food.asList()
        min_food_dist = min([util.manhattanDistance(Pos,fPos) for fPos in FoodPos])
        score += -1*min_food_dist
    return score + currentGameState.getScore()

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


class Node():
    """
     This class provides a data structure to store nodes in our search tree.
    """
    node_id = 0 #Unique node ID for debugging 

    def __init__(self, state, action, parent, agent_index=0):
        """
         action: the action taken to arrive at this state
         state: problem-specific state representation
         parent: parent Node
         agent_index: agent modeled by this mode (Pacman only, currently)
        """
        self.state = state
        self.action = action
        self.parent = parent 
        self.agent_index = agent_index
        self.times_explored = 0 #Number of times this node appears in a simulation
        self.num_wins = 0 #Number of simulation wins that include this node
        self.score_sum = 0 #Sum of scores over all simulations involving this node
        self.children = [] #Children expanded in the search tree
        self.node_id = Node.node_id #Unique node ID assigned to this node for debugging
        Node.node_id += 1 
        
    def best_score_selection(self):
        """Returns child with the best average score over simulations"""
        #Should we consider wins?
        #scores = [1.0 * child.num_wins / child.times_explored for child in self.children]
        scores = [1.0 * child.score_sum / child.times_explored for child in self.children]        
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return self.children[chosenIndex]

    def explore_exploit_selection(self, exploit_weight=0.8):
        """Weights random exploration vs. exploitation"""
        if random.random() < exploit_weight:
            #"EXPLOIT"
            scores = [1.0 * child.score_sum / child.times_explored for child in self.children]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        else:
            #print "EXPLORE"
            chosenIndex = random.choice(range(len(self.children))) 
        return self.children[chosenIndex]

    def most_visited_selection(self):
        """Returns child that has been visited the most"""
        scores = [child.times_explored for child in self.children]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return self.children[chosenIndex]

    def get_action(self):
        """After simulations, choose the best action by following the path along the nodes with the best average score over simulationss"""
        best_child = self.best_score_selection()
        return best_child.action

    def gen_children(self, agent_index=0):
        """Generate all possible child nodes in the game tree for the given agent_index"""
        children = []
        legalMoves = self.state.getLegalActions(agent_index)
        for i in range(len(legalMoves)):
            action = legalMoves[i]
            child_state = self.state.generateSuccessor(agent_index, action)
            children.append(Node(child_state, action, parent=self))
        self.children = children

    def print_tree(self, tab=0):
        """Helper function for debugging"""
        print " "*tab + "ID", self.node_id
        if self.parent:
            print " "*tab + "Parent", self.parent.node_id
        else:
            print " "*tab + "ROOT"
        print " "*tab + "Wins", self.num_wins
        print " "*tab + "Score", self.score_sum
        print " "*tab + "Explored", self.times_explored
        print " "*tab + "Children\n"
        for child in self.children:
            child.print_tree(tab+2)
        
class MonteCarloTreeSearchAgent(MultiAgentSearchAgent):
    """
      Monte Carlo Tree Search agent from R&N Chapter 5
    """

    def __init__(self):
        #TODO: Add to command line options
        self.num_simulations = 100 #Number of simulations to perform for each node
        self.steps_allowed = 100 #Number of iterations of MCTS to do per timestep
    
    def getAction(self, gameState):
        """
          Returns the action chosen by MC Tree Search Agent
          Simulations are implemented with all random moves
        """

        def random_transition(state, agent_index):
            """Return a randomly selected (state, action) tuple for the given agent_index
               Return None if there are no possible moves from this state """
            # Collect legal moves and successor states
            legalMoves = state.getLegalActions(agent_index)
            if legalMoves:
                # Choose random action
                chosenAction = random.choice(legalMoves) 
                return state.generateSuccessor(agent_index, chosenAction), chosenAction
            else: #EndState - no more moves
                return None
            
        def select(tree):
            """Selects a leaf node to expand in the search tree"""
            if not tree.children: #Leaf
                return tree
            #TODO: Write a better SELECT method
            #best_child = tree.best_score_selection()
            best_child = tree.explore_exploit_selection()
            return select(best_child)

        def expand(leaf):
            """Expands all children of the leaf node"""
            leaf.gen_children()
                
        def backpropagate(result, node):
            """Update stats of all nodes traversed in current simulation"""
            win, score = result
            node.times_explored +=1
            node.num_wins += win
            node.score_sum += score
            if node.parent is None:
                return
            backpropagate(result, node.parent)

        def simulate(node):
            #TODO: Simulate more realistic action sequences
            """Simulate game until end state starting at a given node and choosing all random actions"""
            agent_index = 1
            state = node.state
            while True:
                while agent_index < state.getNumAgents():
                    if state.isWin() or state.isLose():
                        return state.isWin(), state.getScore()
                    state, _ = random_transition(state, agent_index)
                    agent_index += 1
                agent_index = 0
                
        ####################
        # MC tree search   #
        #                  #
        # 1. Select        #
        # 2. Expand        #
        # 3. Simulate      #
        # 4. Backpropagate #
        ####################
        
        #Instantiate root node
        tree = Node(gameState, action=None, parent=None)
        #Count number of iterations
        counter = 0        
        while counter < self.steps_allowed:
            leaf = select(tree)
            expand(leaf)
            if leaf.children:
                #Should we simulate only some of the children?
                for child in leaf.children:
                    result = simulate(child)
                    backpropagate(result, child)
            else: #End state
                result = leaf.state.isWin(), leaf.state.getScore()
            counter +=1

        #debugging
        #print "GETTING ACTION"
        #tree.print_tree()
        Node.node_id = 0

        #Select action from child with best simulation stats
        return tree.get_action()
