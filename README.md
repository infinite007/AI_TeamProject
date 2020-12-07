# AI_TeamProject

This repository contains code for the implementation of Monte Carlo Tree Search algorithm for a PacMan agent.


**How to run Monte Carlo Tree Search:**

cd ./multiagent

python pacman.py -p MonteCarloTreeSearchAgent -l mediumClassic


**Parameters that the Monte Carlo Tree Search agent can take:**

*chooseAlg:* name of the algorithm used to choose the next action for Pacman to take. Below are the currently implemented options:

* *best_combination:* uses a combination of the win rate and average score to the pick the action.
* *best_win:* picks the action with the highest win rate.
* *highest_score:* picks the action with highest average score.
* *most_visited:* picks the action that was visited most while building the search tree.

*dangerZone:* float ranging from [0, 1] representing below what win rate threshold will Pacman be considered in danger.

*earlyStop:* boolean on whether generating the tree can end early if Pacman isn't in danger nor is the action that is going to be selected changing.

*exploreAlg:* name of the algorithm used to decide which action to explore in the search tree. Two different algorithms have been implemented:

* *ucb:* upper confidence bound which balances average score and average win rate by using a provided constant value c.
* *eg:* epsilon greedy search which epsilon percent of the time chooses the action with the highest average score. Otherwise, i randomly picks an action.

*exploreVar:* parameter value used by the explore algorithm to decide which action to explore. Please refer to the above section exploreAlg for what the parameter value does in different explore algorithms. 

*optimism:* while a simulation play out is occurring, what percentage of the ghost's moves will be random.

*panics:* boolean of whether Pacman can't early stop if Pacman's expected win rate is below the threshold.

*randSim:* during a simulation play out whether all the Pacman and ghost actions are random or not.

*reuse:* boolean of whether the previous search tree is reused in the next search.

*simDepth:* positive integer for how many turns each agent takes in a simulation before a heuristic to approximate the results of the game.

*simRelevance:* positive float ranging (0, 1] expressing to what degree does the last simulation impact the expected win rate of Pacman.

*steps:* positive integer of the maximum number of steps to run while building the search tree.

*tillBored:* number of steps of still picking the same action before it stops building the search tree early.

**Default Parameters:**

*chooseAlg:* best_combination

*dangerZone:* 0.2

*earlyStop:* True

*exploreAlg:* ucb

*exploreVar:* 150

*optimism:* 0.2

*panics:* True

*randSim:* False

*reuse:* True

*simDepth:* 10

*simRelevance:* 0.1

*steps:* 500

*tillBored:* 80

**Example Running Monte Carlo Tree Search With Custom Parameters:**

cd ./multiagent

python pacman.py -p MonteCarloTreeSearchAgent -l mediumClassic -a steps=500,reuse=True,simDepth=10,chooseAlg=best_combination,exploreAlg=ucb,exploreVar=150,randSim=False,earlyStop=True,tillBored=80,optimism=0.1,panics=True,simRelevance=0.1,dangerZone=0.2


**Custom Test Cases:**

All the test cases have been modified and expanded. All the below examples use the default parameters and run for 20 games while using a random seed of 38548767.

*q1:* AlphaBetaAgent with better heuristic on smallClassic layout with two random ghosts.

*q2:* ExpectimaxAgent with better heuristic on smallClassic layout with two random ghosts.

*q3:* AlphaBetaAgent with better heuristic on mediumClassic layout with two random ghosts.

*q4:* ExpectimaxAgent with better heuristic on mediumClassic layout with two random ghosts.

*q7:* AlphaBetaAgent with better heuristic on smallClassic layout with two adversarial ghosts.

*q8:* ExpectimaxAgent with better heuristic on smallClassic layout with two adversarial ghosts.

*q9:* AlphaBetaAgent with better heuristic on mediumClassic layout with two adversarial ghosts.

*q10:* ExpectimaxAgent with better heuristic on mediumClassic layout with two adversarial ghosts.

*q11:* MonteCarloTreeSearchAgent on smallClassic layout with two random ghosts.

*q12:* MonteCarloTreeSearchAgent on mediumClassic layout with two random ghosts.

*q13:* MonteCarloTreeSearchAgent on smallClassic layout with two adversarial ghosts.

*q14:* MonteCarloTreeSearchAgent on mediumClassic layout with two adversarial ghosts.

*q15:* MonteCarloTreeSearchAgent on originalClassic layout with four random ghosts.

*q16:* ExpectimaxAgent on originalClassic layout with four random ghosts.


**Modified Files:**


Two files were modified from Project 2: multiAgents.py and multiagentTestClasses.py.

* *multiAgents.py:* Added Node and MonteCarlTreeSearchAgent classes to the file. Also modified MultiAgentSearchAgent, MinimaxAgent, AlphaBetaAgent, and ExpectimaxAgent to add number of nodes, tree depth, and time per move metrics.

* *multiagentTestclasses.py:* Modified EvalAgentTest class to display these new metrics.

**Project Results:**

Included in the project repository is the file "Project Results.xlsx" which includes all the results from all the tests run during this project.