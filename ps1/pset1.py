# -*- coding: utf-8 -*-
"""
CS 182 Problem Set 1: Python Coding Questions - Fall 2022 Due September 27, 2022 at 11:59pm
"""

### Package Imports ###
import heapq
import abc
from tracemalloc import start
from typing import List, Optional, Tuple
### Package Imports ###
import os



#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. All code should be written in python 3.6 or higher to be compatible with the autograder
# 2. Your submission file must be named "pset1.py" exactly
# 3. No additional outside packages can be referenced or called, they will result in an import error
#    on the autograder
# 4. Function/method/class/attribute names should not be changed from the default starter code
#    provided
# 5. All helper functions and other supporting code should be wholly contained in the default
#    starter code declarations provided. Functions and objects from your submission are imported in
#    the autograder by name, unexpected functions will not be included in the import sequence


class Stack:
    """A container with a last-in-first-out (LIFO) queuing policy."""
    def __init__(self):
        self.list = []

    def push(self,item):
        """Push 'item' onto the stack"""
        self.list.append(item)

    def pop(self):
        """Pop the most recently pushed item from the stack"""
        return self.list.pop()

    def isEmpty(self):
        """Returns true if the stack is empty"""
        return len(self.list) == 0

class Queue:
    """A container with a first-in-first-out (FIFO) queuing policy."""
    def __init__(self):
        self.list = []

    def push(self,item):
        """Enqueue the 'item' into the queue"""
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This operation removes the item
          from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        """Returns true if the queue is empty"""
        return len(self.list) == 0

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item has a priority associated with
      it and the client is usually interested in quick retrieval of the lowest-priority item in the
      queue. This data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild
        # the heap. If item already in priority queue with equal or lower priority, do nothing. If
        # item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class SearchProblem(abc.ABC):
    """
    This class outlines the structure of a search problem, but doesn't implement any of the methods
    (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    @abc.abstractmethod
    def getStartState(self) -> "State":
        """
        Returns the start state for the search problem.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def isGoalState(self, state: "State") -> bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getSuccessors(self, state: "State") -> List[Tuple["State", str, int]]:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor, action, stepCost),
        where 'successor' is a successor to the current state, 'action' is the action required to
        get there, and 'stepCost' is the incremental cost of expanding to that successor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getCostOfActions(self, actions) -> int:
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions. The sequence must be
        composed of legal moves.
        """
        raise NotImplementedError


ACTION_LIST = ["UP", "DOWN", "LEFT", "RIGHT"]

class GridworldState():
    """
    Class denoting the values held in a single state of the Gridworld state space.
    """

    location = None
    residencesVisited = []

    def __init__(self, location, residencesVisited):
        self.location = location
        self.residencesVisited = residencesVisited

    def __str__(self) -> str:
        return f'{{ location: {self.location}, residencesVisited: {str(self.residencesVisited)} }}'
    
    def __eq__(self, other) : 
        return self.__dict__ == other.__dict__


class GridworldSearchProblem(SearchProblem):
    """
    Fill in these methods to define the grid world search as a search problem. Actions are of type
    `str`. Feel free to use any data type/structure to define your states though. In the type hints,
    we use "State" to denote a data structure that keeps track of the state, and you can use any
    implementation of a "State" you want.
    """
    
    numResidences = 0
    numRows = 0
    numCols = 0
    matrix = []
    startState = None

    def __init__(self, file):
        """Read the text file and initialize all necessary variables for the search problem"""

        # TODO: Potentially comment this out when submitting
        here = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(here, file))

        # Set state matrix dimensions
        firstLine = f.readline()
        self.numRows = int(firstLine.split()[0])
        self.numCols = int(firstLine.split()[1])

        # Define state matrix
        self.matrix = []
        for _ in range(self.numRows):
            row = list(map(int, f.readline().split()))
            numResidencesInRow = sum([1 for pos in row if pos == 1])
            self.matrix.append(row)
            self.numResidences += numResidencesInRow

        # Define startState
        lastLine = f.readline()
        startRow = int(lastLine.split()[0])
        startCol = int(lastLine.split()[1])
        startValue = self.matrix[startRow][startCol]
        startLocation = (startRow, startCol)
        startResidencesVisited = []
        if startValue == 1:
            startResidencesVisited = [startLocation]
        self.startState = GridworldState(startLocation, startResidencesVisited)

        f.close()

    def getStartState(self) -> GridworldState:
        return self.startState

    def isGoalState(self, state: GridworldState) -> bool:
        return len(state.residencesVisited) == self.numResidences

    def getSuccessors(self, state: GridworldState) -> List[Tuple[GridworldState, str, int]]:
        successors = []

        if state.location[0] - 1 >= 0:
            newLocationRow = state.location[0] - 1
            newLocationCol = state.location[1]
            newLocation = (newLocationRow, newLocationCol)
            newLocationVal = self.matrix[newLocationRow][newLocationCol]
            newResidencesVisited = state.residencesVisited
            if newLocationVal == 1 and newLocation not in state.residencesVisited:
                newResidencesVisited = newResidencesVisited + [newLocation]
            if newLocationVal != -1:
                successors.append((GridworldState(newLocation, newResidencesVisited), "UP", 1))
        if state.location[1] + 1 < self.numCols:
            newLocationRow = state.location[0] 
            newLocationCol = state.location[1] + 1
            newLocation = (newLocationRow, newLocationCol)
            newLocationVal = self.matrix[newLocationRow][newLocationCol]
            newResidencesVisited = state.residencesVisited
            if newLocationVal == 1 and newLocation not in state.residencesVisited:
                newResidencesVisited = newResidencesVisited + [newLocation]
            if newLocationVal != -1:
                successors.append((GridworldState(newLocation, newResidencesVisited), "RIGHT", 1))
        if state.location[0] + 1 < self.numRows:
            newLocationRow = state.location[0] + 1
            newLocationCol = state.location[1] 
            newLocation = (newLocationRow, newLocationCol)
            newLocationVal = self.matrix[newLocationRow][newLocationCol]
            newResidencesVisited = state.residencesVisited
            if newLocationVal == 1 and newLocation not in state.residencesVisited:
                newResidencesVisited = newResidencesVisited + [newLocation]
            if newLocationVal != -1:
                successors.append((GridworldState(newLocation, newResidencesVisited), "DOWN", 1))
        if state.location[1] - 1 >= 0:
            newLocationRow = state.location[0]
            newLocationCol = state.location[1] - 1
            newLocation = (newLocationRow, newLocationCol)
            newLocationVal = self.matrix[newLocationRow][newLocationCol]
            newResidencesVisited = state.residencesVisited
            if newLocationVal == 1 and newLocation not in state.residencesVisited:
                newResidencesVisited = newResidencesVisited + [newLocation]
            if newLocationVal != -1:
                successors.append((GridworldState(newLocation, newResidencesVisited), "LEFT", 1))

        return successors

    def getCostOfActions(self, actions: List[str]) -> int:
        return len(actions) # All actions have a cost of 1


def depthFirstSearch(problem: SearchProblem) -> List[str]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the goal. Make sure to
    implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to understand the search
    problem that is being passed in:

    print("Start:", problem.getStartState()) print("Is the start a goal?",
    problem.isGoalState(problem.getStartState())) print("Start's successors:",
    problem.getSuccessors(problem.getStartState()))
    """

    stack = Stack()
    visited = []
    
    stack.push((problem.getStartState(), []))

    while not stack.isEmpty():
        (currentState, actionsToGetToCurrent) = stack.pop()
        if problem.isGoalState(currentState):
            return actionsToGetToCurrent
        visited.append(currentState)

        successors = problem.getSuccessors(currentState)
        for (state, action, cost) in successors:
            if state not in visited:
                stack.push((state, actionsToGetToCurrent + [action]))

def breadthFirstSearch(problem: SearchProblem) -> List[str]:
    queue = Queue()
    visited = []
    
    queue.push((problem.getStartState(), []))

    while not queue.isEmpty():
        (currentState, actionsToGetToCurrent) = queue.pop()
        if problem.isGoalState(currentState):
            return actionsToGetToCurrent
        visited.append(currentState)

        successors = problem.getSuccessors(currentState)
        for (state, action, cost) in successors:
            if state not in visited:
                queue.push((state, actionsToGetToCurrent + [action]))
    
def nullHeuristic(state: "State", problem: Optional[GridworldSearchProblem] = None) -> int:
    """
    A heuristic function estimates the cost from the current state to the nearest goal in the
    provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def simpleHeuristic(state: "State", problem: Optional[GridworldSearchProblem] = None) -> int:
    """
    This heuristic returns the number of residences that you have not yet visited.
    """
    raise NotImplementedError


def customHeuristic(state: "State", problem: Optional[GridworldSearchProblem] = None) -> int:
    """
    Create your own heurstic. The heuristic should
        (1) reduce the number of states that we need to search (tested by the autograder by counting
            the number of calls to GridworldSearchProblem.getSuccessors)
        (2) be admissible and consistent
    """
    raise NotImplementedError


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[str]:
    """Search the node that has the lowest combined cost and heuristic first.
    This function takes in an arbitrary heuristic (which itself is a function) as an input."""
    "*** YOUR CODE HERE ***"
    raise NotImplementedError



def printGridWorldSuccessors(successors: List[Tuple[GridworldState, str, int]]) -> str:
    print('---------------')
    for successor in successors:
        print(successor[1] + ': ' + str(successor[0]))
    print('---------------\n')

if __name__ == "__main__":
    ### Sample Test Cases ###
    # Run the following statements below to test the running of your program

    # gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case2.txt")

    # assert(gridworld_search_problem.isGoalState(gridworld_search_problem.getStartState()) == False)
    # assert(gridworld_search_problem.getCostOfActions(['LEFT', 'RIGHT', 'DOWN']) == 3)

    # testGridworldState = gridworld_search_problem.getStartState()
    # print(testGridworldState)
    # successors = gridworld_search_problem.getSuccessors(testGridworldState)
    # printGridWorldSuccessors(successors)


    gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case1.txt") # Test Case 1
    print(depthFirstSearch(gridworld_search_problem))
    print(breadthFirstSearch(gridworld_search_problem))
    # print(aStarSearch(gridworld_search_problem))
    
    gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case2.txt") # Test Case 2
    print(depthFirstSearch(gridworld_search_problem))
    print(breadthFirstSearch(gridworld_search_problem))
    # print(aStarSearch(gridworld_search_problem))
    
    gridworld_search_problem = GridworldSearchProblem("pset1_sample_test_case3.txt") # Test Case 3
    print(depthFirstSearch(gridworld_search_problem))
    print(breadthFirstSearch(gridworld_search_problem))
    # print(aStarSearch(gridworld_search_problem))
