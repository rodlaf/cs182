# -*- coding: utf-8 -*-
"""
CS 182 Problem Set 2: Python Coding Questions - Fall 2022
Due October 11, 2022 at 11:59pm
"""

### Package Imports ###
from operator import truediv
import numpy as np
import random
from termcolor import colored # You can install termcolor using "pip install termcolor" or "conda install termcolor"
from copy import deepcopy
from typing import Tuple
### Package Imports ###


#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. All code should be written in python 3.7 or higher to be compatible with the autograder
# 2. Your submission file must be named "pset2.py" exactly
# 3. No additional outside packages can be referenced or called, they will result in an import error on the autograder
# 4. Function/method/class/attribute names should not be changed from the default starter code provided
# 5. All helper functions and other supporting code should be wholly contained in the default starter code declarations provided.
#    Functions and objects from your submission are imported in the autograder by name, unexpected functions will not be included in the import sequence


##############################################
#### Helper Functions and Data Structures ####
##############################################
# Do not edit - these helper functions and data structures will serve as the starter code needed to pass the auto-grader test cases, feel free
# to skip ahead and scroll down to the bottom where the coding required by student submissions begins

def BackTrackingSearch(GameBoard, SUV_heuristic, ODV_heuristic, iterations, print_result=True):
    """Implements a recursive backtracking search algorithm to solve a Sudoku puzzle by finding values that satisfy all constraints"""
    
    if GameBoard.is_complete(): # Check if the board is complete i.e. no blank cells remain
        if GameBoard.check_board_validity(): # Check that there are no violations of the constraints
            # Only legal moves are allowed to be selected each iteration so we don't have to explictly check board validity each time
            if print_result == True: # Print the results if the user has toggled print_result to True
                print("Board solved!")
                GameBoard.print_board_colored()
            return True, iterations, GameBoard # Return the number of iterations required to find a solution and the completed GameBoard obj
        else:
            return False, iterations, None # If the board is not valid, return False, trigger recursive backtracking
    
    else: # If the board is not complete i.e. there still remains blank cells with legal candidates
        GameBoard.update_unassigned() # Update the legal candidate lists, remove ones that are no longer legal options given the new assigment
        if not GameBoard.fwd_checking(): # Use forward checking to see if this search tree branch will not lead to a valid solution
            return False, iterations, None # If forward checking fails, i.e. returns fasle, trigger recursive backtracking
            
        while len(GameBoard.coordinate_dict)>0: # Loop until there are no remaining entries in the coordinate_dict
            # Call the select-unassigned-value (SUV) heuristic to get the coordinates of which blank tile to fill in next
            var_to_assign = SUV_heuristic(GameBoard.coordinate_dict)
            
            # Call the order-domain-values (ODV) heuristic to get which value of the legal options to assign to this variable
            val_to_assign = ODV_heuristic(var_to_assign, GameBoard.coordinate_dict[var_to_assign], GameBoard)
            
            # If there are no further values to try for this decision variable, return False and trigger recursive backtracking
            if not val_to_assign:
                return False, iterations, None
            
            # Make this assignment to the game board and recursively call the next evaluation step
            new_board = GameBoard.make_assignment(var_to_assign, val_to_assign)
            iterations+=1 # Count the number of assignments performed as the number of iterations
            
            # Recursive backtracking search call
            result, iter_count, solved_board = BackTrackingSearch(new_board, SUV_heuristic, ODV_heuristic, iterations=0, print_result=print_result)
            iterations+=iter_count # Incriment up the iteration count for all iterations made along the branch explored
            if result: # If a solved board is found, then exit the recustion
                return True, iterations, solved_board
        
        # If no valid solution is able to be found exhaustively searching through all possiable candidates for each decision variable, exit
        return False, iterations, None

def solve_board(input_filename, print_result=True):
    """Helper function to run backtracking search an input Sudoku board saved as a comma delimited .csv file"""
    sudoku_game = SudokuGameBoard(filename=input_filename) # Instantiate the game board object
    result, iterations, solved_board = BackTrackingSearch(sudoku_game, MRV_heuristic, ODV_heuristic_bonus, 0, print_result) # Run BTS
    print("\nIterations:", iterations) # Print the number of iterations it took to find a valid solution
    if print_result==True and result == False: # Report if no valid solution was able to be found
        print("No valid solution found")

def print_board_colored(game_board,starting_board,outline=True):
    """Prints the current game_board to the console in a readable format with magenta colored values to indicate assigned numbers"""
    new_val_color='magenta' # Set the color to indicate which values on the board have been filled in vs the starting board
    if outline==True:
        print('-'*25)
    for i in range(9):
        # Print row of numbers - replace 0s with " "
        for k in range(9):
            # Loop through each number in the row
            board_val=game_board[i,k]
            if board_val==0:
                board_val=" "
            # If the same as the starting numbers, leave as black font color
            if board_val==starting_board[i,k]:
                if k==8:
                    print(str(board_val),end=" |\n")
                elif k==0:
                    print("| "+str(board_val),end=" ")
                elif k==2 or k==5:
                    print(board_val,end=" | ")
                else:
                    print(board_val,end=" ")
            
            else: # Else color with a new color to differentiate
                if k==8:
                    print(colored(board_val,new_val_color),end=" |\n")
                elif k==0:
                    print("| "+colored(str(board_val),new_val_color),end=" ")
                elif k==2 or k==5:
                    print(colored(board_val,new_val_color),end=" | ")
                else:
                    print(colored(board_val,new_val_color),end=" ")
        if i==2 or i==5:
            # For the main dividers, print a row of dashes
            if outline==True:
                print('|'+'-'*23+'|')
    if outline==True:
        print('-'*25,end="")

class SudokuGameBoard:
    """Data structure for performing backtracking search to solve a Sudoku game board, tracks board position and legal candidates for each cell"""
    def __init__(self, filename:str=None, game_board=None, starting_board=None):
        # Users can instantiate with either a filename OR both a game_board and starting_board numpy array
        
        if filename and game_board is None and starting_board is None: # If a .csv filename is provided e.g. "Sudoku_Input_Easy1.csv"   
            starting_board = np.genfromtxt(filename, delimiter=',', encoding="utf-8",dtype=int) # Read in the game board from a .csv file        
            assert np.isnan(starting_board).sum() == 0, "NaN entries found in starting board after import"
            assert starting_board.shape == (9,9), "Size of starting board is not (9, 9), got "+str(starting_board.shape)
            starting_board=starting_board.astype(dtype='int') # Convert to int if not already
            self.starting_board = starting_board.copy() # Save the starting board as an attribute, 0 will be used to incidate a blank cell
            self.game_board = starting_board.copy() # The game_board will start off initially the same as the starting_board
            # starting_board is a numpy array representing the starting board before entries have been filled in by the BTS algorithm
            # game_board is a numpy array representing the current board with entried filled in by the BTS algorithm
            
        # Users can alternatively instantiate by passing in a numpy array of size 9x9 for both game_board and starting_board
        elif game_board is not None and starting_board is not None:
            # game_board is a numpy array representing the current values assigned to the board after the algo has started making assignments
            # This needs to be recorded for recursion in backtracking search. starting_board is recorded so that we can print what values were
            # filled in by the program later, it does not serve a functional purpose in the backtracking search algorithm itself
            self.starting_board = starting_board.copy() # Save a copy of the starting board as an attribute
            self.game_board = game_board.copy() # Save a copy of the game_board passed in as an attribute
        
        else: # If input types are not as expected, raise an error
            raise TypeError("Initialization inputs not understood, please provide either a filename or both a game_board and starting_board")
    
        # Create a numpy array of coordinates (i, j) that is 9x9 representing each coordinate pair for each cell on the board
        self.coord_grid = np.array([(i,j) for i in range(9) for j in range(9)], dtype="i,i").reshape(9,9)
        self.all_options = set(list(range(1,10))) # Save a set of all possiable integer values a cell on the game board can be assigned
        
    def get_indexes(self, value:int):
        """Returns a list of coordinates (tuples) locating all occurances of the input value in the game_board numpy array"""
        return self.coord_grid[self.game_board == value].tolist() # Pull out all the coordinates where the array has the input value recorded
    
    def legal_candidates(self, coord:tuple, game_board:np.array)->list:
        """Takes an input coordinate on the board encoded as a length 2 tuple of integers and returns a list of legal candidates based on the constraints"""
        assert len(coord)==2 and type(coord)==tuple, "Corodinate pair input not understood"
        row, col = coord # Separate into a row and col coordinates, indexing starts at 0
        
        # Work on returning a list of the legal candidates for a cell with a given row and col index
        row_values=list(game_board[row,:]) # Pull out values of all cells in this row where this cell is located
        col_values=list(game_board[:,col]) # Pull out values of all cells in this col where this cell is located
        
        # Pull out the values of all cells in the 3x3 sub-block where this cell is located
        row_start = (row//3)*3;row_end = row_start+3 # Compute the start and end row index of the sub-block
        col_start = (col//3)*3;col_end = col_start+3 # Compute the start and end col index of the sub-block
        block_values = self.game_board[row_start:row_end, col_start:col_end].ravel().tolist() # Extract out entries of the 3x3 sub-block

        # unique_vals will tell us which numbers are not available for this cell since they have been used already in the same row/col/sub-block
        unique_vals=set(row_values+col_values+block_values) # Unique values along this row/col/block where this cell intersects
        unique_vals.remove(0) # Remove the blank option which will always be an element since the unassigned cell in question was extracted
        return list(self.all_options - unique_vals) # Return a list of legal integer candidate values for the cell
    
    def make_assignment(self, coord:tuple, value:int):
        """Takes a coordinate on the game board (a tuple) and assigns a value to that cell. Returns a new SudokuGameBoard object with this assignment made
           and updates the coordinate dictionary that holds the legal candidates remaining for each decision variable that have not yet been eliminated
        """
        self.coordinate_dict[coord] = [item for item in self.coordinate_dict[coord] if item!=value] # Update the legal options for this coordinate,
        # by removing the value just assigned since this one is going to be recursively evaluated in further steps and should not be tried again
        sub_gameboard = deepcopy(self) # Create a deep copy of this object for the next recursive iteration to work off so that we can explore this branch of 
        # the search tree and still be able to revert back in the backtracking step if no solution is found down this branch
        sub_gameboard.game_board[coord] = value # Make the assignment to the board and return this new sub-game instance        
        del sub_gameboard.coordinate_dict[coord] # Remove the unassigned variable from the dict of unassigned variables since it has now been assigned a value
        return sub_gameboard # Return the new SudokuGameBoard object that is the same as the previous but now with a new assignment made

    def update_unassigned(self):
        """A method used to generate a set of legal candidates (a list of ints) for all the blank cells on the game board"""
        # If the coordinate dict does not already exist, generate one for this data object
        if not hasattr(self, 'coordinate_dict'):
            # This method generates a dict of empty tiles currently on the board and stores for each of them a list of legal candidates
            coordinate_list = self.get_indexes(0) # Find all the coordinates of the entry 0 on the game_board, which denote empty squares
            self.coordinate_dict = {coord:self.legal_candidates(coord, self.game_board) for coord in coordinate_list}
        else:
            # Otherwise, if one does exist, then update by only removing options which are no longer legal, do NOT add back options already considered
            self.coordinate_dict = {coord:list(set(self.coordinate_dict[coord]) & set(self.legal_candidates(coord, self.game_board))) for coord in self.coordinate_dict}

    def is_complete(self):
        """Checks if the board is complete by checking if any values on the board are still recorded as zero i.e. blank"""
        if not hasattr(self, 'coordinate_dict'):
            self.update_unassigned() # If the coordinate dict does not already exist, generate one for this data object
        
        if len(self.coordinate_dict)==0: # If the coordinate_dict is empty, then there are no remaining zeros on teh board
            return True
        else:
            return False
        
    def check_board_validity(self, verbose=False):
        """Checks if the current board configuration is legal i.e. if any constraints are violated by the current value assignment,
           returns True or False indicating if the rows, columns, and sub-blocks adhere to having no more than 1 instance of [1-9] each
        """
        
        for row in range(self.game_board.shape[0]): # Loop through each row, check that all rows have only 1 instance of values 1 to 9
            row_values = self.game_board[row,:] # Get the values in this row
            vals, counts = np.unique(row_values, return_counts=True) # Compute the multiplicity of each
            counts[vals==0] = 0 # Set the count of zeros to 0, there can be multiple blanks
            exceptions = vals[counts>1] # Identify all values which have more than 1 instance
            if len(exceptions)>0:
                # If there is more than 1 instance of a particular value in any row, print the row and the value
                if verbose:
                    print(colored("Error in row "+str(row)+" - multiple instances of "+str(exceptions),'red'))
                return False
            
        for col in range(self.game_board.shape[1]): # Loop through each col, check that all rows have only 1 instance of values 1 to 9
            col_values = self.game_board[:,col] # Get the values in this col
            vals, counts = np.unique(col_values, return_counts=True) # Compute the multiplicity of each
            counts[vals==0] = 0 # Set the count of zeros to 0, there can be multiple blanks
            exceptions = vals[counts>1] # Identify all values which have more than 1 instance
            if len(exceptions)>0:
                # If there is more than 1 instance of a particular value in any row, print the row and the value
                if verbose:
                    print(colored("Error in col "+str(col)+" - multiple instances of "+str(exceptions),'red'))
                return False
        
        # Check that all 3x3 sub-blocks have at most only 1 instance of each value 1 to 9
        for i in range(0,3): # Loop through sub-block rows
            for j in range(0,3): # Loop through sub-block columns
                block_subset = self.game_board[(3*i):(3+3*i),(3*j):(3+3*j)] # Pull out the values in this sub-block
                block_values = block_subset.ravel().tolist() # Flatten into 1 list
                vals, counts = np.unique(block_values, return_counts=True) # Compute the multiplicity of each
                counts[vals==0] = 0 # Set the count of zeros to 0, there can be multiple blanks
                exceptions = vals[counts>1] # Identify all values which have more than 1 instance
                if len(exceptions)>0:
                    # If there is more than 1 instance of a particular value in any row, print the row and the value
                    if verbose:
                        print(colored("Error in sub-block ("+str(i)+","+str(j)+") - multiple instances of "+str(exceptions),'red'))
                    return False
        
        # Check that all values on the game_board are integer values contained in [0-9], allow for blank tiles to remain on the board
        remaining_unique_vals = set(np.unique(self.game_board)) - self.all_options
        if remaining_unique_vals != set() and remaining_unique_vals != set():
            print("Values found that are not [0-9] integers: "+str(remaining_unique_vals))
            return False
        
        # If not violation of the constraints are found, then return True
        return True
        
    def print_board(self,outline=True):
        """Prints the current game_board to the console in a readable format"""
        if outline==True:
            print('-'*25)
        for i in range(9):
            # Print row of numbers - replace 0s with " "
            p1=[self.game_board[i,k] if self.game_board[i,k]!=0 else " " for k in range(0,3)]
            p2=[self.game_board[i,k] if self.game_board[i,k]!=0 else " " for k in range(3,6)]
            p3=[self.game_board[i,k] if self.game_board[i,k]!=0 else " " for k in range(6,9)]
            if outline==True:
                print(*["|"]+p1+["|"]+p2+["|"]+p3+["|"])
            else:
                print(*p1+p2+p3)
            if i==2 or i==5:
                # For the main dividers, print a row of dashes
                if outline==True:
                    print('|'+'-'*23+'|')
        if outline==True:
            print('-'*25)
             
    def print_board_colored(self,outline=True):
        """Prints the current game_board to the console in a readable format with magenta colored values to indicate assigned numbers"""
        new_val_color='magenta' # Set the color to indicate which values on the board have been filled in vs the starting board
        if outline==True:
            print('-'*25)
        for i in range(9):
            # Print row of numbers - replace 0s with " "
            for k in range(9):
                # Loop through each number in the row
                board_val=self.game_board[i,k]
                if board_val==0:
                    board_val=" "
                # If the same as the starting numbers, leave as black font color
                if board_val==self.starting_board[i,k]:
                    if k==8:
                        print(str(board_val),end=" |\n")
                    elif k==0:
                        print("| "+str(board_val),end=" ")
                    elif k==2 or k==5:
                        print(board_val,end=" | ")
                    else:
                        print(board_val,end=" ")
                
                else: # Else color with a new color to differentiate
                    if k==8:
                        print(colored(board_val,new_val_color),end=" |\n")
                    elif k==0:
                        print("| "+colored(str(board_val),new_val_color),end=" ")
                    elif k==2 or k==5:
                        print(colored(board_val,new_val_color),end=" | ")
                    else:
                        print(colored(board_val,new_val_color),end=" ")
            if i==2 or i==5:
                # For the main dividers, print a row of dashes
                if outline==True:
                    print('|'+'-'*23+'|')
        if outline==True:
            print('-'*25,end="")
    
    
    ###############################################################
    # Question 2.2: Sudoku Backtracking Search - Forward Checking #
    ###############################################################
    
    # Instructions: Fill in the following 2 methods below to perform forward checking: fwd_checking() and the helper method get_related_coords()
    # It is recommended that you fill in get_related_coords() first before completing fwd_checking() as it is a helper function designed to assist.
    # get_related_coords() takes as an input a length 2 tuple of integer coordinates identifying a cell on the game board and should output a list
    # of tuple coordinate pairs that contains the coordinates of all the cells belonging to the same row, column and 3x3 sub-block as the cell
    # designated by the input coordinates. E.g. get_related_coords((0, 0)) should return the coordinates of all the cells in the first row, the 
    # first column and the upper-left sub-block. Be sure to remove duplicates and also do not include the input coordinate in the output. Ordering
    # does not matter. The expected output in this example for (0,0) would be this list of tuple coordinates in any order: 
    # [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    #  (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
    #  (1, 1), (1, 2), (2, 1), (2, 2)]
    
    # Then complete the fwd_checking() method which should enforce arc-consistency and remove candidate options from unassigned variable domains where 
    # logical. For example, if we have 2 unassigned decision variables in the same row and one has a list of possible candidates: [2,4,5] and the other 
    # has a list of possible candidates: [4], then 4 should be removed from the domain of the first unassigned variable since an assignment of 4 there
    # does not leave a valid assignment to the other unassigned variable. Check that for each option of each decision variable, there exists at least 1 
    # other legal option for all other decision variables. If one does not exist, then remove it from the list of possible candidates.    
    
    # The fwd_checking() method should also return True or False indicating if any foreseeable contradictions exist such that no potential solution exists.
    # If True, then the backtracking search algorithm should continue to proceed down the current branch looking for a potential solution. If False,
    # then an inconsistency has been found that we know will lead to a future situation resulting in no valid solution so we should backtrack immediately 
    # and not spend any further iterations down the current path which has been shown to not lead to a valid solution. Do this by checking that there are 
    # no decision variables with zero remaining legal options after enforcing arc-consistency.

    # coordinate_dict is a dictionary of coordinates recorded as a tuple of integers (keys) and a list of possible candidates (values) for the value 
    # of each remaining unassigned variable on the board. e.g. coordinate_dict = {(1,1):[3,4,5], (3,6):[1,5,8,9]} would indicate that there are 2
    # blank cells remaining on the board, one at coordinates (1,1) and another at (3,6) where cells on the game board are indexed starting at 0. For
    # the blank cell located at (1,1) the values [3,4,5] are potential values that might be placed in that cell. This list of potential candidates
    # is derived from legal candidate logical elimination by enforcing the row/col/block constraints and also reduced by removing values that have
    # already been tried which do not lead to a solution or via enforcing arc-consistency.
    
    # Complete the methods below and test your function against the provided test cases.
    # Do not use any additional external libraries not already imported above.    

    def get_related_coords(self, coord: tuple) -> list:
        """A method that takes in a length 2 tuple of int coordinates and returns a list of the coordinates of cells in the constraints of the input coord. I.e. returns
           a list of tuple coordinate pairs of all cells in the same row, column and sub-block i.e. those that are in the input coord cell's constraints
        """
        coords = np.indices((9, 9)).swapaxes(0, 2).swapaxes(0, 1)
        related_coords = []
  
        related_coords += map(tuple, coords[coord[0], :]) # Row coords
        related_coords += map(tuple, coords[:, coord[1]]) # Column coords

        # Block coords
        row_start = (coord[0] // 3) * 3
        row_end = row_start + 3
        col_start = (coord[1] // 3) * 3
        col_end = col_start + 3

        related_coords += map(tuple, coords[row_start:row_end, col_start:col_end].reshape((9, 2)))

        related_coords = list(set(related_coords)) # Remove duplicates
        related_coords.remove(coord) # Remove input coord
        
        return related_coords

    
    def fwd_checking(self):
        """A method to perform forward checking that enforces arc-consistency to reduce the possible legal candidates remaining for each unassigned decision variable
           and also limit the iterations down branches of the search tree that have foreseeable incompatibilities"""
        
        # Enforce arc-consistency by checking that for each option of each decision variable, there exists at least 1 other legal option for all other decision variables
        # e.g. if in a row there are 2 unassigned variables that have [2,3] and [2] as their possiable remaining candidates respectively, then remove 2 from the first
        # list since there is no legal assignment for the second variable if 2 is assigned to the first.
        
        #### YOUR CODE HERE ####
        def revise(coord_1, coord_2) -> bool:
            if len(self.coordinate_dict[coord_2]) == 1 and self.coordinate_dict[coord_2][0] in self.coordinate_dict[coord_1]:
                self.coordinate_dict[coord_1].remove(self.coordinate_dict[coord_2][0])
                return True
            else:
                return False

        queue = []

        for coord_1 in self.coordinate_dict.keys():
            for coord_2 in self.get_related_coords(coord_1):
                if coord_2 in self.coordinate_dict.keys():
                    queue.append((coord_1, coord_2))

        while len(queue) != 0:
            coord_1, coord_2 = queue.pop()
            if revise(coord_1, coord_2):
                for coord in self.get_related_coords(coord_1):
                    if coord in self.coordinate_dict.keys():
                        queue.append((coord_1, coord))

        # Check that all unassigned variables have legal candidates, otherwise return False indicating that there is a conflict in the current board arrangement
        # If an assignment to the board leaves an unassigned variable with no legal options, then the most recent assignment cannot lead to a valid solution
        
        #### YOUR CODE HERE ####
        for coord in self.coordinate_dict:
            if len(self.coordinate_dict[coord]) == 0:
                return False
 
        # If forward checking has passed, then return True to continue down this search tree branch
        return True
    
    
### Sample Test Cases ###
# Run the following assert statements below to test your function, all should run without raising an assertion error 
if __name__ == "__main__":
    test_board = np.array([ [0, 2, 7, 5, 3, 1, 8, 9, 6],
                            [0, 9, 1, 2, 4, 8, 7, 5, 3],
                            [0, 3, 5, 7, 9, 6, 2, 4, 1],
                            [9, 5, 4, 6, 7, 3, 1, 2, 8],
                            [1, 7, 6, 9, 8, 2, 4, 3, 5],
                            [0, 8, 2, 1, 5, 4, 6, 7, 9],
                            [7, 6, 9, 4, 1, 5, 3, 8, 2],
                            [5, 1, 3, 8, 2, 7, 9, 6, 4],
                            [0, 4, 8, 3, 6, 9, 5, 1, 7]])
    
    TestGameBoard = SudokuGameBoard(None, test_board, test_board);TestGameBoard.update_unassigned()
    
    assert set(TestGameBoard.get_related_coords((0,0))) == set([(4, 0), (8, 0), (0, 2), (0, 5), (2, 2), (1, 0), (0, 8), (3, 0), (5, 0), (0, 1),
                                                        (0, 7), (1, 2), (0, 4), (2, 1), (7, 0), (1, 1), (0, 3), (2, 0), (0, 6), (6, 0)])
    assert set(TestGameBoard.get_related_coords((4,4))) == set([(4, 0), (3, 4), (4, 3), (5, 4), (4, 6), (7, 4), (4, 2), (4, 5), (3, 3), (4, 8),
                                                        (5, 3), (2, 4), (0, 4), (6, 4), (4, 1), (4, 7), (3, 5), (5, 5), (8, 4), (1, 4)])
    assert set(TestGameBoard.get_related_coords((2,8))) == set([(2, 2), (1, 6), (0, 8), (2, 5), (6, 8), (4, 8), (8, 8), (2, 4), (0, 7), (2, 1),
                                                        (2, 7), (1, 8), (3, 8),(5, 8), (2, 0), (0, 6), (2, 3), (1, 7), (2, 6), (7, 8)])
    print("All sample test cases for get_related_coords passed!")
    
    assert TestGameBoard.fwd_checking() == True
    assert TestGameBoard.coordinate_dict == {(0,0):[4], (1,0):[6], (2,0):[8], (5,0):[3], (8,0):[2]}
    
    TestGameBoard.coordinate_dict = {(1,1):[], (3,6):[1,5,8,9], (2,8):[1,4,5]}
    assert TestGameBoard.fwd_checking() == False
    TestGameBoard.coordinate_dict = {(1,1):[3], (8,1):[3], (2,8):[1,4,5]}
    assert TestGameBoard.fwd_checking() == False
    TestGameBoard.coordinate_dict = {(1,1):[2,4], (3,6):[1,5,8,9], (2,8):[2,4]}
    assert TestGameBoard.fwd_checking() == True
    
    TestGameBoard.coordinate_dict = {(1,1):[2], (1,6):[2,3,9], (2,6):[3]}
    assert TestGameBoard.fwd_checking() == True
    assert TestGameBoard.coordinate_dict == {(1, 1):[2], (1, 6):[9], (2, 6):[3]}
    print("All sample test cases for forward_checking passed!")


############################################################
# Question 2.3: Sudoku Backtracking Search - MRV Heuristic #
############################################################

# Instructions: Complete the function below to create a minimum remaining value heuristic function. The function should take in a dictionary
# that encodes the coordinate dictionary described above and return a coordinate pair (a length 2 tuple with integer values [0-8]) that designates
# the coordinates of an unassigned variable (i.e. blank square) with minimal remaining potential candidate values. This function will be called 
# during backtracking search to decide which unassigned variable to make a value assignment to next. Selecting a variable with the fewest remaining
# possible candidates tends to lead to a valid solution faster than random choice. In the case of a time where there are multiple unassigned 
# variables with candidate counts that are equal to the minimum, returning any of them is fine. 

# Complete the methods below and test your function against the provided test cases.
# Do not use any additional external libraries not already imported above.    

# Input: A coordinate_dict dictionary detailing the unassigned variables on the board and their potential candidate lists
# Output: A length 2 tuple containing integer values [0-8] designating an unassigned variable with a minimal number of remaining candidates to explore

# Example: If coordinate_dict = {(1,1):[3,4,5], (3,6):[1,5,8,9]} then return (1,1) since this unassigned variable has the fewest remaining candidates (i.e. 3)

def MRV_heuristic(coordinate_dict:dict)->tuple:
    """Minimum remaining value heuristic - a function that should return the coordinate with the fewest remaining possibilities as the coordinate
       to make the next assignment at among all unassigned variables
    """
    # For a given coordinate dict, return the next coordinate that we should fill in, use the heuristic of choosing the one with the fewest possibilities
    #### YOUR CODE HERE ####
    return min(coordinate_dict, key=lambda k: len(coordinate_dict[k]))


### Sample Test Cases ###
# Run the following assert statements below to test your function, all should run without raising an assertion error 
if __name__ == "__main__":
    assert MRV_heuristic({(1,1):[3,4,5], (3,6):[1,5,8,9]}) == (1,1)
    test_coord_dict = {(1,1):[3,4,5], (3,6):[1,5,8,9], (2,8):[1,4,5]}
    assert MRV_heuristic(test_coord_dict) == (1,1) or MRV_heuristic(test_coord_dict) == (2,8)
    test_coord_dict={(0, 2): [8, 1, 4, 5],
                     (0, 3): [8, 1, 5, 9],
                     (0, 4): [8, 1, 4, 9],
                     (0, 5): [9, 5],
                     (0, 6): [1, 3, 4, 5, 6, 8, 9],
                     (0, 7): [1, 4, 5, 6, 8],
                     (0, 8): [1, 3, 5, 6, 8, 9],
                     (1, 0): [8, 1, 4, 5],
                     (1, 1): [1, 4, 5],
                     (1, 4): [1, 2, 4, 7, 8],
                     (1, 6): [1, 2, 4, 5, 8],
                     (1, 7): [8, 1, 4, 5],
                     (1, 8): [8, 1, 5],
                     (2, 0): [1, 4, 5, 6, 8],
                     (2, 2): [8, 1, 4, 5],
                     (2, 3): [1, 2, 5, 8, 9],
                     (2, 4): [1, 2, 4, 8, 9],
                     (2, 6): [1, 2, 4, 5, 6, 8, 9],
                     (2, 8): [1, 5, 6, 8, 9],
                     (3, 0): [1, 4, 5, 7],
                     (3, 2): [1, 2, 4, 5, 7],
                     (3, 4): [1, 9, 7],
                     (3, 5): [9, 5, 7],
                     (3, 6): [1, 4, 5],
                     (3, 8): [1, 5]}
    
    assert MRV_heuristic(test_coord_dict) == (3, 8) or MRV_heuristic(test_coord_dict) == (0, 5)
    print("All sample test cases for MRV_heuristic passed!")
    


####################################################################
# Question 2.5: Sudoku Backtracking Search - ODV Heuristic (Bonus) #
####################################################################

# Instructions: Complete the ODV_heuristic_bonus() function for a chance to win 5 bonus points on this problem set. Your function should select a 
# value from the list of candidates to assign to the decision variable identified and return None if there are no candidates to choose from. 
# Selecting a value that is more likely to lead to a solution will result in an overall faster average runtime across many test cases. The top 5 
# submissions by total number of iterations across a set of hidden test cases will earn 5 bonus points on this problem set. You also have access 
# to the SudokuGameBoard data structure and coordinate of the unassigned decision variable being assigned a value.

# Do not use any additional external libraries not already imported above.    

# Input: An coordinate tuple designating the unassigned variable being assigned a value. The candidate list of integer values available for assignment
# to this decision variable. The SudokuGameBoard data object for the current game along this branch of backtracking search.

# Output: The integer value to assign to this decision variable or None if there are no candidates in the input candidate_list.

# Example: coord = (1, 5) and candidate_list = [1,6,8] one might return 8 or 6 depending on what your heuristic tells you is the best choice among the
# possible candidate values for this unassigned variable
# Example: coord = (5, 1) and candidate_list = [], then return None

### Bonus Question ###
def ODV_heuristic_bonus(coord:tuple, candidate_list:list, GameBoard:SudokuGameBoard)->Tuple[int, None]:
    """For a given unassigned variable candidate list, make a selection of which value should be assigned next. If candidate_list is empty, return None"""
    
    #### YOUR CODE HERE ####
    # Replace the code below with your own
    if len(candidate_list)>0:
        return random.choice(candidate_list) # Use random selection among the choices present as the default if no heuristic provided
    else:
        return None
    
    #### YOUR CODE HERE ####
     

#################################################################
# Sample Test Cases for the Overall Backtracking Search Program #
#################################################################
# Run the following function calls below, if everything it working correctly, backtracking search will solve each of these boards and print out
# the solution with colored entries where values have been assigned by the algorithm
if __name__ == "__main__":
    ##### Easy Test Cases #####
    solve_board("Sudoku_Input_Easy1.csv")
    solve_board("Sudoku_Input_Easy2.csv")
    solve_board("Sudoku_Input_Easy3.csv")
    solve_board("Sudoku_Input_Easy4.csv")
    
    ##### Medium Test Cases ####
    solve_board("Sudoku_Input_Medium1.csv")
    solve_board("Sudoku_Input_Medium2.csv")
    solve_board("Sudoku_Input_Medium3.csv")
    solve_board("Sudoku_Input_Medium4.csv")
    
    ##### Blank Board Test Case #####
    blank_board = np.array([[0 for i in range(9)] for j in range(9)]);sudoku_game = SudokuGameBoard(None, blank_board, blank_board)
    result, iterations, solved_board = BackTrackingSearch(sudoku_game, MRV_heuristic, ODV_heuristic_bonus, 0, True)
    print("\nIterations:", iterations)





############################################
# Question 5.1: Sudoku Integer Programming #
############################################

import cvxpy as cp
import time

# Instructions: Use the "cvxpy" optimization package to create a function for solving Sudoku boards by applying integer programming.
# You may need to use "pip install cvxpy==1.1.13" and "pip install cvxopt==1.2.6" in order for the code to run locally. Fill in the 
# Sudoku_Solver_IP() function below and test your implementation using the sample test cases included below. Do not edit solve_board_IP()
# as this is a helper function that will run your solution and return the results. Your function should return a tuple containing 3 elements.
# First, a list of constraints. Second, an objective function. Third, a list containing the 9x9 decision variable grids. The input will be 
# a 9x9 numpy array of integer denoting a starting Sudoku board where 0s indicate blank cells.

# NOTE: We've found that cvxpy does not always handle equality constraints well, instead we recommend using inequalities i.e. >= and <= over ==

# Do not use any additional external libraries not already imported above.    

# Helper function, DO NOT EDIT
def solve_board_IP(input_filename, print_result=True):
    """Helper function that reads in a Sudoku board from a .csv file and solves it using the integer programming functionalities of cvxpy"""
    ### Data Processing ###
    starting_board = np.genfromtxt(input_filename, delimiter=',', encoding="utf-8",dtype=int) # Read in the game board from a .csv file        
    assert np.isnan(starting_board).sum() == 0, "NaN entries found in starting board after import" # Data validation
    assert starting_board.shape == (9,9), "Size of starting board is not (9, 9), got "+str(starting_board.shape)
    starting_board = starting_board.astype(dtype='int') # Convert to int if not already

    ### IP Solver ###
    constraints, objective, assignment = Sudoku_Solver_IP(starting_board) # Pass in the starting board into the IP Sudoku solver
    IP_obj = cp.Problem(objective, constraints) 
    IP_obj.solve(solver=cp.GLPK_MI, verbose=False) # Solve the IP using cvxpy
        
    solved_board = np.stack([assignment[k].value*(k+1) for k in range(9)]).sum(axis=0) # Flattend to a 9x9 grid of values
    solved_board = solved_board.astype(int) # Convert to int data type
    
    if print_result==True: # Print the solved board if set to true to visualize the result
        print("\nBoard Solved!")
        print_board_colored(solved_board,starting_board,outline=True)
    return solved_board


def Sudoku_Solver_IP(starting_board: np.array):
    """Function that returns a list of constraints, an objective function and a decision variable object given a starting Sudoku board"""
    decision_variables = [cp.Variable((9,9),boolean=True) for i in range(9)] # Create a set of decision variables as a set of 9 grids
    # each of size 9x9. Each grid represents an array of bool variables indicating if in the cell (i,j) the kth integer is present e.g.
    # if decision_variables[4][2,3]==1 then the [2,3] entry of the regular 9x9 Sudoku board will contain a 5 since 4+1 = 5 and we index
    # starting at 0. The set of all [2,3] entries across all 9 of the 9x9 grids stored in the decision_variables list describes which
    # value is stored in the [2,3] cell of the regular 9x9 Sudoku board. 3d variable objects are not allowed in cvxpy
    # Hint: Use sometimes cvxpy can be a bit picky, try using inequalities instead of equality constraints if you run into errors
            
    constraints=[] # A list to hold the constraints for this integer program
    
    #### YOUR CODE HERE ####
    
    # Create an objective function for this IP, set it to maximize the sum of all the binary decision variables 
    
    objective=1 #### YOUR CODE HERE #### Replace 1 with an actual objective
    
    return (constraints, objective, decision_variables)


### Sample Test Cases ###
# Run the following assert statements below to test your function, all should run without raising an assertion error 
if __name__ == "__main__":
    solved_board1 = np.array([[4, 2, 7, 5, 3, 1, 8, 9, 6],
                              [6, 9, 1, 2, 4, 8, 7, 5, 3],
                              [8, 3, 5, 7, 9, 6, 2, 4, 1],
                              [9, 5, 4, 6, 7, 3, 1, 2, 8],
                              [1, 7, 6, 9, 8, 2, 4, 3, 5],
                              [3, 8, 2, 1, 5, 4, 6, 7, 9],
                              [7, 6, 9, 4, 1, 5, 3, 8, 2],
                              [5, 1, 3, 8, 2, 7, 9, 6, 4],
                              [2, 4, 8, 3, 6, 9, 5, 1, 7]])
    
    solved_board2 = np.array([[6, 3, 7, 2, 4, 1, 9, 5, 8],
                              [5, 4, 9, 3, 6, 8, 7, 1, 2],
                              [2, 1, 8, 5, 9, 7, 3, 6, 4],
                              [8, 6, 3, 7, 2, 5, 1, 4, 9],
                              [4, 2, 1, 9, 8, 6, 5, 3, 7],
                              [9, 7, 5, 1, 3, 4, 2, 8, 6],
                              [7, 8, 2, 6, 1, 3, 4, 9, 5],
                              [3, 9, 6, 4, 5, 2, 8, 7, 1],
                              [1, 5, 4, 8, 7, 9, 6, 2, 3]])
    
    solved_board3 = np.array([[6, 2, 1, 3, 4, 9, 5, 8, 7],
                              [5, 4, 7, 8, 2, 1, 3, 6, 9],
                              [9, 3, 8, 6, 7, 5, 1, 4, 2],
                              [1, 8, 9, 2, 3, 6, 7, 5, 4],
                              [3, 5, 4, 1, 8, 7, 9, 2, 6],
                              [2, 7, 6, 9, 5, 4, 8, 1, 3],
                              [8, 9, 5, 4, 6, 3, 2, 7, 1],
                              [7, 6, 3, 5, 1, 2, 4, 9, 8],
                              [4, 1, 2, 7, 9, 8, 6, 3, 5]])
    
    solved_board4 = np.array([[1, 2, 8, 7, 4, 6, 3, 5, 9],
                              [9, 7, 3, 1, 5, 2, 4, 6, 8],
                              [5, 4, 6, 8, 9, 3, 2, 1, 7],
                              [7, 9, 2, 5, 6, 8, 1, 3, 4],
                              [3, 6, 1, 2, 7, 4, 9, 8, 5],
                              [8, 5, 4, 9, 3, 1, 6, 7, 2],
                              [2, 1, 7, 3, 8, 9, 5, 4, 6],
                              [6, 3, 5, 4, 2, 7, 8, 9, 1],
                              [4, 8, 9, 6, 1, 5, 7, 2, 3]])
    
    assert (solved_board1 == solve_board_IP("Sudoku_Input_Easy1.csv")).all()
    assert (solved_board2 == solve_board_IP("Sudoku_Input_Easy2.csv")).all()
    assert (solved_board3 == solve_board_IP("Sudoku_Input_Medium3.csv")).all()
    assert (solved_board4 == solve_board_IP("Sudoku_Input_Medium4.csv")).all()
    print("\nAll sample test cases for solve_board_IP passed!")


####                    ####
#### Runtime Comparison ####
####                    ####

# Instructions: Compare the runtime of the backtracking-based solver vs the IP-based solver using the following test cases. Which one is faster? Include
# the approximate runtime of your 2 Sudoku solvers in your PDF solutions write up. (Looking for approximate values and the appropriate comparsion between them)
if __name__ == "__main__":
    test_cases = ["Sudoku_Input_Easy1.csv","Sudoku_Input_Easy2.csv","Sudoku_Input_Easy3.csv","Sudoku_Input_Easy4.csv",
                  "Sudoku_Input_Medium1.csv","Sudoku_Input_Medium2.csv","Sudoku_Input_Medium3.csv","Sudoku_Input_Medium4.csv"]
    
    # Back-tracking Search based appraoch
    start_time_IP = time.time()
    for test_case in test_cases:
        solved_board = solve_board(test_case,print_result = False)
    end_time_IP = time.time()
    print("Runtime of Backtracking Search-based Sudoku Solver:",end_time_IP-start_time_IP)
    
    # Integer Programming-based Search based appraoch
    start_time_IP = time.time()
    for test_case in test_cases:
        solved_board = solve_board_IP(test_case,print_result = False)
    end_time_IP = time.time()
    print("Runtime of IP-based Sudoku Solver:",end_time_IP-start_time_IP)
    

##########################################
# Question 5.2: Minimum Makespan Problem #
##########################################

# Instructions: Use the "cvxpy" optimization package to create a function for solving min makespan integer programming problems given an
# input list of task_times and an integer K denoting the number of machines. Complete the Min_Makespan_Solver() function and test your 
# implementation using the sample test cases included below. Your function should return a tuple that contains 1). overall min makespan
# i.e. the objective function value for your optimal assignment and 2). a N by K numpy array of 0s and 1s (hint, use .value on a cp.Variable() 
# obj) with assignments made such that the max makespan is minimized across all machines where each of the N rows represents each task and 
# each of the K columns represents the assignment of those tasks to a machine.

# Do not use any additional external libraries not already imported above.    

def Min_Makespan_Solver(task_times:list, K:int)->tuple:
    """Function that returns the objective function value (i.e. the makespan) and an optimal assignment of tasks to machines for an input 
    min makespan problem, solves using integer programming"""
    
    #### YOUR CODE HERE ####
    pass


### Sample Test Cases ###
# Run the following assert statements below to test your function, all should run without raising an assertion error 
if __name__ == "__main__":
    task_times = [5,10,2,3,4,5,6] # The time for each required for each of the N tasks
    K = 3 # The number of machines
    
    response = Min_Makespan_Solver(task_times, K)
    assert response[0] == 12 # Check the objective function value
    assert (response[1].sum(axis=1) == np.ones(len(task_times))).all() # Check that the assignment is valid i.e. only 1 machine per task
    
    task_times = [25,5,1,3,1,2,8.5,7.2,5,10,2,3,4,5,6] # The time for each required for each of the N tasks
    K = 3 # The number of machines
    response = Min_Makespan_Solver(task_times, K)
    assert response[0] == 29.5 # Check the objective function value
    assert (response[1].sum(axis=1) == np.ones(len(task_times))).all() # Check that the assignment is valid i.e. only 1 machine per task
    print("All sample test cases for Min_Makespan_Solver passed!")
