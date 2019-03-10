import random
import numpy as np
import pandas as pd
import pickle


class Automaton:

    def __init__(self, number_of_states, alphabet):
        self.epsilon = 'E'
        self.__NULL = 'O'  # Pythonic?
        self.number_of_states = number_of_states
        # TODO change 'O' initialization
        self.matrix = [[[] for y in range(number_of_states)]
                       for x in range(number_of_states)]
        self.states = np.array([f"q{y}" for y in range(number_of_states)])
        # maybe name _alphabet? pythonic?
        self.alphabet = alphabet + [self.epsilon]
        self.initial_states = None
        self.final_states = []
    ########## AUTOMATON DEFINITION ############

    def redefine_states_names(self, names):
        if len(names) == len(self.states):
            self.states = np.array([y for y in names])
        else:
            raise ValueError(
                f"Error : {names} is not of the same size as the actuals states.")

    @staticmethod
    def random_moves_automaton(number_of_states, alphabet):

        rma = Automaton(number_of_states, alphabet)
        rma.matrix = np.array([[random.choice(alphabet) for y in range(
            number_of_states)] for x in range(number_of_states)])
        return rma

    def display_matrix(self):
        parapandas = np.array(self.matrix)
        try:
            return pd.DataFrame(
                data=parapandas,
                index=self.states,
                columns=self.states)
        except:
            return "The matrix is empty"
    # Check symbol name

    def state_index(self, state):
        """
        Gives the index of a state example q0 -> 0
        """
        index, = np.where(self.states == state)
        return index[0]

    def is_on_alphabet(self, symbol):

        return symbol in self.alphabet
    # Add transition check if its on the dictonary

    def add_move(self, origin, destiny, transition):
        if self.is_on_alphabet(transition):
            self.matrix[origin][destiny].append(transition)
            # self.matrix[origin,destiny] = transition
        else:
            raise ValueError(
                f"Error : {transition} movement is not on the alphabet.")

    def delete_move(self, origin, destiny):

        self.matrix[origin][destiny] = []

    def delete_state(self, state):
        # eliminate all the input and output of the state
        for i in range(self.number_of_states):
            self.delete_move(state, i)
            self.delete_move(i, state)

    def add_initial_state(self, initial_states):
        # TODO check if its inside the states
        # TODO check if its already one
        self.initial_states = initial_states

    def delete_initial_state(self, initial_states):

        self.initial_states = None

    def add_final_state(self, final_states):
        # TODO check if its inside the states
        # TODO fix problems with list and no list
        self.final_states.extend(final_states)

    def delete_final_state(self, final_states):

        self.final_states.remove(final_states)
    ######################################################
    # AUTOMATON INFO ###########################º

    def moves_of_the_state(self, state):

        return [idx for idx, x in enumerate(self.matrix[state]) if x]

    def moves_to_the_state(self, state):

        return [idx for idx in range(self.number_of_states) if len(self.matrix[idx][state]) != 0]

    def states_accesibles(self):

        vector = []      # Set Si-1
        vector2 = [0]    # Set Si,the state 0 is always the initial state
        while(vector != vector2):  # stop when we can´t add more accessible states
            vector = vector2
            for AS in vector2:
                # A union of all the accessible sates with its next move
                vector2 = list(set(vector2+(self.moves_of_the_state(AS))))
        return vector2

    def states_coaccesibles(self):
        """
        Check if a state is coaccesible this means that a final state can be reach from this state
        """
        vector = []                      # Set Si-1
        # Set final states of the automaton
        vector2 = [self.state_index(x) for x in self.final_states]
        while(vector != vector2):  # stop when we can´t add more accessible states
            vector = vector2
            for CS in vector2:
                # A union of all the accessible sates with its next move
                vector2 = list(set(vector2+(self.moves_to_the_state(CS))))
        return vector2
    
    def usefull_states(self):
        accessible_states = set(self.states_accesibles())
        coaccessible_states = set(self.states_coaccesibles())

        return list((accessible_states & coaccessible_states))

        return True

    def is_cycle_present_helper(self, v, visited, on_stack):
        """Return True if the DFS traversal starting at vertex v detects a
        cycle. Uses set visited to keep track of nodes that have been visited. Uses
        set on_stack to keep track of nodes that are 'on the stack' of the recursive
        calls."""
        if v in on_stack:
            return True
        on_stack.add(v)
        for dest in self.moves_of_the_state(v):
            if dest not in visited:
                if self.is_cycle_present_helper( dest, visited, on_stack):
                    return True
        on_stack.remove(v)
        visited.add(v)
        return False

    def is_cycle_present(self):
        """Return True if cycle is present in the graph."""
        on_stack = set()
        visited = set()
        for v in [self.state_index(x) for x in self.states]:
            if v not in visited:
                if self.is_cycle_present_helper( v, visited, on_stack):
                    return True
        return False

    def is_infinite(self):
        """
        Check if the language accepted by the automaton is infinite
        """

        self.reduced_automaton()
        return self.is_cycle_present()
    ######################################################
    ################ AUTOMATON LOAD ##########################

    @staticmethod
    def load_automaton(path):

        return pickle.load(open(path, "rb"))
    ######################################################


if __name__ == "__main__":
    A3 = Automaton.load_automaton("A3.p")    
    A2 = Automaton.load_automaton("A2.p")
    A = Automaton.load_automaton("A.p")

