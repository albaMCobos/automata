import random
import numpy as np
import pandas as pd
import pickle


class Automaton:

    def __init__(self, number_of_states, alphabet):

        self.__NULL = 'O'  # Pythonic?
        self.number_of_states = number_of_states
        # TODO change 'O' initialization
        self.matrix = [[[] for y in range(number_of_states)]
                       for x in range(number_of_states)]
        self.states = np.array([f"q{y}" for y in range(number_of_states)])
        # maybe name _alphabet? pythonic?
        self.alphabet = alphabet
        self.initial_states = None
        self.final_states = []
    ########## AUTOMATON DEFINITION ############

    def redefine_states_names(self, names):
        if len(names) == len(self.states):
            self.states = np.array([y for y in names])
        else:
            raise ValueError(f"Error : {names} is not of the same size as the actuals states.")

    @staticmethod
    def random_moves_automaton(number_of_states, alphabet):

        rma = Automaton(number_of_states, alphabet)
        rma.matrix = np.array([[random.choice(alphabet) for y in range(
            number_of_states)] for x in range(number_of_states)])
        return rma

    def display_matrix(self):
        parapandas = np.array(self.matrix)

        return pd.DataFrame(
            data=parapandas,
            index=self.states,
            columns=self.states)
    # Check symbol name
    def is_on_alphabet(self, symbol):

        return symbol in self.alphabet
    # Add transition check if its on the dictonary

    def add_move(self, origin, destiny, transition):
        if self.is_on_alphabet(transition):
            self.matrix[origin][destiny].append(transition)
            #self.matrix[origin,destiny] = transition
        else:
            raise ValueError(f"Error : {transition} movement is not on the alphabet.")

    def delete_move(self, origin, destiny):

        self.matrix[origin, destiny] = self.__NULL

    def delete_state(self, state):
        #eliminate all the input and output of the state
        for i in range(self.number_of_states):
            self.delete_move(state,i)
            self.delete_move(i,state)

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
        
        
        vector =[]      # Set Si-1
        vector2 =[0]    # Set Si,the state 0 is always the initial state
        while(vector != vector2):   #stop when we can´t add more accessible states
            vector = vector2
            for AS in vector2:
                vector2=list(set(vector2+(self.moves_of_the_state(AS))))      #A union of all the accessible sates with its next move
        return vector2
    
    def state_coaccesible(self,state):
        """
        Check if a state is coaccesible this means that a final state can be reach from this state
        """
        # TODO
        return None

    def is_empty(self):#not completed
        empty=False
        for i in self.states_accesibles():
            if self.final_states.__contains__(i):       #Have to find the way to compare i such a integer with final_states
                empty = True
                break
        return empty
        # TODO
        return None

    def is_infinite(self):
        """
        Check if the language accepted by the automaton is infinite
        """
        # TODO
        return None
    ######################################################
    ################ AUTOMATON LOAD ##########################
    @staticmethod
    def load_automaton(path):

        return pickle.load(open(path, "rb"))
    ######################################################
