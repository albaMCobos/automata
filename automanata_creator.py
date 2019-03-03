from automata import Automaton
import pickle


def create_automaton_one():
    A = Automaton(7, ['a', 'b', 'c'])
    A.redefine_states_names(['S','A','B','C','D','E','F'])
    #S:0,A:1,B:2,C:3,D:4,E:5,F:6
    A.add_initial_state('S')
    A.add_final_state(['F'])
    
    # S:0 transition
    A.add_move(0, 1, 'a')
    A.add_move(0, 4, 'a')
    A.add_move(0, 4, 'b')
    A.add_move(0, 3, 'b')

    # A:1 transition

    # B:2 transition
    A.add_move(2, 2, 'b')
    A.add_move(2, 1, 'b')
    A.add_move(2, 6, 'b')

    # C:3 transition

    # D:4 transition
    A.add_move(4, 3, 'a')
    A.add_move(4, 5, 'c')

    # E:5 transition
    A.add_move(5, 6, 'b')

    # F:6 transition

    pickle.dump( A, open( "A.p", "wb" ) )

    return A

def create_automaton_two():
    A2 = Automaton(14, ['a', 'b'])
    A2.add_initial_state('q0')
    A2.add_final_state(['q7'])
    
    # q0 transition
    A2.add_move(0, 1, 'a')
    A2.add_move(0, 8, 'b')
    
    # q1 transition
    A2.add_move(1, 2, 'a')
    
    # q2 transition
    A2.add_move(2, 3, 'b')
    
    # q3 transition
    A2.add_move(3, 4, 'a')
    A2.add_move(3, 6, 'a')
    A2.add_move(3, 6, 'b')
    
    # q4 transition
    A2.add_move(4, 5, 'b')
    
    # q5 transition
    A2.add_move(5, 4, 'a')
    A2.add_move(5, 6, 'a')
    
    # q6 transition
    
    # q7 transition
    A2.add_move(7, 6, 'a')
    A2.add_move(7, 13, 'b')

    
    # q8 transition
    A2.add_move(8, 9, 'b')
    
    # q9 transition
    A2.add_move(9, 10, 'a')
    
    # q10 transition
    A2.add_move(10, 11, 'b')
    A2.add_move(10, 13, 'b')
    
    # q11 transition
    A2.add_move(11, 12, 'a')
    
    # q12 transition
    A2.add_move(12, 11, 'b')
    A2.add_move(12, 13, 'b')
    
    # q13 transition

    pickle.dump( A2, open( "A2.p", "wb" ) )

    return A2

def create_automaton_three():
    A3 = Automaton(5, ['a', 'b'])
    A3.add_initial_state('q0')
    A3.add_final_state(['q2'])
    
    # q0 transition
    A3.add_move(0, 1, 'a')
    A3.add_move(1, 2, 'b')

    A3.add_move(4, 3, 'b')
    A3.add_move(3, 4, 'b')
    
    

    pickle.dump( A3, open( "A3.p", "wb" ) )

    return A3


create_automaton_one()
create_automaton_two()
create_automaton_three()