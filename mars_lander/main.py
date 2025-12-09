import numpy as np
from parameter import *
import random
from cma import CMA

def make_board(link):
    pass

def simulation(gene, board):
    return random.uniform()

def fitness(gene):
    # 強加法性が成り立ってた方が良くないか?
    board1 = make_board('/test/case1.txt')
    board2 = make_board('/test/case2.txt')
    board3 = make_board('/test/case3.txt')
    board4 = make_board('/test/case4.txt')
    board5 = make_board('/test/case5.txt')

    pass


optimizer = CMA(mean=np.zeros(input_size), sigma=0.5)

for generation in range(max_generation):
    solutions = []
    for _ in range(optimizer.population_size):
        gene = optimizer.ask()
        value = fitness(gene)
        solutions.append((gene, value))
    optimizer.tell(solutions)
    

